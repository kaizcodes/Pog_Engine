"""Shared tunables for the VOD highlight pipeline.

Imported by analyze_highlights_emotion.py, OrganizeVODAndFixSRT_Emotion.py, and
isolate_vocals.py. Machine-specific paths (whisper.cpp install, emotion-model
dir, gallery folder) are NOT here - they stay hardcoded in the scripts that
use them, since they're tied to this machine, not tunable per-run. Everything
below can still be overridden per-run via env var without editing this file
(same os.environ.get pattern the original script used).
"""

import os
import re
import subprocess
from pathlib import Path


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no"}


def _env_list(name, default):
    """Read a newline- or semicolon-separated list from the environment."""
    raw = os.environ.get(name)
    values = default if raw is None else re.split(r"[\r\n;]+", raw)
    unique: list[str] = []
    seen: set[str] = set()
    for item in values:
        item = item.strip()
        key = item.casefold()
        if item and key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


# --- Ollama models ---------------------------------------------------------
# JUDGE_MODEL is separate from MODEL: discovery just reads a transcript chunk
# and proposes candidates; verify/judge/titling need more careful structured
# reasoning over shorter, denser prompts.
# qwen3:8b is the long-context discovery role; qwen3.5:9b-q4_K_M fits a
# 10GB 3080 as the judge role. qwen3.6:35b-a3b needs partial CPU offload,
# so its preset uses a smaller JUDGE_NUM_CTX to leave memory for KV cache.
MODEL = os.environ.get("HIGHLIGHT_MODEL", "qwen3:8b")
JUDGE_MODEL = os.environ.get("HIGHLIGHT_JUDGE_MODEL", "qwen3.5:9b-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
# qwen3.5 moved thinking control to Ollama's renderer/parser instead of the
# old "/no_think"-in-prompt Jinja template. Confirmed Ollama 0.17.7 bug
# (ollama/ollama#14793): /api/generate ignores think:false for qwen3.5, so
# the model burns the whole num_predict budget on hidden reasoning and
# "response" comes back empty. /api/chat with top-level think:false works.
# Any JUDGE_MODEL call needing thinking OFF (title_audio_candidates,
# verify_candidates, run_judge_batch) must use this URL - see
# ollama_generate(..., url=...). run_judge_batch used to be the exception
# (deliberately thinking ON for comparative ranking), but newer Ollama
# returns qwen3.5's reasoning in a separate `thinking` field that eats the
# whole num_predict budget, leaving "response" empty - so judging needs it
# off too.
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_RETRIES = _env_int("OLLAMA_RETRIES", 2)
OLLAMA_RETRY_BACKOFF_SECONDS = _env_float("OLLAMA_RETRY_BACKOFF_SECONDS", 5)
# Ollama is normally a user-launched tray app; when the user forgets to start
# it, stage 5 used to die on the first ConnectionError with no recovery.
# ensure_ollama_running() in analyze_highlights_emotion.py now boots it
# automatically (ollama serve) when a call gets a connection-level failure,
# then waits for /api/version to answer before retrying. Only fires for a
# local server - a remote OLLAMA_URL can't be fixed by spawning a process.
# OLLAMA_SERVE_READY_TIMEOUT_SECONDS bounds how long a stage waits for a
# freshly started server to accept connections (cold start is ~1-3s).
OLLAMA_SERVE = os.environ.get("OLLAMA_SERVE", "ollama")
OLLAMA_START_ON_CONNECTION_ERROR = _env_bool("HIGHLIGHT_OLLAMA_START_ON_CONNECTION_ERROR", True)
OLLAMA_SERVE_READY_TIMEOUT_SECONDS = _env_float("HIGHLIGHT_OLLAMA_SERVE_READY_TIMEOUT_SECONDS", 60)
# Ollama's `options.num_ctx` is sent with each API request, so keep separate
# budgets for the long transcript discovery prompt and the shorter judge-side
# prompts. Discovery needs the larger window; qwen3.6's judge preset lowers
# the judge window to reduce memory pressure.
DISCOVERY_NUM_CTX = _env_int("HIGHLIGHT_DISCOVERY_NUM_CTX", 8192)
JUDGE_NUM_CTX = _env_int("HIGHLIGHT_JUDGE_NUM_CTX", 8192)

# --- Output size / selection ------------------------------------------------
TOP_N = _env_int("HIGHLIGHT_TOP_N", 50)
JUDGE_POOL_SIZE = _env_int("HIGHLIGHT_JUDGE_POOL_SIZE", 100)
# Trimmed to this size before verification, not after - only JUDGE_POOL_SIZE
# ever reach judging, so verifying much more than that wastes LLM calls on
# candidates that were never going to make the cut.
VERIFY_POOL_SIZE = _env_int("HIGHLIGHT_VERIFY_POOL_SIZE", int(JUDGE_POOL_SIZE * 1.5))
VERIFY_BATCH_SIZE = _env_int("HIGHLIGHT_VERIFY_BATCH_SIZE", 10)
# A batch that fails to parse keeps its candidates unverified rather than
# dropping them (see verify_candidates()) - fine for one flaky batch, but if
# most/all batches fail to parse that same fallback silently makes verify a
# no-op. Below this parsed-verdicts/items-sent ratio, the stage is untrusted
# and not checkpointed (see run_stage_verify()).
VERIFY_MIN_COVERAGE_RATIO = _env_float("HIGHLIGHT_VERIFY_MIN_COVERAGE_RATIO", 0.5)
# Most likely value to raise if you see the 0-coverage warning above: a
# thinking-capable JUDGE_MODEL needs room to finish reasoning before it
# reaches the actual PASS/FAIL lines if /no_think isn't fully suppressed.
VERIFY_NUM_PREDICT = _env_int("HIGHLIGHT_VERIFY_NUM_PREDICT", 1500)
JUDGE_BATCH_SIZE = _env_int("HIGHLIGHT_JUDGE_BATCH_SIZE", 20)

# --- Anti-hallucination ------------------------------------------------------
# Max seconds between a candidate's claimed timestamp and the nearest one
# actually in the transcript; beyond this it's treated as hallucinated.
TIMESTAMP_TOLERANCE_SECONDS = _env_int("HIGHLIGHT_TIMESTAMP_TOLERANCE_SECONDS", 15)

# --- Vocal isolation (single merged-track VODs, e.g. downloaded Twitch VODs) -
# A locally recorded OBS VOD has separate game/mic audio tracks, so step 1
# just extracts track index 1 directly (unchanged, see
# make_extract_mic_bat_multitrack() in OrganizeVODAndFixSRT_Emotion.py). A
# downloaded Twitch VOD has everything - game audio, music, alerts, mic -
# flattened into one track, so there's nothing to "extract"; instead
# isolate_vocals.py runs a Demucs source-separation pass on the full mix to
# pull the streamer's voice back out before the rest of the pipeline (which
# never needs to know which path produced its *_mic.wav) sees it.
# OrganizeVODAndFixSRT_Emotion.py picks between the two paths automatically
# via count_audio_streams() (ffprobe) when a video is first dropped/organized.
VOCAL_ISOLATION_MODEL = os.environ.get("VOCAL_ISOLATION_MODEL", "htdemucs")
# "auto" picks CUDA if available (same as the emotion model), else CPU - see
# detect_device() in isolate_vocals.py. CPU works but is much slower on a
# multi-hour VOD.
VOCAL_ISOLATION_DEVICE = os.environ.get("VOCAL_ISOLATION_DEVICE", "auto")
# Unset by default (demucs' own default chunking behavior). Demucs processes
# audio in windows this many seconds long rather than the whole file at once;
# if a run dies with a CUDA out-of-memory error on a very long VOD, set this
# (e.g. 20) via env var to trade a little separation quality at chunk
# boundaries for bounded memory use.
VOCAL_ISOLATION_SEGMENT_SECONDS = os.environ.get("VOCAL_ISOLATION_SEGMENT_SECONDS") or None
# Shared ffmpeg noise gate used by both audio-extraction paths.
NOISE_GATE_THRESHOLD_DB = _env_int("HIGHLIGHT_NOISE_GATE_THRESHOLD_DB", -35)
NOISE_GATE_RATIO = _env_int("HIGHLIGHT_NOISE_GATE_RATIO", 8)
NOISE_GATE_ATTACK_MS = _env_int("HIGHLIGHT_NOISE_GATE_ATTACK_MS", 10)
NOISE_GATE_RELEASE_MS = _env_int("HIGHLIGHT_NOISE_GATE_RELEASE_MS", 200)


# --- Speech-emotion sidecar (existing signal, unchanged) ---------------------
# EMOTION_LOCAL_MODEL_DIR / _FILE stay in analyze_highlights_emotion.py
# (machine-specific, see top-of-file note).
EMOTION_MODEL_ID = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
EMOTION_SCORES_CSV = "emotion_scores.csv"
EMOTION_WINDOW_SECONDS = 10
EMOTION_MODEL_INPUT_SECONDS = 30
EMOTION_CONFIDENCE_FLOOR = 0.55
EMOTION_MAX_CANDIDATES = _env_int("EMOTION_MAX_CANDIDATES", 250)
EMOTION_BATCH_SIZE = max(1, _env_int("EMOTION_BATCH_SIZE", 4))
EMOTION_USE_FP16 = _env_bool("EMOTION_USE_FP16", True)
EMOTION_ENABLED = os.environ.get("DISABLE_EMOTION_SCORING", "").lower() not in {"1", "true", "yes"}
EMOTION_BOOSTS = {
    "angry": 1.5,
    "happy": 1.25,
    "surprised": 1.5,
    "fearful": 1.0,
    "sad": 0.5,
    "disgust": 0.5,
}

# --- Hype phrase signal -------------------------------------------------------
# Enter one phrase per line in the configurator. Matching is case-insensitive
# substring matching within the configured window around each candidate.
HYPE_PHRASES_ENABLED = _env_bool("HYPE_PHRASES_ENABLED", True)
HYPE_PHRASE_WINDOW_SECONDS = _env_int("HYPE_PHRASE_WINDOW_SECONDS", 15)
HYPE_PHRASE_BOOST = _env_float("HYPE_PHRASE_BOOST", 1.5)
HYPE_PHRASE_MIN_MATCHES = _env_int("HYPE_PHRASE_MIN_MATCHES", 1)

DEFAULT_HYPE_PHRASES = [
    "clip that",
    "someone clip",
    "did you see that",
    "no shot",
    "let's fucking go",
    "what was that",
    "i'm built different",
    "easy",
    "i'm crazy",
    "i'm cracked",
    "did i just",
    "that was disgusting",
    "i can't believe it",
    "outplayed",
    "outsmarted",
    "get out of my lobby",
    "clip it",
    "clip it chat",
    "that's crazy",
    "insane",
    "bruh",
    "i'm the goat",
    "what can i say",
    "someone clip that",
    "chat clip it",
    "holy shit",
    "hell yeah",
    "crazy movement",
    "killed everyone",
    "deleted",
    "nuked",
    "what just happened",
    "what the fuck",
    "no way",
    "oh my god",
]
HYPE_PHRASES = _env_list("HYPE_PHRASES", DEFAULT_HYPE_PHRASES)

# --- Full-file audio scan (new candidate source) -----------------------------
# Closes a gap: candidates used to come only from LLM discovery reading
# transcript text, so a wordless reaction (scream, silence-then-yell) could
# never qualify no matter how loud. This scans the whole mic track with cheap
# DSP (no model) for energetic/fast moments, then runs the expensive emotion
# model only on that shortlist. See find_audio_scan_candidates().
AUDIO_SCAN_ENABLED = _env_bool("AUDIO_SCAN_ENABLED", True)
AUDIO_SCAN_HOP_SECONDS = _env_float("AUDIO_SCAN_HOP_SECONDS", 2.0)
# Peaks closer than this collapse into one, so one long scream doesn't become
# ten near-identical candidates.
AUDIO_SCAN_MIN_SEPARATION_SECONDS = _env_float("AUDIO_SCAN_MIN_SEPARATION_SECONDS", 20.0)
# Std devs above *this stream's own* average loudness/rate needed to qualify -
# relative to its own baseline, not a fixed dB/rate, since mic gain and
# baseline energy vary a lot between streamers.
AUDIO_SCAN_MIN_ZSCORE = _env_float("AUDIO_SCAN_MIN_ZSCORE", 1.0)
# Cap on DSP peaks promoted to real (titled + emotion-scored) candidates,
# bounding the expensive stages on long VODs.
AUDIO_SCAN_MAX_CANDIDATES = _env_int("AUDIO_SCAN_MAX_CANDIDATES", 120)
# Skip an audio-scan candidate within this many seconds of an existing
# LLM-discovered highlight - audio scan should fill gaps, not duplicate.
AUDIO_SCAN_SKIP_NEAR_EXISTING_SECONDS = _env_float("AUDIO_SCAN_SKIP_NEAR_EXISTING_SECONDS", 12.0)
# Loudness weighted higher by default - cleaner signal than onset-rate, which
# reacts to any percussive sound, not just speech.
AUDIO_SCAN_LOUDNESS_WEIGHT = _env_float("AUDIO_SCAN_LOUDNESS_WEIGHT", 0.6)
AUDIO_SCAN_RATE_WEIGHT = _env_float("AUDIO_SCAN_RATE_WEIGHT", 0.4)
AUDIO_SCAN_TITLE_BATCH_SIZE = _env_int("AUDIO_SCAN_TITLE_BATCH_SIZE", 10)
# Raise if titles come back empty. Titling now calls JUDGE_MODEL via
# OLLAMA_CHAT_URL with think:false, so this no longer needs to cover a hidden
# reasoning trace - 1200 holds with margin.
AUDIO_SCAN_TITLE_NUM_PREDICT = _env_int("AUDIO_SCAN_TITLE_NUM_PREDICT", 1200)

# --- Preview clips (optional, off by default) --------------------------------
# Cuts a short stream-copy mp4 (fast, no re-encode) around each final
# highlight into stream_folder/clips/, for scrubbing on phone/couch before
# opening Resolve. Start may land on the nearest keyframe, not the exact
# second - fine for a preview, not a real-edit EDL replacement.
EXPORT_PREVIEW_CLIPS = _env_bool("EXPORT_PREVIEW_CLIPS", False)
PREVIEW_CLIP_SECONDS_BEFORE = _env_float("PREVIEW_CLIP_SECONDS_BEFORE", 5.0)
PREVIEW_CLIP_SECONDS_AFTER = _env_float("PREVIEW_CLIP_SECONDS_AFTER", 10.0)

# --- Run metadata -------------------------------------------------------------
RUN_INFO_FILENAME = "run_info.json"
# Unlike RUN_INFO_FILENAME, NOT written into the stream folder - see
# SCRIPT_DIR / record_pipeline_run_history().
RUN_HISTORY_FILENAME = "pipeline_run_history.csv"


# The GUI in configure_models.py reads this registry and writes selected
# defaults back to the definitions above. Runtime imports stay unchanged.
#
# Keeping the registry here gives the GUI one source for fields, labels,
# presets, and the environment names used by the save-back code.


# kind controls the editor shown by the GUI. stage controls grouping.
# env is the environment variable used by the corresponding definition above.
# Keep environment names unique.
EDITABLE_PARAMS = [
    # --- Models (per-step role assignment) ---
    {"key": "MODEL",            "env": "HIGHLIGHT_MODEL",            "kind": "model", "stage": "Models",
     "label": "Discovery model (MODEL)",
     "help": "Reads each transcript chunk and proposes highlight candidates. Long context, less careful reasoning."},
    {"key": "JUDGE_MODEL",      "env": "HIGHLIGHT_JUDGE_MODEL",      "kind": "model", "stage": "Models",
     "label": "Judge / verify / titling model (JUDGE_MODEL)",
     "help": "Verify, judge ranking, and audio-scan titling. Shorter prompts, careful structured output. Must support /api/chat with think:false (the qwen3.5 family does)."},
    # --- Ollama connection / generation budget ---
    {"key": "DISCOVERY_NUM_CTX", "env": "HIGHLIGHT_DISCOVERY_NUM_CTX", "kind": "int", "stage": "Ollama",
     "label": "Discovery context window (DISCOVERY_NUM_CTX, tokens)",
     "help": "Max prompt+output tokens for each full transcript discovery request. This is sent as API options.num_ctx; larger values preserve more transcript context but use more KV-cache VRAM."},
    {"key": "JUDGE_NUM_CTX",     "env": "HIGHLIGHT_JUDGE_NUM_CTX",     "kind": "int", "stage": "Ollama",
     "label": "Judge context window (JUDGE_NUM_CTX, tokens)",
     "help": "Max prompt+output tokens for audio titling, verification, and final judging. Keep lower for qwen3.6:35b-a3b to reduce CPU/GPU memory pressure."},
    {"key": "VERIFY_NUM_PREDICT","env": "HIGHLIGHT_VERIFY_NUM_PREDICT","kind": "int",  "stage": "Ollama",
     "label": "Verify max output tokens (VERIFY_NUM_PREDICT)",
     "help": "Per verify batch. If you see '0 coverage' warnings, raise this - the model is running out of output budget before reaching the PASS/FAIL lines."},
    {"key": "AUDIO_SCAN_TITLE_NUM_PREDICT","env":"AUDIO_SCAN_TITLE_NUM_PREDICT","kind":"int","stage":"Ollama",
     "label": "Audio-scan titling max output tokens (AUDIO_SCAN_TITLE_NUM_PREDICT)",
     "help": "Per titling batch. Raise if audio-scan titles come back empty."},
    {"key": "OLLAMA_RETRIES",   "env": "OLLAMA_RETRIES",             "kind": "int",   "stage": "Ollama",
     "label": "Ollama retries on failure (OLLAMA_RETRIES)",
     "help": "Linear backoff between retries; transient failures no longer silently drop coverage."},
    {"key": "OLLAMA_RETRY_BACKOFF_SECONDS","env":"OLLAMA_RETRY_BACKOFF_SECONDS","kind":"float","stage":"Ollama",
     "label": "Ollama retry backoff seconds (OLLAMA_RETRY_BACKOFF_SECONDS)",
     "help": "Seconds between retries."},
    {"key": "OLLAMA_START_ON_CONNECTION_ERROR","env":"HIGHLIGHT_OLLAMA_START_ON_CONNECTION_ERROR","kind":"bool","stage":"Ollama",
     "label": "Auto-start local Ollama on connection failure",
     "help": "If a call can't reach Ollama, spawn `ollama serve` (local host only) and retry after it responds to /api/version."},
    {"key": "OLLAMA_SERVE_READY_TIMEOUT_SECONDS","env":"HIGHLIGHT_OLLAMA_SERVE_READY_TIMEOUT_SECONDS","kind":"float","stage":"Ollama",
     "label": "Ollama serve readiness timeout (seconds)",
     "help": "How long to wait for a freshly spawned `ollama serve` to accept connections."},
    # --- Selection / counts ---
    {"key": "TOP_N",            "env": "HIGHLIGHT_TOP_N",           "kind": "int",   "stage": "Selection",
     "label": "Final highlights to export (TOP_N)",
     "help": "Number of rows in the CSV and markers in the EDL."},
    {"key": "JUDGE_POOL_SIZE",  "env": "HIGHLIGHT_JUDGE_POOL_SIZE",  "kind": "int",   "stage": "Selection",
     "label": "Judge pool size (JUDGE_POOL_SIZE)",
     "help": "Only this many top-scored candidates reach the judge. Larger = more judging calls."},
    {"key": "VERIFY_POOL_SIZE", "env": "HIGHLIGHT_VERIFY_POOL_SIZE", "kind": "int",   "stage": "Selection",
     "label": "Verify pool size (VERIFY_POOL_SIZE)",
     "help": "Candidates trimmed to this before verification. Defaults to 1.5x the judge pool."},
    {"key": "VERIFY_BATCH_SIZE","env": "HIGHLIGHT_VERIFY_BATCH_SIZE","kind": "int",   "stage": "Selection",
     "label": "Verify batch size (VERIFY_BATCH_SIZE)",
     "help": "Candidates per verify LLM call."},
    {"key": "VERIFY_MIN_COVERAGE_RATIO","env":"HIGHLIGHT_VERIFY_MIN_COVERAGE_RATIO","kind":"float","stage":"Selection",
     "label": "Min verify coverage ratio (VERIFY_MIN_COVERAGE_RATIO)",
     "help": "If parsed-verdicts/items-sent drops below this, verify is untrusted and NOT checkpointed."},
    {"key": "JUDGE_BATCH_SIZE", "env": "HIGHLIGHT_JUDGE_BATCH_SIZE", "kind": "int",   "stage": "Selection",
     "label": "Judge batch size (JUDGE_BATCH_SIZE)",
     "help": "Candidates per judge LLM call. The judge ranks them comparatively within the batch."},
    {"key": "TIMESTAMP_TOLERANCE_SECONDS","env":"HIGHLIGHT_TIMESTAMP_TOLERANCE_SECONDS","kind":"int","stage":"Selection",
     "label": "Anti-hallucination timestamp tolerance (seconds)",
     "help": "Max seconds between a candidate's claimed timestamp and the nearest real transcript timestamp; beyond this it's dropped as hallucinated."},
    # --- Vocal isolation ---
    {"key": "VOCAL_ISOLATION_SEGMENT_SECONDS","env":"VOCAL_ISOLATION_SEGMENT_SECONDS","kind":"text","stage":"Vocal isolation",
     "label": "Demucs segment seconds (VOCAL_ISOLATION_SEGMENT_SECONDS)",
     "help": "Demucs processes audio in windows this many seconds long. Set to 20 if a long VOD OOMs in Demucs on a 10GB card; empty/0 = whole-file (slightly cleaner boundaries)."},
    # --- Audio cleanup ---
    {"key": "NOISE_GATE_THRESHOLD_DB", "env": "HIGHLIGHT_NOISE_GATE_THRESHOLD_DB", "kind": "int", "stage": "Audio cleanup",
     "label": "Noise-gate threshold (dB)",
     "help": "Audio below this level is attenuated before transcription. Raise it to suppress more room noise; lower it to preserve quiet speech."},
    {"key": "NOISE_GATE_RATIO", "env": "HIGHLIGHT_NOISE_GATE_RATIO", "kind": "int", "stage": "Audio cleanup",
     "label": "Noise-gate ratio",
     "help": "How strongly the gate attenuates audio below the threshold."},
    {"key": "NOISE_GATE_ATTACK_MS", "env": "HIGHLIGHT_NOISE_GATE_ATTACK_MS", "kind": "int", "stage": "Audio cleanup",
     "label": "Noise-gate attack (ms)",
     "help": "How quickly the gate closes when the signal drops below the threshold."},
    {"key": "NOISE_GATE_RELEASE_MS", "env": "HIGHLIGHT_NOISE_GATE_RELEASE_MS", "kind": "int", "stage": "Audio cleanup",
     "label": "Noise-gate release (ms)",
     "help": "How quickly the gate reopens when speech resumes."},

    # --- Emotion sidecar ---
    {"key": "EMOTION_MAX_CANDIDATES","env":"EMOTION_MAX_CANDIDATES","kind":"int","stage":"Emotion",
     "label": "Max emotion-scored candidates (EMOTION_MAX_CANDIDATES)",
     "help": "Caps the expensive emotion-model pass on long VODs."},
    {"key": "EMOTION_BATCH_SIZE", "env": "EMOTION_BATCH_SIZE", "kind": "int", "stage": "Emotion",
     "label": "Emotion batch size (EMOTION_BATCH_SIZE)",
     "help": "Candidates per emotion model forward pass."},
    {"key": "EMOTION_ENABLED",  "env": "DISABLE_EMOTION_SCORING",   "kind": "bool",  "stage": "Emotion",
     "label": "Enable speech-emotion scoring (EMOTION_ENABLED)",
     "help": "When ON, boosts candidates with angry/happy/surprised speech. The env var is inverted (DISABLE_EMOTION_SCORING)."},
    # --- Audio scan ---
    {"key": "AUDIO_SCAN_ENABLED","env":"AUDIO_SCAN_ENABLED","kind":"bool","stage":"Audio scan",
     "label": "Enable model-free audio scan (AUDIO_SCAN_ENABLED)",
     "help": "DSP loudness/rate scan for wordless-reaction candidates that transcript discovery can't find."},
    {"key": "AUDIO_SCAN_MAX_CANDIDATES","env":"AUDIO_SCAN_MAX_CANDIDATES","kind":"int","stage":"Audio scan",
     "label": "Max audio-scan candidates (AUDIO_SCAN_MAX_CANDIDATES)",
     "help": "Caps DSP peaks promoted to real titled+emotion-scored candidates."},
    {"key": "AUDIO_SCAN_TITLE_BATCH_SIZE","env":"AUDIO_SCAN_TITLE_BATCH_SIZE","kind":"int","stage":"Audio scan",
     "label": "Audio-scan titling batch size",
     "help": "Peaks titled per LLM call."},
    # --- Hype phrase signal ---
    {"key": "HYPE_PHRASES_ENABLED",    "env": "HYPE_PHRASES_ENABLED",    "kind": "bool",  "stage": "Hype phrase signal",
     "label": "Enable hype phrase detection (HYPE_PHRASES_ENABLED)",
     "help": "Detects streamer hype phrases like 'clip that', 'no shot', 'let\\'s fucking go', etc. in transcript around candidates and boosts their scores."},
    {"key": "HYPE_PHRASE_WINDOW_SECONDS","env": "HYPE_PHRASE_WINDOW_SECONDS","kind": "int",  "stage": "Hype phrase signal",
     "label": "Search window around candidate (seconds)",
     "help": "How many seconds before/after each candidate timestamp to search for hype phrases in the transcript."},
    {"key": "HYPE_PHRASE_BOOST",      "env": "HYPE_PHRASE_BOOST",      "kind": "float", "stage": "Hype phrase signal",
     "label": "Hype phrase score boost (HYPE_PHRASE_BOOST)",
     "help": "Score points added when hype phrases are detected (capped at 2.0). Applied like emotion boost."},
    {"key": "HYPE_PHRASE_MIN_MATCHES","env": "HYPE_PHRASE_MIN_MATCHES","kind": "int",   "stage": "Hype phrase signal",
     "label": "Min phrase matches to trigger boost",
     "help": "Minimum number of distinct hype phrase matches needed within the window to apply the boost."},
    {"key": "HYPE_PHRASES", "env": "HYPE_PHRASES", "kind": "phrases", "stage": "Hype phrase signal",
     "label": "Hype phrases",
     "help": "One phrase per line. Matching is case-insensitive and searches the transcript window around each candidate."},

]


# Presets target an RTX 3080 with 10GB VRAM and 32GB system RAM.
#
# The 35B qwen3.6 model is larger than the card, so Ollama must offload part
# of it to system memory. Its preset lowers JUDGE_NUM_CTX and uses larger
# output budgets for the structured verify/judge responses. Discovery stays on
# qwen3:8b with its own DISCOVERY_NUM_CTX.
PRESETS = {
    "3080 10GB - qwen3:8b (discovery) + qwen3.6:35b-a3b (judge)": {
        "name_short": "qwen3.6:35b-a3b",
        "description": ("Discovery on qwen3:8b; judge/verify/titling on qwen3.6:35b-a3b "
                        "(slow; partial CPU offload). For RTX 3080 10GB + 32GB DDR4. "
                        "Better ranking; expect longer per-call latency from the "
                        "judge/verify stages."),
        "values": {
            "MODEL":                       "qwen3:8b",
            "JUDGE_MODEL":                 "qwen3.6:35b-a3b",
            "DISCOVERY_NUM_CTX":            8192,
            "JUDGE_NUM_CTX":                 6144,
            "VERIFY_NUM_PREDICT":          2200,
            "AUDIO_SCAN_TITLE_NUM_PREDICT":1800,
            "OLLAMA_RETRIES":              3,
            "OLLAMA_RETRY_BACKOFF_SECONDS":8.0,
            "OLLAMA_START_ON_CONNECTION_ERROR": True,
            "OLLAMA_SERVE_READY_TIMEOUT_SECONDS": 90.0,
            "TOP_N":                       50,
            "JUDGE_POOL_SIZE":             100,
            "VERIFY_POOL_SIZE":           150,
            "VERIFY_BATCH_SIZE":           8,
            "VERIFY_MIN_COVERAGE_RATIO":   0.5,
            "JUDGE_BATCH_SIZE":           10,
            "TIMESTAMP_TOLERANCE_SECONDS": 15,
            "VOCAL_ISOLATION_SEGMENT_SECONDS": 20,
            "NOISE_GATE_THRESHOLD_DB":   -35,
            "NOISE_GATE_RATIO":             8,
            "NOISE_GATE_ATTACK_MS":        10,
            "NOISE_GATE_RELEASE_MS":       200,

            "EMOTION_MAX_CANDIDATES":      200,
            "EMOTION_BATCH_SIZE":          2,
            "EMOTION_ENABLED":             True,
            "HYPE_PHRASES_ENABLED":        True,
            "HYPE_PHRASE_WINDOW_SECONDS":  15,
            "HYPE_PHRASE_BOOST":           1.5,
            "HYPE_PHRASE_MIN_MATCHES":     1,
            "AUDIO_SCAN_ENABLED":          True,
            "AUDIO_SCAN_MAX_CANDIDATES":  100,
            "AUDIO_SCAN_TITLE_BATCH_SIZE": 8,
            "EXPORT_PREVIEW_CLIPS":       False,
            "PREVIEW_CLIP_SECONDS_BEFORE": 5.0,
            "PREVIEW_CLIP_SECONDS_AFTER":  10.0,
        },
    },
    "3080 10GB - qwen3:8b (discovery) + qwen3.5:9b-q4_K_M (judge)": {
        "name_short": "qwen3.5:9b-q4_K_M",
        "description": ("Discovery on qwen3:8b; judge/verify/titling on "
                        "qwen3.5:9b-q4_K_M (fast, fully resident in 10GB). For "
                        "RTX 3080 10GB + 32GB DDR4. The balanced default; judge "
                        "in ~4-6s per batch."),
        "values": {
            "MODEL":                       "qwen3:8b",
            "JUDGE_MODEL":                 "qwen3.5:9b-q4_K_M",
            "DISCOVERY_NUM_CTX":            8192,
            "JUDGE_NUM_CTX":                 8192,
            "VERIFY_NUM_PREDICT":          1500,
            "AUDIO_SCAN_TITLE_NUM_PREDICT":1200,
            "OLLAMA_RETRIES":              2,
            "OLLAMA_RETRY_BACKOFF_SECONDS":5.0,
            "OLLAMA_START_ON_CONNECTION_ERROR": True,
            "OLLAMA_SERVE_READY_TIMEOUT_SECONDS": 60.0,
            "TOP_N":                       50,
            "JUDGE_POOL_SIZE":             100,
            "VERIFY_POOL_SIZE":           150,
            "VERIFY_BATCH_SIZE":          10,
            "VERIFY_MIN_COVERAGE_RATIO":   0.5,
            "JUDGE_BATCH_SIZE":           20,
            "TIMESTAMP_TOLERANCE_SECONDS": 15,
            "VOCAL_ISOLATION_SEGMENT_SECONDS": 20,
            "NOISE_GATE_THRESHOLD_DB":   -35,
            "NOISE_GATE_RATIO":             8,
            "NOISE_GATE_ATTACK_MS":        10,
            "NOISE_GATE_RELEASE_MS":       200,
            "EMOTION_MAX_CANDIDATES":      250,
            "EMOTION_BATCH_SIZE":          4,
            "EMOTION_ENABLED":             True,
            "HYPE_PHRASES_ENABLED":        True,
            "HYPE_PHRASE_WINDOW_SECONDS":  15,
            "HYPE_PHRASE_BOOST":           1.5,
            "HYPE_PHRASE_MIN_MATCHES":     1,
            "AUDIO_SCAN_ENABLED":          True,
            "AUDIO_SCAN_MAX_CANDIDATES":  120,
            "AUDIO_SCAN_TITLE_BATCH_SIZE":10,
            "EXPORT_PREVIEW_CLIPS":       False,
            "PREVIEW_CLIP_SECONDS_BEFORE": 5.0,
            "PREVIEW_CLIP_SECONDS_AFTER":  10.0,
        },
    },
}


# --- Model detection ---------------------------------------------------------
def list_ollama_models() -> tuple[list[str], str | None]:
    """Return installed model names and an optional warning."""
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        return [], "ollama is not on PATH - install it from https://ollama.com/download, then `ollama pull <model>`."
    except Exception as exc:
        return [], f"couldn't run `ollama list`: {exc}"
    if proc.returncode != 0:
        return [], f"`ollama list` exited {proc.returncode}: {proc.stderr.strip()}"
    names: list[str] = []
    for line in proc.stdout.splitlines()[1:]:  # skip the "NAME ID SIZE MODIFIED" header
        line = line.strip()
        if not line:
            continue
        name = line.split()[0]
        if name:
            names.append(name)
    return names, None


# --- Save-back ---------------------------------------------------------------
# Rewrite only the default literal on definitions listed in EDITABLE_PARAMS.
# The source file is replaced atomically so a failed write cannot leave a
# partial configuration behind.
def _coerce_for_write(kind: str, value):
    """Return the source literal for a registry value."""
    if kind == "bool":
        return "True" if value else "False"
    if kind == "int":
        return str(int(value)) if value not in (None, "") else "0"
    if kind == "float":
        return repr(float(value)) if value not in (None, "") else "0.0"
    if kind == "model":
        return f'"{value}"'
    if value in (None, ""):
        return "None"
    return f'"{value}"'


def _phrase_defaults_source(value, indent: str, newline: str) -> str:
    raw_phrases = value.splitlines() if isinstance(value, str) else (value or [])
    phrases: list[str] = []
    seen: set[str] = set()
    for raw_phrase in raw_phrases:
        phrase = str(raw_phrase).strip()
        key = phrase.casefold()
        if phrase and key not in seen:
            phrases.append(phrase)
            seen.add(key)
    rows = "".join(f"{indent}    {phrase!r},{newline}" for phrase in phrases)
    return f"{indent}DEFAULT_HYPE_PHRASES = [{newline}{rows}{indent}]{newline}"

def apply_config_values(values: dict, file_path: str | None = None) -> tuple[bool, str]:
    """Write the supplied defaults to the config file.

    Only keys present in ``values`` are changed. The two definitions that do
    not use the standard helper-call form are handled separately below.
    """
    path = Path(file_path) if file_path else Path(__file__).resolve()
    source = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in source else "\n"
    changed = 0
    touched: set[str] = set()

    if "HYPE_PHRASES" in values:
        phrase_block = re.compile(
            r"(?ms)^DEFAULT_HYPE_PHRASES\s*=\s*\[.*?^\]\s*(?:\r?\n|$)"
        )
        replacement = _phrase_defaults_source(values["HYPE_PHRASES"], "", newline)
        new_source, replacements = phrase_block.subn(replacement, source, count=1)
        if replacements:
            touched.add("HYPE_PHRASES")
            if new_source != source:
                changed += 1
            source = new_source

    lines = source.splitlines(keepends=True)

    env_to_param = {param["env"]: param for param in EDITABLE_PARAMS}
    keys_in_values = set(values.keys())

    helper_alt = r'(?:os\.environ\.get|_env_int|_env_float|_env_bool)'
    nested_default = r'[^()]*(?:\([^()]*\)[^()]*)*'
    lhs_pat = re.compile(r'^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)')
    helper_pat = re.compile(
        r'(' + helper_alt + r'\(\s*"([^"]+)"\s*,\s*)(' + nested_default + r')(\))'
    )
    editable_keys = {param["key"] for param in EDITABLE_PARAMS}

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue

        # These definitions do not use the standard helper-call form.
        if stripped.startswith("EMOTION_ENABLED =") and "EMOTION_ENABLED" in keys_in_values:
            new_val = "True" if values.get("EMOTION_ENABLED") else "False"
            match = re.match(r'^(\s*EMOTION_ENABLED\s*=\s*).*?(\r?\n)$', line)
            if match:
                new_line = f'{match.group(1)}{new_val}{match.group(2)}'
                if new_line != line:
                    lines[i] = new_line
                    changed += 1
                touched.add("EMOTION_ENABLED")
            continue

        if stripped.startswith("VOCAL_ISOLATION_SEGMENT_SECONDS =") and "VOCAL_ISOLATION_SEGMENT_SECONDS" in keys_in_values:
            raw_value = values.get("VOCAL_ISOLATION_SEGMENT_SECONDS", "")
            text = raw_value.strip() if isinstance(raw_value, str) else str(raw_value).strip()
            literal = "None" if text in ("", "0", "None", "none") else f'"{text}"'
            match = re.match(
                r'^(\s*VOCAL_ISOLATION_SEGMENT_SECONDS\s*=\s*).*?(\r?\n)?$',
                line,
            )
            if match:
                new_line = (
                    f'{match.group(1)}os.environ.get("VOCAL_ISOLATION_SEGMENT_SECONDS") '
                    f'or {literal}{match.group(2)}'
                )
                if new_line != line:
                    lines[i] = new_line
                    changed += 1
                touched.add("VOCAL_ISOLATION_SEGMENT_SECONDS")
            continue

        # The helper call may be wrapped in another expression.
        assignment = lhs_pat.match(line)
        if not assignment:
            continue
        var_name = assignment.group(2)
        if var_name not in editable_keys:
            continue
        if var_name not in keys_in_values:
            continue
        helper_match = helper_pat.search(line, assignment.end())
        if not helper_match:
            continue
        call_prefix, env_name, _, close_paren = helper_match.groups()
        param = env_to_param.get(env_name)
        if param is None or param["key"] != var_name:
            continue
        new_literal = _coerce_for_write(param["kind"], values.get(param["key"]))
        new_line = (
            line[:helper_match.start()]
            + call_prefix
            + new_literal
            + close_paren
            + line[helper_match.end():]
        )
        if new_line != line:
            lines[i] = new_line
            changed += 1
        touched.add(param["key"])

    # Report params the caller asked for but we couldn't find a line for.
    requested = keys_in_values & {param["key"] for param in EDITABLE_PARAMS}
    skipped = sorted(requested - touched)

    if changed == 0:
        if not requested:
            return True, "Nothing to do - no editable params in the request."
        if not touched:
            return False, f"Couldn't rewrite any of {len(skipped)} parameter(s): {', '.join(skipped)}. The config file structure may have changed."
        return True, "No changes needed - values already match the file."

    new_source = "".join(lines)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(new_source, encoding="utf-8")
    os.replace(temp_path, path)
    skipped_text = f" Skipped ({len(skipped)}): {', '.join(skipped)}." if skipped else ""
    return True, f"Updated {changed} parameter(s) in {path.name}.{skipped_text}"

