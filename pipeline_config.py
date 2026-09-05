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
import requests
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


# --- Per-step VOD folder layout ----------------------------------------------
# The RunAll GUI organizes each step's artifacts into its own subfolder so the
# VOD folder stays readable: the runner bat, the big run log, and the two
# DaVinci hand-off files (fixed SRT + marker EDL) stay at the root; everything
# a step generates (including its debug bats for step 5) lives in that step's
# folder. Helper shared by the organizer and the analyzer.
STEP_FOLDER_NAMES = {
    1: "step1_extract_mic_audio",
    2: "step2_transcribe_audio",
    3: "step3_fix_srt",
    4: "step4_split_srt",
    5: "step5_analyze_highlights",
}


def step_subdir(stream_folder, step: int):
    """Per-step subfolder inside a VOD folder (created on demand)."""
    return Path(stream_folder) / STEP_FOLDER_NAMES[step]


# --- Ollama models ---------------------------------------------------------
# JUDGE_MODEL is separate from MODEL: discovery just reads a transcript chunk
# and proposes candidates; verify/judge/titling need more careful structured
# reasoning over shorter, denser prompts.
# qwen3.5:9b-q4_K_M (~6.6 GB) serves BOTH roles by default - it fits a
# 10GB 3080 with 8192 context in either role, and the per-role tunings
# below carry the entries for every other model (including qwen3:8b).
# qwen3.6:35b-a3b / qwen3.5:35b need partial CPU offload, so their presets
# use a smaller context to leave memory for KV cache.
MODEL = os.environ.get("HIGHLIGHT_MODEL", "qwen3.5:9b-q4_K_M")
JUDGE_MODEL = os.environ.get("HIGHLIGHT_JUDGE_MODEL", "qwen3.5:9b-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
# qwen3.5 thinking is controlled via Ollama's renderer, not the old
# "/no_think" prompt tag. /api/generate ignores think:false for qwen3.5
# (ollama/ollama#14793) and newer Ollama emits its reasoning into a separate
# `thinking` field that counts against num_predict, leaving "response" empty.
# Every JUDGE_MODEL call needing thinking OFF (titling, verify, judge) must
# therefore use OLLAMA_CHAT_URL with top-level think:false - see the call
# sites in analyze_highlights_emotion.py.
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_RETRIES = _env_int("OLLAMA_RETRIES", 2)
OLLAMA_RETRY_BACKOFF_SECONDS = _env_float("OLLAMA_RETRY_BACKOFF_SECONDS", 5)
# Ollama must be running before the highlight analyzer starts. The analyzer
# performs a /api/version preflight and stops with instructions to close the
# RunAll GUI, launch Ollama manually, and rerun it if the server is unavailable.
# Ollama's `options.num_ctx` is sent with each API request, so keep separate
# budgets for the long transcript discovery prompt and the shorter judge-side
# prompts. Discovery needs the larger window; qwen3.6's judge preset lowers
# the judge window to reduce memory pressure.
DISCOVERY_NUM_CTX = _env_int("HIGHLIGHT_DISCOVERY_NUM_CTX", 8192)
JUDGE_NUM_CTX = _env_int("HIGHLIGHT_JUDGE_NUM_CTX", 8192)

# --- LLM backend toggle ------------------------------------------------------
# "ollama" (default) talks to an Ollama daemon at OLLAMA_URL/OLLAMA_CHAT_URL.
# "llamacpp" talks to a llama.cpp `llama-server` process at LLAMA_SERVER_URL
# (OpenAI-style endpoints). The backend is a single toggle: with llamacpp the
# same ONE GGUF model serves every role (discovery, titling, verify, judge) -
# llama-server hosts exactly one model per process, so the two-tier Ollama
# model split collapses. Launch llama-server before Step 5; the analyzer only
# preflights /health and never starts or stops the server itself (same policy
# as Ollama since 2026-08-21).
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")
# Two-model support for llama.cpp: discovery and judge can use different GGUFs.
# The server is managed by the pipeline - it starts with the discovery model,
# stops to free VRAM for the emotion stage, then restarts with the judge model
# for audioscan/verify/judge - matching Ollama's sequential load/unload pattern.
# LLAMA_MODEL_PATH is kept for backward compat and as a fallback when the
# per-role path is empty (single-GGUF mode).
LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "")
LLAMA_DISCOVERY_MODEL_PATH = os.environ.get("LLAMA_DISCOVERY_MODEL_PATH", LLAMA_MODEL_PATH)
LLAMA_JUDGE_MODEL_PATH = os.environ.get("LLAMA_JUDGE_MODEL_PATH", LLAMA_MODEL_PATH)
# Server-launch context window. Unlike Ollama's per-request options.num_ctx,
# llama.cpp's context is fixed at launch (-c), so DISCOVERY_NUM_CTX and
# JUDGE_NUM_CTX do not apply on this backend; this value drives the generated
# Start_LlamaServer.bat(s) and the configurator's guidance text. A single value
# is used for both roles to keep the toggle simple; per-role ctx can be added
# later if VRAM tuning needs it.
LLAMA_CONTEXT_SIZE = _env_int("LLAMA_CONTEXT_SIZE", 8192)

# --- Transcription -----------------------------------------------------------
# Long uninterrupted Whisper runs can enter a repeated-text decoding loop.
# Independent chunks reset the decoder context; overlap protects words at
# chunk boundaries. Both values remain environment-overridable for tuning.
TRANSCRIPTION_CHUNK_MINUTES = _env_int("TRANSCRIPTION_CHUNK_MINUTES", 30)
TRANSCRIPTION_CHUNK_OVERLAP_SECONDS = _env_int("TRANSCRIPTION_CHUNK_OVERLAP_SECONDS", 10)

# --- Pipeline behavior -------------------------------------------------------
# The RunAll GUI can optionally suspend Windows after every pipeline step
# completes successfully. Keep this opt-in so an unattended run never sleeps
# the machine unless the user explicitly enables it in the configurator.
AUTO_SLEEP_AFTER_PIPELINE = _env_bool("AUTO_SLEEP_AFTER_PIPELINE", False)
# Grace period between the pipeline finishing and Windows actually suspending.
# Any mouse movement or keystroke during the countdown cancels the auto-sleep
# entirely (the PC stays awake and the GUI returns to its normal Done state).
# Only used when AUTO_SLEEP_AFTER_PIPELINE is enabled; default 30 seconds.
AUTO_SLEEP_DELAY_SECONDS = _env_int("AUTO_SLEEP_DELAY_SECONDS", 30)


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
# HTDemucs' model-native window is 7.8 seconds; Demucs' CLI accepts integer
# overrides only, so the safe default is 7. Long inputs are externally split
# into bounded windows by isolate_vocals.py because Demucs otherwise allocates
# its output tensor for the entire file.
VOCAL_ISOLATION_SEGMENT_SECONDS = os.environ.get("VOCAL_ISOLATION_SEGMENT_SECONDS") or "7"
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
# One row is appended for every big step attempted by the RunAll GUI. The
# separate file keeps these five user-facing durations independent from the
# analyzer's six internal-stage history.
STEP_HISTORY_FILENAME = "View_Pipeline_Duration_History.csv"
BIG_STEP_LABELS = (
    "1. Extract mic audio",
    "2. Transcribe audio",
    "3. Fix SRT",
    "4. Split SRT",
    "5. Analyze highlights",
)


# The GUI in configure_models.py reads this registry and writes selected
# defaults back to the definitions above. Runtime imports stay unchanged.
#
# Keeping the registry here gives the GUI one source for fields, labels,
# presets, and the environment names used by the save-back code.
# kind controls the editor shown by the GUI. stage controls grouping.
# env is the environment variable used by the corresponding definition above.
# Keep environment names unique.
EDITABLE_PARAMS = [
    # --- LLM backend ---
    {"key": "LLM_BACKEND", "env": "LLM_BACKEND", "kind": "backend", "stage": "LLM backend",
     "label": "LLM backend (LLM_BACKEND)",
     "help": "ollama = Ollama daemon with separate discovery/judge models. llamacpp = managed llama-server that hot-swaps GGUFs (discovery GGUF then judge GGUF, matching Ollama's unload/load)."},
    {"key": "LLAMA_SERVER_URL", "env": "LLAMA_SERVER_URL", "kind": "text", "stage": "LLM backend",
     "label": "llama.cpp server URL (LLAMA_SERVER_URL)",
     "help": "Base URL for the managed llama-server. The pipeline starts/stops the server itself, so this is just the host:port it binds to (default http://localhost:8080). Only used when LLM_BACKEND is llamacpp."},
    {"key": "LLAMA_DISCOVERY_MODEL_PATH", "env": "LLAMA_DISCOVERY_MODEL_PATH", "kind": "text", "stage": "LLM backend",
     "label": "llama.cpp discovery GGUF (LLAMA_DISCOVERY_MODEL_PATH)",
     "help": "GGUF for discovery passes. Falls back to LLAMA_MODEL_PATH if empty (single-GGUF mode). Pipeline starts the server with this before discovery, then restarts with the judge GGUF for verify/judge."},
    {"key": "LLAMA_JUDGE_MODEL_PATH", "env": "LLAMA_JUDGE_MODEL_PATH", "kind": "text", "stage": "LLM backend",
     "label": "llama.cpp judge GGUF (LLAMA_JUDGE_MODEL_PATH)",
     "help": "GGUF for audioscan/verify/judge. Falls back to LLAMA_MODEL_PATH/discovery path if empty. Pipeline hot-swaps to this after the emotion stage to free VRAM."},
    {"key": "LLAMA_MODEL_PATH", "env": "LLAMA_MODEL_PATH", "kind": "text", "stage": "LLM backend",
     "label": "llama.cpp GGUF fallback (LLAMA_MODEL_PATH)",
     "help": "Legacy single-GGUF path. Used when per-role paths are empty. Kept for backward compat."},
    {"key": "LLAMA_CONTEXT_SIZE", "env": "LLAMA_CONTEXT_SIZE", "kind": "int", "stage": "LLM backend",
     "label": "llama.cpp context size (LLAMA_CONTEXT_SIZE, tokens)",
     "help": "Server-launch context window (-c) for both roles. Only used when LLM_BACKEND is llamacpp."},
    # --- General pipeline behavior ---
    {"key": "AUTO_SLEEP_AFTER_PIPELINE", "env": "AUTO_SLEEP_AFTER_PIPELINE", "kind": "bool", "stage": "General",
     "label": "Put PC to sleep after pipeline completes",
     "help": "When ON, the RunAll GUI puts this Windows PC to sleep after all five pipeline steps finish successfully. Moving the mouse or pressing any key during the countdown cancels the sleep. Default: OFF."},
    {"key": "AUTO_SLEEP_DELAY_SECONDS", "env": "AUTO_SLEEP_DELAY_SECONDS", "kind": "int", "stage": "General",
     "label": "Seconds to wait before sleep",
     "help": "How long the RunAll GUI waits after the pipeline finishes before putting the PC to sleep. Any mouse movement or keystroke during the countdown cancels the sleep. Default: 30."},
    # --- Transcription ---
    {"key": "TRANSCRIPTION_CHUNK_MINUTES", "env": "TRANSCRIPTION_CHUNK_MINUTES", "kind": "int", "stage": "Transcription",
     "label": "Whisper chunk length (minutes)",
     "help": "Each audio chunk starts a fresh Whisper decoder context. 30 minutes is a safe default for long VODs."},
    {"key": "TRANSCRIPTION_CHUNK_OVERLAP_SECONDS", "env": "TRANSCRIPTION_CHUNK_OVERLAP_SECONDS", "kind": "int", "stage": "Transcription",
     "label": "Whisper chunk overlap (seconds)",
     "help": "Overlap between adjacent audio chunks so speech at boundaries is not lost. 10 seconds is the default."},
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
     "help": "Demucs processes audio in windows this many seconds long. For HTDemucs, use an integer from 1 to 7; larger values are rejected by the model, so the runner uses its native 7.8-second window. Empty/0 uses that native window."},
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


# --- Per-role model tunings (replaces combo PRESETS) ----------------------------
# Each installed Ollama model has separate tunings for when it's used as
# discovery vs judge. The GUI auto-fills role-specific values on selection
# so switching qwen3.5:9b on discovery fills discovery-optimal values, and
# switching the same tag on judge fills judge-optimal values. Values are
# derived from the old combo presets but split per role; heavy 35B MoE
# (23 GB) must offload to CPU on a 10 GB 3080, so judge needs 6144 ctx,
# larger num_predict, higher retries/backoff, smaller batches and lower
# emotion caps. Light models (8b/9b/14b) fit fully at 8192 ctx.
#
# DISCOVERY_ROLE_KEYS and JUDGE_ROLE_KEYS enumerate which editable keys
# belong to each role; changing a model's assignment updates only its role.
DISCOVERY_ROLE_KEYS = ["DISCOVERY_NUM_CTX"]
JUDGE_ROLE_KEYS = [
    "JUDGE_NUM_CTX",
    "VERIFY_NUM_PREDICT",
    "AUDIO_SCAN_TITLE_NUM_PREDICT",
    "OLLAMA_RETRIES",
    "OLLAMA_RETRY_BACKOFF_SECONDS",
    "JUDGE_BATCH_SIZE",
    "VERIFY_BATCH_SIZE",
    "EMOTION_MAX_CANDIDATES",
    "EMOTION_BATCH_SIZE",
    "AUDIO_SCAN_MAX_CANDIDATES",
    "AUDIO_SCAN_TITLE_BATCH_SIZE",
]

# Discovery tunings: only DISCOVERY_NUM_CTX varies today, but keep as dict
# so future per-model discovery tuning (e.g. prompt size) can be added.
DISCOVERY_MODEL_TUNINGS: dict[str, dict] = {
    "qwen3:8b":                               {"DISCOVERY_NUM_CTX": 8192},
    "qwen3:14b-q4_K_M":                       {"DISCOVERY_NUM_CTX": 8192},
    "richardyoung/qwen3-14b-abliterated:IQ4_XS": {"DISCOVERY_NUM_CTX": 8192},
    "qwen3.5:9b-q4_K_M":                      {"DISCOVERY_NUM_CTX": 8192},
    "qwen3.6:35b-a3b":                        {"DISCOVERY_NUM_CTX": 6144},
    "qwen3.5:35b-a3b-q4_K_M":                 {"DISCOVERY_NUM_CTX": 6144},
    # phi4:14b is a dense 14B (Q4_K_M ~ 9.1 GB) - a 10GB card needs the
    # discovery window trimmed to leave room for KV cache. Pull it first:
    # ollama pull phi4:14b
    "phi4:14b":                               {"DISCOVERY_NUM_CTX": 6144},
    # qwen3.6-35b-a3b:iq4_xs is the lighter IQ4_XS quant of the 35B MoE
    # (~17 GB vs the 23 GB Q4/Q5 tag) - partial offload needs meaningfully
    # less system RAM than the bigger tag. Same MoE profile as the 35B
    # entries; matching is case-insensitive.
    "qwen3.6-35b-a3b:iq4_xs":                 {"DISCOVERY_NUM_CTX": 6144},
}

_JUDGE_LIGHT = {
    "JUDGE_NUM_CTX": 8192,
    "VERIFY_NUM_PREDICT": 1500,
    "AUDIO_SCAN_TITLE_NUM_PREDICT": 1200,
    "OLLAMA_RETRIES": 2,
    "OLLAMA_RETRY_BACKOFF_SECONDS": 5.0,
    "JUDGE_BATCH_SIZE": 20,
    "VERIFY_BATCH_SIZE": 10,
    "EMOTION_MAX_CANDIDATES": 250,
    "EMOTION_BATCH_SIZE": 4,
    "AUDIO_SCAN_MAX_CANDIDATES": 120,
    "AUDIO_SCAN_TITLE_BATCH_SIZE": 10,
}
_JUDGE_MEDIUM = {
    "JUDGE_NUM_CTX": 8192,
    "VERIFY_NUM_PREDICT": 1600,
    "AUDIO_SCAN_TITLE_NUM_PREDICT": 1300,
    "OLLAMA_RETRIES": 2,
    "OLLAMA_RETRY_BACKOFF_SECONDS": 5.0,
    "JUDGE_BATCH_SIZE": 16,
    "VERIFY_BATCH_SIZE": 10,
    "EMOTION_MAX_CANDIDATES": 230,
    "EMOTION_BATCH_SIZE": 3,
    "AUDIO_SCAN_MAX_CANDIDATES": 110,
    "AUDIO_SCAN_TITLE_BATCH_SIZE": 9,
}
_JUDGE_HEAVY = {
    "JUDGE_NUM_CTX": 6144,
    "VERIFY_NUM_PREDICT": 2200,
    "AUDIO_SCAN_TITLE_NUM_PREDICT": 1800,
    "OLLAMA_RETRIES": 3,
    "OLLAMA_RETRY_BACKOFF_SECONDS": 8.0,
    "JUDGE_BATCH_SIZE": 10,
    "VERIFY_BATCH_SIZE": 8,
    "EMOTION_MAX_CANDIDATES": 200,
    "EMOTION_BATCH_SIZE": 2,
    "AUDIO_SCAN_MAX_CANDIDATES": 100,
    "AUDIO_SCAN_TITLE_BATCH_SIZE": 8,
}

JUDGE_MODEL_TUNINGS: dict[str, dict] = {
    "qwen3:8b":                               dict(_JUDGE_LIGHT),
    "qwen3:14b-q4_K_M":                       dict(_JUDGE_MEDIUM),
    "richardyoung/qwen3-14b-abliterated:IQ4_XS": dict(_JUDGE_MEDIUM),
    "qwen3.5:9b-q4_K_M":                      dict(_JUDGE_LIGHT),
    "qwen3.6:35b-a3b":                        dict(_JUDGE_HEAVY),
    "qwen3.5:35b-a3b-q4_K_M":                 dict(_JUDGE_HEAVY),
    # phi4:14b judge uses the same VRAM-pressure profile as the 35B MoE:
    # smaller batches + more retries so the dense 14B's KV cache fits.
    "phi4:14b":                               dict(_JUDGE_HEAVY),
    "qwen3.6-35b-a3b:iq4_xs":                 dict(_JUDGE_HEAVY),
}

# Fallback tunings for models not explicitly listed (e.g. future pulls).
_FALLBACK_DISCOVERY_TUNING = {"DISCOVERY_NUM_CTX": 8192}
_FALLBACK_JUDGE_TUNING = dict(_JUDGE_LIGHT)


def tuning_for_role(model: str, role: str) -> dict:
    """Return role-specific tuning dict for *model*.

    *role* is ``"discovery"`` (MODEL) or ``"judge"`` (JUDGE_MODEL).
    Lookup is exact, then case-insensitive. Unknown models return a
    generic fallback that fits a 10 GB card.
    """
    name = str(model or "").strip()
    if not name:
        return {}
    if role == "discovery":
        for table in (DISCOVERY_MODEL_TUNINGS,):
            if name in table:
                return dict(table[name])
            low = name.lower()
            for key, val in table.items():
                if key.lower() == low:
                    return dict(val)
        return dict(_FALLBACK_DISCOVERY_TUNING)
    if role == "judge":
        for table in (JUDGE_MODEL_TUNINGS,):
            if name in table:
                return dict(table[name])
            low = name.lower()
            for key, val in table.items():
                if key.lower() == low:
                    return dict(val)
        return dict(_FALLBACK_JUDGE_TUNING)
    return {}


def discovery_tuning_for(model: str) -> dict:
    return tuning_for_role(model, "discovery")


def judge_tuning_for(model: str) -> dict:
    return tuning_for_role(model, "judge")


# Back-compat: old code imported PRESETS. Keep an empty mapping so
# ``cfg.PRESETS`` access does not crash; the GUI no longer uses it.
PRESETS: dict[str, dict] = {}


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

def llm_is_reachable(base_url: str, timeout: float = 3) -> bool:
    """True only when the active backend's server answers its health endpoint:
    Ollama /api/version, llama-server /health."""
    try:
        if LLM_BACKEND == "llamacpp":
            response = requests.get(base_url.rstrip("/") + "/health", timeout=timeout)
        else:
            response = requests.get(base_url.rstrip("/") + "/api/version", timeout=timeout)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def llm_not_ready_message(base_url: str) -> str:
    if LLM_BACKEND == "llamacpp":
        return (
            f"llama-server is not active at {base_url}.\n"
            "Close the RunAll GUI, launch llama-server (double-click Start_LlamaServer.bat "
            "in the VOD folder, or run it from the install folder), wait for it to finish "
            "starting, then run 6_RunAllSteps.bat again."
        )
    return ollama_not_ready_message(base_url)


def ollama_base_url(url: str) -> str:
    """Strip the /api/... path off an Ollama endpoint to get the server base."""
    return url.split("/api/", 1)[0]


def ollama_is_reachable(base_url: str, timeout: float = 3) -> bool:
    """True only when the Ollama server answers /api/version."""
    try:
        response = requests.get(base_url.rstrip("/") + "/api/version", timeout=timeout)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def ollama_not_ready_message(base_url: str) -> str:
    return (
        f"Ollama is not active at {base_url}.\n"
        "Close the RunAll GUI, launch Ollama manually, wait for it to finish "
        "starting, then run 6_RunAllSteps.bat again."
    )


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
    if kind in ("model", "backend"):
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

