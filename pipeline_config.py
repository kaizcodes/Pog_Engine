"""Shared tunables for the VOD highlight pipeline.

Imported by analyze_highlights_emotion.py. Machine-specific paths (whisper.cpp
install, emotion-model dir, gallery folder) are NOT here - they stay hardcoded
in the scripts that use them, since they're tied to this machine, not tunable
per-run. Everything below can still be overridden per-run via env var without
editing this file (same os.environ.get pattern the original script used).
"""

import os


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no"}


# --- Ollama models ---------------------------------------------------------
# JUDGE_MODEL is separate from MODEL: discovery just reads a transcript chunk
# and proposes candidates; verify/judge/titling need more careful structured
# reasoning over shorter, denser prompts.
# qwen3:8b + qwen3.5:9b-q4_K_M both fit a 10GB 3080 at NUM_CTX below with no
# CPU offload. qwen3:14b-q4_K_M (previous JUDGE_MODEL) doesn't - ~8.3GB in
# weights alone before KV cache, which was making things slow.
MODEL = os.environ.get("HIGHLIGHT_MODEL", "qwen3:8b")
JUDGE_MODEL = os.environ.get("HIGHLIGHT_JUDGE_MODEL", "qwen3.5:9b-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
# qwen3.5 moved thinking control to Ollama's renderer/parser instead of the
# old "/no_think"-in-prompt Jinja template. Confirmed Ollama 0.17.7 bug
# (ollama/ollama#14793): /api/generate ignores think:false for qwen3.5, so
# the model burns the whole num_predict budget on hidden reasoning and
# "response" comes back empty. /api/chat with top-level think:false works.
# Any JUDGE_MODEL call needing thinking OFF (title_audio_candidates,
# verify_candidates) must use this URL - see ollama_generate(..., url=...).
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_RETRIES = _env_int("OLLAMA_RETRIES", 2)
OLLAMA_RETRY_BACKOFF_SECONDS = _env_float("OLLAMA_RETRY_BACKOFF_SECONDS", 5)
# Was 32768 everywhere; at Q4_K_M an 8-9B model's KV cache alone costs ~4-4.5GB
# at that size - most of a 10GB card's headroom after weights. No prompt here
# needs it (discovery, the biggest, is ~6-7K tokens; others 1-2K), so 8192
# leaves margin while freeing VRAM for the model.
NUM_CTX = _env_int("HIGHLIGHT_NUM_CTX", 8192)

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
