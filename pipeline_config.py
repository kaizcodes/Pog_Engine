"""Shared tunables for the VOD highlight pipeline.

analyze_highlights_emotion.py imports from here. This file intentionally does
NOT hold machine-specific paths (whisper.cpp install location, local
emotion-model directory, gallery folder) - those stay hardcoded at the top of
the scripts that use them, since they're tied to this machine's setup rather
than something you'd tune between runs.

Everything here can still be overridden per-run with an environment variable
without editing this file, using the same os.environ.get(...) pattern the
original script already used for the emotion-scoring constants.
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
# JUDGE_MODEL is deliberately separate from MODEL: discovery just needs to
# read a transcript chunk and propose candidates, while verification/judging/
# titling need more careful structured reasoning over shorter, denser prompts.
#
# MODEL=qwen3:8b, JUDGE_MODEL=qwen3.5:9b-q4_K_M: both comfortably fit a 10GB
# 3080 at NUM_CTX below, without CPU offload. qwen3:14b-q4_K_M (previously
# JUDGE_MODEL) does not - roughly 8.3GB in weights alone before KV cache,
# which is what was making things slow.
MODEL = os.environ.get("HIGHLIGHT_MODEL", "qwen3:8b")
JUDGE_MODEL = os.environ.get("HIGHLIGHT_JUDGE_MODEL", "qwen3.5:9b-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
# qwen3.5 moved thinking control to Ollama's own renderer/parser instead of
# the old Jinja template that literally looked for "/no_think" in the
# prompt text - and as of Ollama 0.17.7 there's a confirmed bug
# (ollama/ollama#14793) where /api/generate ignores think:false entirely
# for qwen3.5: the model keeps thinking, burns the whole num_predict budget
# on the hidden reasoning trace, and "response" comes back empty no matter
# how large num_predict is. /api/chat with think:false as a top-level
# field is confirmed to work correctly. Any JUDGE_MODEL call that needs
# thinking OFF (title_audio_candidates, verify_candidates) must use this
# URL, not OLLAMA_URL - see ollama_generate(..., url=...) call sites.
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_RETRIES = _env_int("OLLAMA_RETRIES", 2)
OLLAMA_RETRY_BACKOFF_SECONDS = _env_float("OLLAMA_RETRY_BACKOFF_SECONDS", 5)
# Context window used by every Ollama call in the pipeline. Was 32768
# everywhere; at Q4_K_M an 8-9B model's KV cache alone costs roughly
# 4-4.5GB at that size, which is most of a 10GB card's remaining headroom
# after weights. None of this pipeline's actual prompts get close to
# needing it - discovery (the biggest, one 25-minute transcript chunk) is
# roughly 6-7K tokens even for a talkative streamer; verify/judge/titling
# batches are 1-2K. 8192 leaves real margin over that while freeing several
# GB of VRAM for the model itself.
NUM_CTX = _env_int("HIGHLIGHT_NUM_CTX", 8192)

# --- Output size / selection ------------------------------------------------
TOP_N = _env_int("HIGHLIGHT_TOP_N", 50)
JUDGE_POOL_SIZE = _env_int("HIGHLIGHT_JUDGE_POOL_SIZE", 100)
# Candidates are trimmed to this size *before* the content-verification pass
# now, not after - only JUDGE_POOL_SIZE of them ever reach judging anyway, so
# verifying more than ~1.5x that spends LLM calls on candidates that were
# never going to make the cut regardless of whether they pass verification.
VERIFY_POOL_SIZE = _env_int("HIGHLIGHT_VERIFY_POOL_SIZE", int(JUDGE_POOL_SIZE * 1.5))
VERIFY_BATCH_SIZE = _env_int("HIGHLIGHT_VERIFY_BATCH_SIZE", 10)
# A batch that fails to parse falls back to keeping its candidates
# unverified rather than dropping them (see verify_candidates() in
# analyze_highlights_emotion.py) - a reasonable safety net for one flaky
# batch, but if parsing is breaking down across most/all batches that same
# safety net quietly turns the whole verify stage into a no-op instead of
# actually verifying anything. Below this stage-wide ratio of successfully
# parsed verdicts to items sent, the result is treated as untrustworthy and
# is not checkpointed - see run_stage_verify().
VERIFY_MIN_COVERAGE_RATIO = _env_float("HIGHLIGHT_VERIFY_MIN_COVERAGE_RATIO", 0.5)
# Separated out from the titling/judge num_predict values (which stay
# inline in analyze_highlights_emotion.py) because this is the one you're
# most likely to want to raise if you see the 0-coverage warning above: a
# thinking-capable JUDGE_MODEL needs room to finish a reasoning trace
# before it gets to the actual PASS/FAIL lines, if /no_think isn't fully
# suppressing that trace.
VERIFY_NUM_PREDICT = _env_int("HIGHLIGHT_VERIFY_NUM_PREDICT", 1500)
JUDGE_BATCH_SIZE = _env_int("HIGHLIGHT_JUDGE_BATCH_SIZE", 20)

# --- Anti-hallucination ------------------------------------------------------
# Max allowed distance (seconds) between a candidate's claimed timestamp and
# the nearest timestamp that actually appears in the transcript. Anything
# further than this is treated as a hallucinated timestamp.
TIMESTAMP_TOLERANCE_SECONDS = _env_int("HIGHLIGHT_TIMESTAMP_TOLERANCE_SECONDS", 15)

# --- Speech-emotion sidecar (existing signal, unchanged) ---------------------
# EMOTION_LOCAL_MODEL_DIR / EMOTION_LOCAL_MODEL_FILE stay in
# analyze_highlights_emotion.py itself - see the note at the top of this file.
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
# Closes a gap in the old pipeline: candidates could previously only come from
# LLM discovery reading transcript text, so a pure reaction with no words (a
# scream, a long silence-then-yell) could never become a candidate no matter
# how loud it was. This scans the *entire* mic track with cheap signal
# processing (no model) to find energetic/fast moments, then only runs the
# (expensive) emotion model on the resulting shortlist. See
# find_audio_scan_candidates() in analyze_highlights_emotion.py.
AUDIO_SCAN_ENABLED = _env_bool("AUDIO_SCAN_ENABLED", True)
AUDIO_SCAN_HOP_SECONDS = _env_float("AUDIO_SCAN_HOP_SECONDS", 2.0)
# Two candidate peaks closer together than this collapse into one - stops one
# long scream from producing ten near-identical candidates.
AUDIO_SCAN_MIN_SEPARATION_SECONDS = _env_float("AUDIO_SCAN_MIN_SEPARATION_SECONDS", 20.0)
# How many standard deviations above this stream's *own* average loudness/
# rate a moment must be to qualify. Thresholding relative to the stream's own
# baseline (rather than a fixed dB/rate number) matters because streamers
# vary a lot in mic gain and baseline energy.
AUDIO_SCAN_MIN_ZSCORE = _env_float("AUDIO_SCAN_MIN_ZSCORE", 1.0)
# Cap on how many DSP-detected peaks get promoted to real candidates (i.e.
# get titled and run through the emotion model). Keeps the expensive stages
# bounded on long VODs.
AUDIO_SCAN_MAX_CANDIDATES = _env_int("AUDIO_SCAN_MAX_CANDIDATES", 120)
# If an LLM-discovered highlight already exists within this many seconds,
# skip adding a redundant audio-scan candidate there - audio scan is meant to
# fill gaps, not duplicate what discovery already found.
AUDIO_SCAN_SKIP_NEAR_EXISTING_SECONDS = _env_float("AUDIO_SCAN_SKIP_NEAR_EXISTING_SECONDS", 12.0)
# Composite score weighting: loudness is a cleaner signal than the onset-rate
# proxy (which reacts to any percussive sound, not just speech), so it gets
# more weight by default.
AUDIO_SCAN_LOUDNESS_WEIGHT = _env_float("AUDIO_SCAN_LOUDNESS_WEIGHT", 0.6)
AUDIO_SCAN_RATE_WEIGHT = _env_float("AUDIO_SCAN_RATE_WEIGHT", 0.4)
AUDIO_SCAN_TITLE_BATCH_SIZE = _env_int("AUDIO_SCAN_TITLE_BATCH_SIZE", 10)
# Was hardcoded inline in title_audio_candidates(); pulled out here for the
# same reason as VERIFY_NUM_PREDICT above - room to raise it if titles ever
# come back empty again. Now that titling calls JUDGE_MODEL through
# OLLAMA_CHAT_URL with think:false (see note above), this budget no longer
# has to cover a hidden reasoning trace, so 1200 should hold with margin.
AUDIO_SCAN_TITLE_NUM_PREDICT = _env_int("AUDIO_SCAN_TITLE_NUM_PREDICT", 1200)

# --- Preview clips (optional, off by default) --------------------------------
# Cuts a short mp4 around each final highlight into stream_folder/clips/ so
# you can scrub candidates on a phone/couch before opening Resolve. Uses
# stream copy (fast, no re-encode), so a clip's start can land on the nearest
# keyframe rather than the exact second - fine for a quick preview, not meant
# to replace the EDL for a real edit.
EXPORT_PREVIEW_CLIPS = _env_bool("EXPORT_PREVIEW_CLIPS", False)
PREVIEW_CLIP_SECONDS_BEFORE = _env_float("PREVIEW_CLIP_SECONDS_BEFORE", 5.0)
PREVIEW_CLIP_SECONDS_AFTER = _env_float("PREVIEW_CLIP_SECONDS_AFTER", 10.0)

# --- Run metadata -------------------------------------------------------------
RUN_INFO_FILENAME = "run_info.json"
# Unlike RUN_INFO_FILENAME, this is NOT written into the stream folder - see
# SCRIPT_DIR / record_pipeline_run_history() in analyze_highlights_emotion.py.
RUN_HISTORY_FILENAME = "pipeline_run_history.csv"
