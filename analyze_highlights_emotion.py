"""Emotion-enhanced highlight analyzer - discovery, full-file audio scan,
speech-emotion scoring, content verification, judging, and export, all in
one file/process.

Walks 6 checkpointed sub-stages (discovery -> audioscan -> emotion ->
verify -> judge -> export). Each saves its result to a JSON checkpoint in
the stream folder before the next, so a run that dies partway (crash,
Ollama timeout, Stop in the RunAll GUI) resumes with no wasted work:

  python analyze_highlights_emotion.py <folder>
      Runs every non-checkpointed stage in order through export. Used by
      5_AnalyzeHighlights.bat and 6_RunAllSteps.bat - safe to re-run.

  python analyze_highlights_emotion.py <folder> --stage verify
      Force-reruns ONE stage (used by the 5a-5f debug bats), clearing
      every checkpoint after it (else stale). For testing a prompt/logic
      change to one stage without redoing everything before it.

Checkpoint files (in the stream folder):
  checkpoint_1_discovery.json  - after discovery (raw, deduped/filtered)
  checkpoint_2_audioscan.json  - after audio scan (+ candidates, merged)
  checkpoint_3_emotion.json    - after emotion scoring (+ scores, sorted)
  checkpoint_4_verified.json   - after content verification
  checkpoint_5_judged.json     - after judging (final ranked top N)
  pipeline_stats.json          - running Ollama call/timing totals + the
                                  last-completed-stage marker, read by
                                  export for the final run summary.
  log_<stage>.txt              - full console output for that stage,
                                  appended across every run/resume.

discovery, audioscan, and verify each sanity-check their result before
checkpointing: if it looks like a broken run (Ollama unreachable, a
response that stopped parsing entirely) rather than a stream that
legitimately had nothing to offer, the stage exits WITHOUT writing its
checkpoint, so the next run retries instead of treating a bad result as
done. Only guards future runs - a pre-existing empty checkpoint from
before this check won't retroactively clear, so a stuck folder still
needs one manual nudge (delete the checkpoint_*.json, or run its 5a/5b/5d
debug bat).

pipeline_run_history.csv is NOT in the stream folder - it lives next to
this script (SCRIPT_DIR) and gets one row per normal (non --stage) run
that reaches the end, accumulating per-stage/total timing across every
stream processed. See record_pipeline_run_history().

See STAGE_ORDER / STAGE_FUNCS near the bottom for stage dispatch, and
invalidate_downstream() for how forced single-stage reruns keep the
checkpoint chain consistent.
"""

import re
import requests
import os
import csv
import sys
import time
import json
import subprocess
import contextlib
import bisect
import difflib
import io
import argparse
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path

from pipeline_config import (
    MODEL, JUDGE_MODEL, NUM_CTX, OLLAMA_URL, OLLAMA_CHAT_URL,
    OLLAMA_RETRIES, OLLAMA_RETRY_BACKOFF_SECONDS,
    OLLAMA_SERVE, OLLAMA_START_ON_CONNECTION_ERROR,
    OLLAMA_SERVE_READY_TIMEOUT_SECONDS,
    TOP_N, JUDGE_POOL_SIZE, VERIFY_POOL_SIZE, VERIFY_BATCH_SIZE, VERIFY_MIN_COVERAGE_RATIO,
    VERIFY_NUM_PREDICT, JUDGE_BATCH_SIZE,
    TIMESTAMP_TOLERANCE_SECONDS,
    EMOTION_MODEL_ID, EMOTION_SCORES_CSV, EMOTION_WINDOW_SECONDS,
    EMOTION_MODEL_INPUT_SECONDS, EMOTION_CONFIDENCE_FLOOR, EMOTION_MAX_CANDIDATES,
    EMOTION_BATCH_SIZE, EMOTION_USE_FP16, EMOTION_ENABLED, EMOTION_BOOSTS,
    AUDIO_SCAN_ENABLED, AUDIO_SCAN_HOP_SECONDS,
    AUDIO_SCAN_MIN_SEPARATION_SECONDS, AUDIO_SCAN_MIN_ZSCORE, AUDIO_SCAN_MAX_CANDIDATES,
    AUDIO_SCAN_SKIP_NEAR_EXISTING_SECONDS, AUDIO_SCAN_LOUDNESS_WEIGHT,
    AUDIO_SCAN_RATE_WEIGHT, AUDIO_SCAN_TITLE_BATCH_SIZE, AUDIO_SCAN_TITLE_NUM_PREDICT,
    EXPORT_PREVIEW_CLIPS, PREVIEW_CLIP_SECONDS_BEFORE, PREVIEW_CLIP_SECONDS_AFTER,
    RUN_INFO_FILENAME, RUN_HISTORY_FILENAME,
)

# Folder this script lives in (vs stream_folder, the VOD folder passed on
# the command line) - constant since .bat launchers always invoke this
# same script in place, never a per-VOD copy. Used only for
# pipeline_run_history.csv, deliberately kept next to the scripts (not
# per-VOD) so it accumulates timing across every run - see
# record_pipeline_run_history().
SCRIPT_DIR = Path(__file__).resolve().parent

EMOTION_HALF_WINDOW_SECONDS = EMOTION_WINDOW_SECONDS / 2

# Machine-specific paths - deliberately kept hardcoded here rather than in
# pipeline_config.py. Edit these directly if your whisper-cublas install moves.
EMOTION_LOCAL_MODEL_DIR = r"G:\pog_dev\models"
EMOTION_LOCAL_MODEL_FILE = r"G:\pog_dev\models\speech-emotion-recognition-with-openai-whisper-large-v3.safetensors"

# Running totals for THIS PROCESS's Ollama usage (see ollama_generate()
# below). Each stage merges its own contribution into the persistent
# pipeline_stats.json at the end - see record_stage_stats().
CALL_STATS = {"ollama_calls": 0, "ollama_seconds": 0.0, "ollama_retries": 0}
AUDIO_SCAN_STATS = {"candidates_found": 0, "candidates_kept": 0}
PROCESS_START_TIME = time.time()

# Handle of the last `ollama serve` this process spawned via
# ensure_ollama_running(); used to avoid double-starting and to detect that
# a spawned server died before becoming ready.
_OLLAMA_SERVE_PROC = None

# --- Checkpoint I/O -----------------------------------------------------
# Makes each stage resumable: every stage function checks checkpoint_exists()
# for its own output before doing work, and every RunAll GUI re-run relies
# on the same files to decide what to skip.

def checkpoint_path(stream_folder, name):
    return os.path.join(stream_folder, f"checkpoint_{name}.json")

def checkpoint_exists(stream_folder, name):
    return os.path.exists(checkpoint_path(stream_folder, name))

def save_checkpoint(stream_folder, name, data):
    """Atomic write: build the file fully as a .tmp, then os.replace() it
    into place. os.replace is atomic on both Windows and POSIX, so a
    process killed mid-write (e.g. the GUI Stop button) can never leave a
    half-written, corrupt checkpoint behind - either the old file is still
    there, or the fully-written new one is."""
    path = checkpoint_path(stream_folder, name)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)
    return path

def load_checkpoint(stream_folder, name):
    path = checkpoint_path(stream_folder, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def require_checkpoint(stream_folder, name, stage_label):
    """Load a checkpoint that this stage depends on, or exit with a clear
    error instead of crashing on a missing-file traceback. Used at the top
    of every stage function after the first (discovery has no checkpoint
    input, just the transcript_part files)."""
    data = load_checkpoint(stream_folder, name)
    if data is None:
        print(f"[{stage_label}] Missing required checkpoint: {checkpoint_path(stream_folder, name)}")
        print(f"[{stage_label}] Run the earlier steps first (or run RunAllSteps.bat, which runs them in order).")
        sys.exit(1)
    return data

# Canonical stage order and which checkpoint each one produces. "export" has
# no checkpoint of its own - its output IS the final CSV/EDL/run_info.json.
STAGE_ORDER = ["discovery", "audioscan", "emotion", "verify", "judge", "export"]
STAGE_CHECKPOINT_NAMES = {
    "discovery": "1_discovery",
    "audioscan": "2_audioscan",
    "emotion": "3_emotion",
    "verify": "4_verified",
    "judge": "5_judged",
}
STAGE_LABELS = {
    "discovery": "5a. Discovery",
    "audioscan": "5b. Audio Scan",
    "emotion": "5c. Emotion Scoring",
    "verify": "5d. Verification",
    "judge": "5e. Judging",
    "export": "5f. Export",
}

def invalidate_downstream(stream_folder, from_stage):
    """Deletes the checkpoint for from_stage and every stage after it (plus
    the final CSV/EDL, which are derived from the judge checkpoint), since
    they're now stale relative to a forced re-run of from_stage. Used when a
    single stage is force-rerun directly (via a debug bat's --stage flag)
    rather than through the normal skip-if-done RunAll flow.
    """
    idx = STAGE_ORDER.index(from_stage)
    removed = []
    for stage in STAGE_ORDER[idx:]:
        name = STAGE_CHECKPOINT_NAMES.get(stage)
        if name:
            path = checkpoint_path(stream_folder, name)
            if os.path.exists(path):
                os.remove(path)
                removed.append(os.path.basename(path))

    for path in Path(stream_folder).glob("top*_highlights.csv"):
        path.unlink()
        removed.append(path.name)
    for path in Path(stream_folder).glob("top*_markers.edl"):
        path.unlink()
        removed.append(path.name)

    if removed:
        print(f"Invalidated {len(removed)} stale downstream file(s): {', '.join(removed)}")

# --- Pipeline-wide stats (Ollama calls, per-stage timing) ----------------
# Persisted across stages/processes so stage 10 (export) can report a full
# run summary even though no single process saw the whole pipeline.

def load_pipeline_stats(stream_folder):
    path = os.path.join(stream_folder, "pipeline_stats.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "ollama_calls": 0, "ollama_seconds": 0.0, "ollama_retries": 0,
        "audio_scan_candidates_found": 0, "audio_scan_candidates_kept": 0,
        "stage_seconds": {},
    }

def save_pipeline_stats(stream_folder, stats):
    path = os.path.join(stream_folder, "pipeline_stats.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    os.replace(tmp_path, path)

def record_stage_stats(stream_folder, stage_name, stage_seconds):
    """Call at the end of every stage that finishes successfully: folds this
    stage's Ollama call stats and wall-clock time into the shared
    pipeline_stats.json, and marks this stage as the last one to fully
    complete. Only ever called after a stage's checkpoint has already been
    written, so last_completed_stage is always trustworthy - if the process
    gets killed mid-stage, this line never runs and the previous stage
    remains "last completed", which is exactly correct.
    """
    stats = load_pipeline_stats(stream_folder)
    stats["ollama_calls"] += CALL_STATS["ollama_calls"]
    stats["ollama_seconds"] += CALL_STATS["ollama_seconds"]
    stats["ollama_retries"] += CALL_STATS["ollama_retries"]
    if AUDIO_SCAN_STATS.get("candidates_found"):
        stats["audio_scan_candidates_found"] += AUDIO_SCAN_STATS["candidates_found"]
        stats["audio_scan_candidates_kept"] += AUDIO_SCAN_STATS["candidates_kept"]
    # CALL_STATS/AUDIO_SCAN_STATS are process-wide running totals; reset them
    # after folding so a later stage in the SAME process (the normal
    # RunAll/5_AnalyzeHighlights flow) doesn't re-add this stage's counts.
    # Each call therefore records only the delta since the last successful
    # stage end - without this, single-process runs inflated the persisted
    # totals at every stage boundary (e.g. audioscan 7/4 recorded as 35/20,
    # 31 Ollama calls recorded for 7 actual).
    CALL_STATS.update(ollama_calls=0, ollama_seconds=0.0, ollama_retries=0)
    AUDIO_SCAN_STATS.update(candidates_found=0, candidates_kept=0)
    stats.setdefault("stage_seconds", {})[stage_name] = round(stage_seconds, 1)
    stats["last_completed_stage"] = stage_name
    stats["last_completed_at"] = datetime.now().isoformat(timespec="seconds")
    save_pipeline_stats(stream_folder, stats)
    print(f"[{stage_name}] Stage finished in {stage_seconds:.1f}s")

class _Tee:
    """Duplicates writes to multiple streams - used to make every stage's
    console output also land in a persistent per-stage log file, without
    having to route every print() call through something new."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

@contextlib.contextmanager
def stage_log(stream_folder, stage_name):
    """Wrap a stage's work in `with stage_log(folder, "verify"):` and every
    print() during that block also gets appended to log_verify.txt in the
    stream folder - so the data needed to debug a failed/stopped run is
    still there even after the console window that ran it is long closed,
    whether that stage was invoked via RunAll, a debug bat, or the GUI.
    """
    log_path = os.path.join(stream_folder, f"log_{stage_name}.txt")
    f = open(log_path, "a", encoding="utf-8")
    f.write(f"\n{'=' * 60}\nRun started: {datetime.now().isoformat(timespec='seconds')}\n{'=' * 60}\n")
    f.flush()
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, f)
    try:
        yield log_path
    except Exception as exc:
        f.write(f"\n[!] Stage raised an exception: {exc}\n")
        raise
    finally:
        sys.stdout = original_stdout
        f.close()

# --- Transcript utilities shared across stages ---------------------------
# Stages after discovery run as separate processes, so they can't share its
# in-memory transcript_blocks_by_part dict - rebuilt fresh here instead by
# re-parsing the already-split transcript_part*.txt files (cheap regardless
# of VOD length).

def list_transcript_parts(stream_folder):
    parts = sorted(
        [f for f in os.listdir(stream_folder) if re.match(r'^transcript_part\d+\.txt$', f)],
        key=lambda f: int(re.search(r'\d+', f).group())
    )
    return parts

def build_transcript_blocks_by_part(stream_folder):
    blocks_by_part = {}
    for part in list_transcript_parts(stream_folder):
        path = os.path.join(stream_folder, part)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            transcript = f.read()
        blocks_by_part[part] = parse_srt_blocks(transcript)
    return blocks_by_part

def _ollama_base_url(url):
    """Strip the /api/... path off an Ollama endpoint to get the server
    base (http://host:port) used for health checks."""
    return url.split("/api/", 1)[0]


def ollama_is_reachable(base_url, timeout=3):
    """True if an Ollama server answers /api/version at base_url. Any HTTP
    response counts (a busy server is still up); only connection-level
    failures mean there is no server there."""
    try:
        requests.get(base_url.rstrip("/") + "/api/version", timeout=timeout)
        return True
    except requests.exceptions.RequestException:
        return False


def ensure_ollama_running(url):
    """If no Ollama server is answering at url, start one (`ollama serve`,
    detached so it outlives this process) and wait until it accepts
    connections. Keeps a pipeline run from dying because the user forgot to
    launch Ollama first.

    Only auto-starts for a local server (localhost/127.0.0.1/::1) - a remote
    OLLAMA_URL can't be fixed by spawning a process here. Keeps at most one
    spawned server alive per process; re-spawns only if a previous one
    exited, and reports if that one dies before becoming ready.
    """
    base_url = _ollama_base_url(url)
    if ollama_is_reachable(base_url):
        return True
    if not OLLAMA_START_ON_CONNECTION_ERROR:
        return False
    host = urlparse(url).hostname
    if host not in ("localhost", "127.0.0.1", "::1"):
        print(f"     [!] Ollama not reachable at {base_url} and host {host!r} is not local - not auto-starting.")
        return False

    global _OLLAMA_SERVE_PROC
    if _OLLAMA_SERVE_PROC is not None and _OLLAMA_SERVE_PROC.poll() is None:
        # We already launched one and it is still alive; it just isn't
        # answering yet (or just went down). Wait on it, don't re-spawn.
        pass
    else:
        print(f"     [!] Ollama is not running; starting it ({OLLAMA_SERVE} serve)...")
        try:
            flags = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            _OLLAMA_SERVE_PROC = subprocess.Popen(
                [OLLAMA_SERVE, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
                close_fds=True,
            )
        except Exception as exc:
            print(f"     [WARN] Could not launch {OLLAMA_SERVE!r} serve: {exc}")
            return False

    deadline = time.time() + OLLAMA_SERVE_READY_TIMEOUT_SECONDS
    while time.time() < deadline:
        if ollama_is_reachable(base_url):
            print(f"     [OK] Ollama is up at {base_url}.")
            return True
        if _OLLAMA_SERVE_PROC is not None and _OLLAMA_SERVE_PROC.poll() is not None:
            print(f"     [WARN] {OLLAMA_SERVE!r} serve exited with code {_OLLAMA_SERVE_PROC.returncode} before becoming ready.")
            return False
        time.sleep(0.5)
    print(f"     [WARN] Ollama still not answering at {base_url} after {OLLAMA_SERVE_READY_TIMEOUT_SECONDS:.0f}s.")
    return False


def ollama_generate(payload, timeout, url=None):
    """POST to an Ollama endpoint (default /api/generate) with a couple of
    retries.

    A transient hiccup (Ollama busy loading a different model, a brief
    connection reset) used to just silently reduce candidate coverage for
    whichever part/batch was in flight, since the raw requests.post() call
    had no retry. This wraps the one HTTP call every stage of the pipeline
    makes, so a retry only has to be written once. Also tracks call count and
    elapsed time centrally for the end-of-run summary.

    url defaults to OLLAMA_URL (/api/generate) but callers that need
    thinking reliably disabled on JUDGE_MODEL should pass OLLAMA_CHAT_URL
    instead - see the OLLAMA_CHAT_URL note in pipeline_config.py for why
    /api/generate's think:false can't be trusted for qwen3.5.
    """
    if url is None:
        url = OLLAMA_URL
    last_exc = None
    for attempt in range(OLLAMA_RETRIES + 1):
        start = time.time()
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            CALL_STATS["ollama_calls"] += 1
            CALL_STATS["ollama_seconds"] += time.time() - start
            return response
        except Exception as exc:
            last_exc = exc
            if isinstance(exc, requests.exceptions.ConnectionError):
                ensure_ollama_running(url)
            if attempt < OLLAMA_RETRIES:
                CALL_STATS["ollama_retries"] += 1
                print(f"     [!] Ollama call failed ({exc}); retrying in {OLLAMA_RETRY_BACKOFF_SECONDS:.0f}s...")
                time.sleep(OLLAMA_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_exc

def find_mic_wav(stream_folder):
    """Find the mic WAV produced by step 1."""
    candidates = []
    for name in os.listdir(stream_folder):
        lowered = name.lower()
        if lowered.endswith("_mic.wav"):
            path = os.path.join(stream_folder, name)
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def find_source_video(stream_folder):
    """Find the original recording (mp4/mkv/mov) that step 0 moved into this
    folder, so preview clips can be cut with picture, not just the extracted
    mono mic track."""
    candidates = []
    for name in os.listdir(stream_folder):
        if name.lower().endswith((".mp4", ".mkv", ".mov")):
            candidates.append(os.path.join(stream_folder, name))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def export_preview_clips(stream_folder, highlights, before_seconds, after_seconds):
    """Cuts a short clip around each final highlight into stream_folder/clips/
    so candidates can be scrubbed on a phone/couch before opening Resolve.

    Uses stream copy (-c copy) rather than re-encoding, so this is fast, but
    it means a clip's actual start can land on the nearest keyframe rather
    than the exact requested second - fine for a quick preview, not a
    substitute for the frame-accurate EDL import for a real edit.
    """
    video_path = find_source_video(stream_folder)
    if not video_path:
        print("[preview] No source video (.mp4/.mkv/.mov) found in folder; skipping preview clip export.")
        return

    clips_dir = os.path.join(stream_folder, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    exported = 0
    for rank, h in enumerate(highlights, start=1):
        center_seconds = timestamp_to_seconds(h["Timestamp"])
        start_seconds = max(0, center_seconds - before_seconds)
        duration_seconds = before_seconds + after_seconds

        safe_title = re.sub(r'[\\/:*?"<>|]', "_", h["Title"]).strip()[:40] or "clip"
        out_name = f"{rank:02d}_{safe_title}.mp4"
        out_path = os.path.join(clips_dir, out_name)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_seconds),
            "-i", video_path,
            "-t", str(duration_seconds),
            "-c", "copy",
            out_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            exported += 1
        except FileNotFoundError:
            print("[preview] ffmpeg not found on PATH; skipping preview clip export.")
            return
        except subprocess.CalledProcessError as exc:
            print(f"[preview] Failed to cut clip for rank {rank} ({h['Title']}): {exc.stderr.strip()[-300:]}")

    if exported:
        print(f"[preview] Saved {exported} preview clip(s) to: {clips_dir}")

def unload_ollama_model(model_name):
    """Ask Ollama to unload a model so CUDA VRAM is available for emotion scoring."""
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": "", "keep_alive": 0},
            timeout=30,
        )
        print(f"[emotion] Requested Ollama unload for {model_name}")
    except Exception as exc:
        print(f"[emotion] Could not unload Ollama model {model_name}: {exc}")

def emotion_boost(label, confidence):
    label = (label or "").strip().lower()
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    if confidence < EMOTION_CONFIDENCE_FLOOR:
        return 0.0
    boost = EMOTION_BOOSTS.get(label, 0.0)
    if boost and confidence >= 0.80:
        boost += 0.5
    return min(boost, 2.0)

def load_existing_emotion_scores(csv_path):
    if not os.path.exists(csv_path):
        return {}
    loaded = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = (row.get("Timestamp") or "").strip()
            if not timestamp:
                continue
            loaded[timestamp] = row
    return loaded

def classify_candidate_emotions(stream_folder, highlights):
    """Classify audio around candidate timestamps and write emotion_scores.csv."""
    csv_path = os.path.join(stream_folder, EMOTION_SCORES_CSV)
    existing = load_existing_emotion_scores(csv_path)
    if existing:
        print(f"Loaded existing emotion scores: {csv_path}")
        return existing

    if not EMOTION_ENABLED:
        print("Emotion scoring disabled by DISABLE_EMOTION_SCORING.")
        return {}

    mic_wav = find_mic_wav(stream_folder)
    if not mic_wav:
        print("[emotion] No *_mic.wav file found; continuing without audio emotion scoring.")
        return {}

    try:
        import numpy as np
        import torch
        import librosa
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification
        from safetensors.torch import load_file as load_safetensors_file
    except Exception as exc:
        print("[emotion] Missing optional dependencies; continuing without emotion scoring.")
        print("[emotion] Install if wanted: pip install torch transformers librosa soundfile")
        print("[emotion]", exc)
        return {}

    print("=" * 60)
    print("Speech Emotion Scoring")
    print("=" * 60)
    print("Model:", EMOTION_MODEL_ID)
    print("Audio:", mic_wav)
    unload_ollama_model(MODEL)
    unload_ollama_model(JUDGE_MODEL)
    print("Output:", csv_path)

    try:
        local_model_dir = os.environ.get("EMOTION_MODEL_DIR", EMOTION_LOCAL_MODEL_DIR)
        local_model_file = os.environ.get("EMOTION_MODEL_FILE", EMOTION_LOCAL_MODEL_FILE)
        local_model_dir_ready = (
            local_model_dir
            and os.path.exists(os.path.join(local_model_dir, "config.json"))
            and os.path.exists(os.path.join(local_model_dir, "preprocessor_config.json"))
        )
        model_source = local_model_dir if local_model_dir_ready else EMOTION_MODEL_ID
        print("[emotion] Model metadata source:", model_source)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_source, do_normalize=True)
        config = AutoConfig.from_pretrained(model_source)

        if local_model_file and os.path.exists(local_model_file):
            print("[emotion] Loading local safetensors weights:", local_model_file)
            model = AutoModelForAudioClassification.from_config(config)
            state_dict = load_safetensors_file(local_model_file, device="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"[emotion] Local model missing {len(missing_keys)} key(s); continuing with loaded weights.")
            if unexpected_keys:
                print(f"[emotion] Local model had {len(unexpected_keys)} unexpected key(s); continuing with loaded weights.")
        else:
            model = AutoModelForAudioClassification.from_pretrained(model_source)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        use_fp16 = device.type == "cuda" and EMOTION_USE_FP16
        if use_fp16:
            model = model.half()
        print(f"[emotion] Device: {device}; batch size: {EMOTION_BATCH_SIZE}; fp16: {use_fp16}")
        model.eval()
        sample_rate = feature_extractor.sampling_rate
        audio, _ = librosa.load(mic_wav, sr=sample_rate, mono=True)
    except Exception as exc:
        print("[emotion] Failed to load emotion model/audio; continuing without emotion scoring.")
        print("[emotion]", exc)
        return {}

    timestamps = sorted({h["Timestamp"] for h in highlights}, key=timestamp_to_seconds)
    if len(timestamps) > EMOTION_MAX_CANDIDATES:
        ranked = sorted(highlights, key=lambda h: h["Score"], reverse=True)
        timestamps = sorted({h["Timestamp"] for h in ranked[:EMOTION_MAX_CANDIDATES]}, key=timestamp_to_seconds)
        print(f"[emotion] Limiting scoring to top {len(timestamps)} candidate timestamps.")

    labels = [model.config.id2label.get(index, model.config.id2label.get(str(index), str(index))) for index in range(len(model.config.id2label))]
    max_length = getattr(feature_extractor, "n_samples", int(sample_rate * EMOTION_MODEL_INPUT_SECONDS))
    rows = []

    batch_items = []
    for timestamp in timestamps:
        center_seconds = timestamp_to_seconds(timestamp)
        start_seconds = max(0.0, center_seconds - EMOTION_HALF_WINDOW_SECONDS)
        end_seconds = start_seconds + EMOTION_WINDOW_SECONDS
        start_sample = int(start_seconds * sample_rate)
        end_sample = int(end_seconds * sample_rate)
        segment = audio[start_sample:end_sample]

        if segment.size == 0:
            continue
        if segment.shape[0] < max_length:
            segment = np.pad(segment, (0, max_length - segment.shape[0]))
        else:
            segment = segment[:max_length]

        batch_items.append((timestamp, start_seconds, end_seconds, segment))

    for batch_start in range(0, len(batch_items), EMOTION_BATCH_SIZE):
        batch = batch_items[batch_start:batch_start + EMOTION_BATCH_SIZE]
        inputs = feature_extractor(
            [item[3] for item in batch],
            sampling_rate=sample_rate,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        prepared_inputs = {}
        for key, value in inputs.items():
            value = value.to(device)
            if use_fp16 and torch.is_floating_point(value):
                value = value.half()
            prepared_inputs[key] = value

        with torch.inference_mode():
            outputs = model(**prepared_inputs)
            probability_rows = torch.softmax(outputs.logits, dim=-1).detach().cpu().tolist()

        for offset, ((timestamp, start_seconds, end_seconds, _segment), probabilities) in enumerate(zip(batch, probability_rows), start=1):
            processed = batch_start + offset
            best_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
            emotion = labels[best_index]
            confidence = probabilities[best_index]

            row = {
                "Timestamp": timestamp,
                "Start": format_plain_seconds(start_seconds),
                "End": format_plain_seconds(end_seconds),
                "Emotion": emotion,
                "Confidence": f"{confidence:.4f}",
            }
            for label, probability in zip(labels, probabilities):
                row[label] = f"{probability:.4f}"
            rows.append(row)

            if processed == 1 or processed % 10 == 0 or processed == len(batch_items):
                print(f"[emotion] {processed}/{len(batch_items)} {timestamp}: {emotion} {confidence:.2f}")

    if not rows:
        return {}

    fieldnames = ["Timestamp", "Start", "End", "Emotion", "Confidence", *labels]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[emotion] Saved {len(rows)} emotion score row(s).")

    try:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {row["Timestamp"]: row for row in rows}

def format_plain_seconds(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def apply_emotion_scores_to_highlights(highlights, stream_folder):
    if not highlights:
        return

    scores_by_timestamp = classify_candidate_emotions(stream_folder, highlights)
    if not scores_by_timestamp:
        for h in highlights:
            h.setdefault("TranscriptScore", h["Score"])
            h.setdefault("Emotion", "")
            h.setdefault("EmotionConfidence", "")
            h.setdefault("EmotionBoost", 0.0)
        return

    boosted = 0
    for h in highlights:
        h["TranscriptScore"] = h["Score"]
        row = scores_by_timestamp.get(h["Timestamp"])
        if not row:
            h["Emotion"] = ""
            h["EmotionConfidence"] = ""
            h["EmotionBoost"] = 0.0
            continue

        emotion = row.get("Emotion", "")
        confidence = float(row.get("Confidence") or 0.0)
        boost = emotion_boost(emotion, confidence)
        h["Emotion"] = emotion
        h["EmotionConfidence"] = f"{confidence:.2f}"
        h["EmotionBoost"] = boost
        if boost > 0:
            h["Score"] = int(min(10, max(1, round(h["Score"] + boost))))
            boosted += 1

    print(f"Emotion scoring attached to {len(scores_by_timestamp)} timestamp(s); boosted {boosted} candidate(s).")

def timestamp_to_seconds(ts):
    h, m, s = map(int, ts.split(":"))
    return h * 3600 + m * 60 + s

def extract_transcript_timestamps(transcript_text):
    """Returns a sorted list of all HH:MM:SS timestamps (in seconds)
    that actually appear in the transcript text."""
    raw = re.findall(r'(\d{2}:\d{2}:\d{2})', transcript_text)
    seconds = sorted(set(timestamp_to_seconds(ts) for ts in raw))
    return seconds

def nearest_distance(target_seconds, sorted_seconds_list):
    """Returns distance in seconds to the closest real transcript timestamp."""
    if not sorted_seconds_list:
        return None

    pos = bisect.bisect_left(sorted_seconds_list, target_seconds)

    candidates = []
    if pos < len(sorted_seconds_list):
        candidates.append(sorted_seconds_list[pos])
    if pos > 0:
        candidates.append(sorted_seconds_list[pos - 1])

    return min(abs(target_seconds - c) for c in candidates)

def similar_titles(title_a, title_b, threshold=0.75):
    """Fuzzy similarity check for near-duplicate detection."""
    a = title_a.strip().lower()
    b = title_b.strip().lower()
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold

def merge_near_duplicates(highlights_list, time_window=10):
    """Merges highlights that are within `time_window` seconds of each
    other AND have similar titles, keeping only the highest-scored one
    from each near-duplicate group."""

    sorted_highlights = sorted(
        highlights_list, key=lambda h: timestamp_to_seconds(h["Timestamp"])
    )

    kept = []

    for h in sorted_highlights:
        h_seconds = timestamp_to_seconds(h["Timestamp"])
        merged_into_existing = False

        for existing in kept:
            existing_seconds = timestamp_to_seconds(existing["Timestamp"])

            if abs(h_seconds - existing_seconds) <= time_window and similar_titles(
                h["Title"], existing["Title"]
            ):
                if h["Score"] > existing["Score"]:
                    existing.update(h)
                merged_into_existing = True
                break

        if not merged_into_existing:
            kept.append(h)

    return kept

def parse_srt_blocks(transcript_text):
    """Parses the transcript_partN.txt block format produced by
    split_srt_into_chunks() in OrganizeVODAndFixSRT_Emotion.py:

        [HH:MM:SS]
        subtitle text here

    separated by blank lines. Returns a list of (start_seconds, text) tuples.
    Note: this format has no end time, only a start timestamp per block."""
    blocks = []

    pattern = re.compile(
        r'\[(\d{2}):(\d{2}):(\d{2})\]\s*\n(.*?)(?=\n\s*\n|\Z)',
        re.DOTALL
    )

    for m in pattern.finditer(transcript_text):
        h, mi, s, text = m.groups()
        start = int(h) * 3600 + int(mi) * 60 + int(s)
        blocks.append((start, text.strip().replace("\n", " ")))

    blocks.sort(key=lambda b: b[0])
    return blocks

def get_snippet(blocks, target_seconds, window=20):
    """Returns the joined transcript text for all blocks whose timestamp
    falls within +/- window seconds of target_seconds."""
    if not blocks:
        return ""

    matched = [
        text for (start, text) in blocks
        if abs(start - target_seconds) <= window
    ]

    return " / ".join(matched)

# --- Full-file audio scan: a second, independent candidate source ----------
# Everything above can only produce a candidate if it starts as text in a
# transcript_part file and gets picked by an LLM discovery prompt - so a
# pure reaction with no words (a scream, silence-then-yell, laughter with
# no dialogue) could never become one, no matter how loud, since there was
# no transcript text for the model to read.
#
# This closes that gap by scanning the *entire* mic track directly:
#   1. Cheap, model-free signal processing (loudness + a speech-rate proxy)
#      sweeps the file for energetic/fast moments.
#   2. Peaks not already close to an LLM-discovered candidate become new
#      candidates in their own right.
#   3. New candidates merge into `highlights` *before*
#      apply_emotion_scores_to_highlights() runs, so the existing
#      speech-emotion model scores them too automatically - no changes
#      needed there, it just sees more timestamps.

def _zscore(values):
    """Standardize a 1-D numpy array against its own mean/std. Thresholding
    relative to a stream's own baseline (rather than a fixed dB/rate number)
    matters because streamers vary a lot in mic gain and baseline energy."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    std = float(values.std())
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - values.mean()) / std

def compute_audio_arousal_series(audio, sample_rate):
    """Sweeps the whole track, returning per hop:
      - times: seconds from file start
      - composite: weighted loudness+rate z-score (the peak-picking signal)
      - loudness_z / rate_z: the two components, kept separate for scoring

    Loudness is RMS energy. "Speech rate" has no ground truth here (no
    word-level alignment), so it's approximated by how much the energy
    envelope fluctuates *within* each hop (split into sub-frames) - a
    fast/emphatic proxy, not a real words-per-minute measurement.

    Processes one hop at a time instead of framing the whole track into one
    array: librosa.feature.rms / onset.onset_strength build a dense
    (frame_length x n_frames) matrix internally, which is gigabytes for a
    multi-hour VOD at multi-second frames (this was crashing). Per-hop
    processing uses O(hop_length) memory regardless of stream length.
    """
    import numpy as np

    hop_length = max(1, int(AUDIO_SCAN_HOP_SECONDS * sample_rate))
    sub_frames = 8  # sub-divisions per hop, used for the rate proxy

    n_hops = len(audio) // hop_length
    if n_hops < 2:
        empty = np.array([])
        return empty, empty, empty, empty

    rms = np.empty(n_hops, dtype=np.float64)
    rate_proxy = np.empty(n_hops, dtype=np.float64)
    sub_len = hop_length // sub_frames

    for i in range(n_hops):
        block = audio[i * hop_length: (i + 1) * hop_length]
        rms[i] = np.sqrt(np.mean(np.square(block, dtype=np.float64)))

        if sub_len > 0:
            usable = sub_frames * sub_len
            sub_blocks = block[:usable].reshape(sub_frames, sub_len)
            sub_rms = np.sqrt(np.mean(np.square(sub_blocks, dtype=np.float64), axis=1))
            rate_proxy[i] = np.mean(np.abs(np.diff(sub_rms)))
        else:
            rate_proxy[i] = 0.0

    times = np.arange(n_hops) * (hop_length / sample_rate)

    loudness_z = _zscore(rms)
    rate_z = _zscore(rate_proxy)
    composite = AUDIO_SCAN_LOUDNESS_WEIGHT * loudness_z + AUDIO_SCAN_RATE_WEIGHT * rate_z

    return times, composite, loudness_z, rate_z

def pick_energy_peaks(times, composite, min_zscore, min_separation_seconds, max_candidates):
    """Greedy non-max suppression: take the highest-scoring hop, reject
    anything within min_separation_seconds of an already-picked peak, repeat.
    Stops one long scream from producing ten near-identical candidates."""
    import numpy as np

    order = np.argsort(composite)[::-1]
    chosen_indices = []
    chosen_times = []

    for idx in order:
        score = composite[idx]
        if score < min_zscore:
            break
        t = times[idx]
        if all(abs(t - ct) >= min_separation_seconds for ct in chosen_times):
            chosen_indices.append(int(idx))
            chosen_times.append(t)
        if len(chosen_indices) >= max_candidates:
            break

    chosen_indices.sort(key=lambda i: times[i])
    return chosen_indices

def build_part_time_ranges(transcript_blocks_by_part):
    """For each transcript_partN.txt, the (min, max) timestamp actually
    observed in its blocks - used to map an audio-scan timestamp (which has
    no part of its own) back to the right SourcePart for the verification
    pass's transcript-snippet lookup."""
    ranges = []
    for part, blocks in transcript_blocks_by_part.items():
        if not blocks:
            continue
        starts = [b[0] for b in blocks]
        ranges.append((part, min(starts), max(starts)))
    ranges.sort(key=lambda r: r[1])
    return ranges

def part_for_timestamp(seconds, part_ranges):
    """Which transcript part a given absolute timestamp falls into."""
    if not part_ranges:
        return ""
    for part, start, end in part_ranges:
        if start - 30 <= seconds <= end + 30:
            return part
    return min(part_ranges, key=lambda r: min(abs(seconds - r[1]), abs(seconds - r[2])))[0]

def flatten_transcript_blocks(transcript_blocks_by_part):
    """All parsed (start_seconds, text) blocks across every part, sorted -
    used so title generation can find nearby dialogue regardless of which
    side of a part boundary a peak happens to fall on."""
    all_blocks = []
    for blocks in transcript_blocks_by_part.values():
        all_blocks.extend(blocks)
    all_blocks.sort(key=lambda b: b[0])
    return all_blocks

def audio_candidate_base_score(loudness_z, rate_z):
    """Initial 1-10 score before emotion boost/judging. These candidates
    already cleared AUDIO_SCAN_MIN_ZSCORE (they're a statistical outlier for
    THIS stream), so they start in the middle of the range; only a very
    strong reading nudges them higher. Emotion boost and the judge stage
    still do the real work of separating the good ones from the rest."""
    base = 5
    peak_z = max(loudness_z, rate_z)
    if peak_z > 2.0:
        base += 1
    if peak_z > 3.0:
        base += 1
    return min(base, 7)

def describe_audio_signal(loudness_z, rate_z):
    """Qualitative tiers for the titling prompt - easier for the model to
    reason about than raw z-scores."""
    def tier(z):
        if z > 3.0:
            return "very high"
        if z > 2.0:
            return "high"
        return "elevated"
    return f"loudness: {tier(loudness_z)} for this stream, speech-rate proxy: {tier(rate_z)} for this stream"

def title_audio_candidates(raw_candidates, all_blocks, stream_folder):
    """Generates a Title/Reason for each audio-scan peak, batched through
    JUDGE_MODEL. When nearby transcript text exists it's given as grounding
    context (same as the discovery passes); when it doesn't, the model is
    explicitly told to describe the audio signal itself rather than invent
    dialogue, so the Reason stays honest for the later verification pass.

    raw_candidates: list of dicts with Timestamp, LoudnessZ, RateZ, Score,
    SourcePart. Returns highlight-dicts ready to merge into `highlights`.

    Writes debug_audioscan_titles_batch_<N>.txt for any batch that parses
    to zero titles, mirroring the debug dumps discovery and verify already
    write on a similar zero/low-yield result - the raw response is the
    fastest way to tell "nothing here was worth titling" apart from a
    broken/truncated model response.
    """
    titled = []

    header = """You are writing short clip titles for moments detected purely from audio
signal analysis (loudness and speech-rate), not from what the model thinks
happened. For each numbered item below you're given a description of the
audio signal, and - if one exists - the nearby transcript text.

If transcript text is provided, ground the Title/Reason in it, the same way
you would for a normal highlight clip.

If NO transcript text is provided (it will say "no transcript text nearby"),
do NOT invent dialogue or claim specific words were said. Instead, describe
the audio signal itself, e.g. a title like "Sudden loud reaction" and a
reason like "Sharp volume and speech-rate spike with no matching dialogue -
likely a non-verbal reaction such as a shout or scream."

STRICT OUTPUT FORMAT:
- Exactly one moment per line
- Exactly 3 comma-separated fields per line: ItemNumber,"Title","Reason"
- Wrap Title and Reason in double quotes since they may contain commas
- No extra commentary, headers, or explanation - CSV rows only

"""

    for batch_start in range(0, len(raw_candidates), AUDIO_SCAN_TITLE_BATCH_SIZE):
        batch = raw_candidates[batch_start:batch_start + AUDIO_SCAN_TITLE_BATCH_SIZE]

        prompt = header
        for i, c in enumerate(batch, start=1):
            snippet = get_snippet(all_blocks, timestamp_to_seconds(c["Timestamp"]), window=15)
            snippet_text = snippet if snippet else "no transcript text nearby"
            signal_desc = describe_audio_signal(c["LoudnessZ"], c["RateZ"])
            prompt += f"{i}. Signal: {signal_desc} | Transcript: {snippet_text}\n"

        # /api/chat + top-level think:false, not /api/generate + "/no_think"
        # in the prompt: qwen3.5 handles thinking via Ollama's own renderer/
        # parser, and /api/generate ignores think:false outright for it
        # (confirmed Ollama bug ollama/ollama#14793) - it just thinks until
        # num_predict runs out and "response" comes back empty (the "0/10
        # titles parsed" failure this used to produce for every batch).
        # /api/chat + think:false is the combination Ollama confirms works.
        payload = {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": NUM_CTX,
                "num_predict": AUDIO_SCAN_TITLE_NUM_PREDICT,
            },
        }

        try:
            response = ollama_generate(payload, timeout=900, url=OLLAMA_CHAT_URL)
            result = response.json().get("message", {}).get("content", "")
        except Exception as exc:
            print(f"     [!] Audio-candidate titling batch failed ({exc}); skipping this batch")
            continue

        parsed = {}
        for line in result.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("itemnumber"):
                continue
            try:
                reader = csv.reader(io.StringIO(line), skipinitialspace=True)
                pieces = next(reader)
            except Exception:
                continue
            if len(pieces) != 3:
                continue
            idx_text, title, reason = pieces
            match = re.search(r"\d+", idx_text)
            if not match:
                continue
            parsed[int(match.group())] = (title.strip(), reason.strip())

        added_this_batch = 0
        for i, c in enumerate(batch, start=1):
            if i not in parsed:
                continue
            title, reason = parsed[i]
            if is_low_content_title(title):
                continue
            titled.append({
                "Timestamp": c["Timestamp"],
                "Score": c["Score"],
                "Title": title,
                "Reason": reason,
                "SourcePart": c["SourcePart"],
                "Category": "Audio",
            })
            added_this_batch += 1

        if added_this_batch == 0:
            debug_path = os.path.join(stream_folder, f"debug_audioscan_titles_batch_{batch_start}.txt")
            try:
                with open(debug_path, "w", encoding="utf-8") as dbg:
                    dbg.write("PROMPT:\n")
                    dbg.write(prompt)
                    dbg.write("\n\n---RESPONSE---\n")
                    dbg.write(result)
                print(f"     [!] 0/{len(batch)} titles parsed for this batch - raw response saved to {debug_path}")
            except Exception as exc:
                print(f"     [!] 0/{len(batch)} titles parsed for this batch, and failed to save debug dump: {exc}")

    return titled

def find_audio_scan_candidates(stream_folder, highlights, transcript_blocks_by_part):
    """Orchestrates the full-file audio scan: cheap DSP sweep -> peak-pick ->
    skip anything near an existing candidate -> title the survivors. Returns
    a list of highlight-dicts ready to merge into the main `highlights` list.
    Fails soft (returns []) if the mic wav or optional dependencies aren't
    available, matching the existing emotion-scoring sidecar's behavior.
    """
    if not AUDIO_SCAN_ENABLED:
        return []

    mic_wav = find_mic_wav(stream_folder)
    if not mic_wav:
        print("[audio-scan] No *_mic.wav file found; skipping full-file audio scan.")
        return []

    try:
        import librosa  # numpy is a hard dependency of librosa, no need to check separately
    except Exception as exc:
        print("[audio-scan] Missing optional dependency (librosa); skipping full-file audio scan.")
        print("[audio-scan]", exc)
        return []

    print("=" * 60)
    print("Full-File Audio Scan (energy + speech-rate proxy)")
    print("=" * 60)
    print("Audio:", mic_wav)

    try:
        audio, sample_rate = librosa.load(mic_wav, sr=None, mono=True)
    except Exception as exc:
        print("[audio-scan] Failed to load audio; skipping full-file audio scan.")
        print("[audio-scan]", exc)
        return []

    times, composite, loudness_z, rate_z = compute_audio_arousal_series(audio, sample_rate)

    peak_indices = pick_energy_peaks(
        times, composite,
        min_zscore=AUDIO_SCAN_MIN_ZSCORE,
        min_separation_seconds=AUDIO_SCAN_MIN_SEPARATION_SECONDS,
        max_candidates=AUDIO_SCAN_MAX_CANDIDATES * 3,  # generous pre-filter, trimmed below
    )
    AUDIO_SCAN_STATS["candidates_found"] = len(peak_indices)
    print(f"[audio-scan] {len(peak_indices)} energy peak(s) found before de-duplication against existing candidates")

    existing_seconds = [timestamp_to_seconds(h["Timestamp"]) for h in highlights]
    part_ranges = build_part_time_ranges(transcript_blocks_by_part)
    all_blocks = flatten_transcript_blocks(transcript_blocks_by_part)

    raw_candidates = []
    for idx in peak_indices:
        t = float(times[idx])
        if any(abs(t - e) <= AUDIO_SCAN_SKIP_NEAR_EXISTING_SECONDS for e in existing_seconds):
            continue

        lz = float(loudness_z[idx])
        rz = float(rate_z[idx])
        raw_candidates.append({
            "Timestamp": format_plain_seconds(t),
            "LoudnessZ": lz,
            "RateZ": rz,
            "Score": audio_candidate_base_score(lz, rz),
            "SourcePart": part_for_timestamp(t, part_ranges),
        })
        if len(raw_candidates) >= AUDIO_SCAN_MAX_CANDIDATES:
            break

    AUDIO_SCAN_STATS["candidates_kept"] = len(raw_candidates)
    print(f"[audio-scan] {len(raw_candidates)} candidate(s) remain after skipping ones near existing highlights")

    if not raw_candidates:
        return []

    print(f"[audio-scan] Generating titles for {len(raw_candidates)} candidate(s)...")
    titled = title_audio_candidates(raw_candidates, all_blocks, stream_folder)
    print(f"[audio-scan] {len(titled)} candidate(s) added from audio scan")
    return titled

def is_low_content_title(title):
    """Hard filter for junk titles the model is supposed to exclude via
    prompting but sometimes scores highly anyway: single words, repeated
    words/phrases, pure sound effects, etc. Returns True if the title
    should be rejected outright regardless of its score.

    Deliberately does NOT reject a title just for being short - "He's
    cooked" and "No way" are exactly the kind of punchy, quotable hooks the
    judge prompt asks for. Only single-word titles and titles that collapse
    to one word/phrase repeated back-to-back get rejected.
    """
    cleaned = title.strip().strip('"').strip()

    if not cleaned:
        return True

    words = [
        w.strip(".,!?'\"").lower()
        for w in re.split(r'[\s,]+', cleaned)
        if w.strip(".,!?'\"")
    ]

    if not words:
        return True

    # Single word title (e.g. "beep", "Mac")
    if len(words) == 1:
        return True

    # Repeated single word/phrase, e.g. "bum, bum, bum, bum, bum"
    unique_words = set(words)
    if len(unique_words) == 1 and len(words) >= 2:
        return True

    # Repeated multi-word phrase separated by commas, e.g.
    # "I'd be low, I'd be low"
    comma_segments = [s.strip().lower() for s in cleaned.split(",") if s.strip()]
    if len(comma_segments) >= 2 and len(set(comma_segments)) == 1:
        return True

    # Repeated multi-word phrase with no commas, e.g. "no way no way"
    for phrase_len in (2, 3):
        if len(words) >= phrase_len * 2:
            chunks = [
                tuple(words[i:i + phrase_len])
                for i in range(0, len(words) - phrase_len + 1, phrase_len)
            ]
            if len(chunks) >= 2 and len(set(chunks)) == 1:
                return True

    # Catches a title that's a stack of repeated interjections, e.g. "dude
    # dude dude what what" (two words, each repeated back-to-back) - unlike
    # the phrase-chunk check above (one phrase repeated uniformly), this
    # allows multiple different runs, as long as every word is in a run >= 2.
    runs = []
    for w in words:
        if runs and runs[-1][0] == w:
            runs[-1][1] += 1
        else:
            runs.append([w, 1])
    if runs and all(count >= 2 for _, count in runs):
        return True

    return False

# --- Stage 5: Discovery --------------------------------------------------

def run_discovery(stream_folder, parts, prompts):
    """Runs the LLM discovery passes across every transcript_partN.txt,
    then dedupes and drops low-content titles. Returns the raw candidate
    list (not yet audio-scanned, emotion-scored, verified, or judged).

    prompts: list of (prompt_text, pass_name) tuples - see PROMPTS below.

    Returns (highlights, part_errors) - part_errors counts how many
    transcript-part iterations raised an exception and were skipped
    entirely (e.g. Ollama unreachable for that call). run_stage_discovery()
    uses this to tell "this stream genuinely has nothing" apart from "this
    run failed before it got a real look" - an empty result that coincides
    with errors shouldn't be checkpointed as a done, trustworthy stage.
    """
    highlights = []
    part_errors = 0

    for idx, part in enumerate(parts, start=1):
        path = os.path.join(stream_folder, part)

        if not os.path.exists(path):
            print(f"[{idx}/{len(parts)}] Missing: {part}")
            continue

        print(f"[{idx}/{len(parts)}] Analyzing {part}")

        with open(path, "r", encoding="utf-8") as f:
            transcript = f.read()

        real_timestamps = extract_transcript_timestamps(transcript)

        if not real_timestamps:
            print("     [!] No timestamps found in transcript - hallucination check disabled for this part")

        try:
            for prompt_text, pass_name in prompts:

                print(f"  -> {pass_name} pass")

                payload = {
                    "model": MODEL,
                    "prompt": prompt_text + "\n\n" + transcript,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_ctx": NUM_CTX
                    }
                }

                start_time = time.time()
                response = ollama_generate(payload, timeout=1800)
                elapsed = time.time() - start_time
                print(f"     Response received in {elapsed:.1f} seconds")

                result = response.json().get("response", "")

                added = 0
                rejected_hallucinated = 0
                rejected_malformed = 0

                for line in result.splitlines():
                    line = line.strip()

                    if not line:
                        continue

                    if line.lower().startswith("timestamp"):
                        continue

                    # Use a CSV reader so quoted fields containing commas
                    # (e.g. "Uhhh, so sad") are parsed correctly.
                    try:
                        reader = csv.reader(io.StringIO(line), skipinitialspace=True)
                        pieces = next(reader)
                    except Exception:
                        pieces = line.split(",", 3)

                    # Fallback: model sometimes puts a space instead of a
                    # comma between timestamp and score, e.g. "[03:17:37] 9,
                    # ..." (merges the first two fields into one - 3 fields
                    # instead of 4). Detect and split that case.
                    if len(pieces) == 3 and re.match(
                        r'^\s*[\[]?\d{2}:\d{2}:\d{2}[\]]?\s+\d+\s*$', pieces[0]
                    ):
                        fixed = re.match(
                            r'^\s*([\[]?\d{2}:\d{2}:\d{2}[\]]?)\s+(\d+)\s*$',
                            pieces[0]
                        )
                        if fixed:
                            pieces = [fixed.group(1), fixed.group(2)] + pieces[1:]

                    if len(pieces) != 4:
                        rejected_malformed += 1
                        continue

                    timestamp, score, title, reason = pieces

                    timestamp = timestamp.strip()
                    match = re.search(r'(\d{2}:\d{2}:\d{2})', timestamp)

                    if not match:
                        rejected_malformed += 1
                        continue

                    timestamp = match.group(1)

                    try:
                        score = int(re.sub(r'[^\d-]', '', str(score)))
                    except Exception:
                        rejected_malformed += 1
                        continue

                    if real_timestamps:
                        candidate_seconds = timestamp_to_seconds(timestamp)
                        distance = nearest_distance(candidate_seconds, real_timestamps)

                        if distance is not None and distance > TIMESTAMP_TOLERANCE_SECONDS:
                            rejected_hallucinated += 1
                            continue

                    if is_low_content_title(title):
                        continue

                    highlights.append({
                        "Timestamp": timestamp.strip(),
                        "Score": score,
                        "Title": title.strip(),
                        "Reason": reason.strip(),
                        "SourcePart": part,
                        "Category": pass_name
                    })

                    added += 1

                print(f"Added {added} candidates")

                if rejected_hallucinated > 0:
                    print(f"     [!] Rejected {rejected_hallucinated} candidates with timestamps not found in transcript")

                if rejected_malformed > 0:
                    print(f"     [!] Skipped {rejected_malformed} malformed lines")

                if added == 0:
                    debug_path = os.path.join(stream_folder, f"debug_{part}_{pass_name}.txt")
                    with open(debug_path, "w", encoding="utf-8") as dbg:
                        dbg.write(result)
                    print(f"     [!] 0 candidates parsed - raw response saved to {debug_path}")

                print()

        except Exception as e:
            print(f"ERROR processing {part}")
            print(e)
            print()
            part_errors += 1

    # Deduplicate by timestamp + title
    unique = {}
    for h in highlights:
        key = (h["Timestamp"].strip(), h["Title"].strip().lower())
        if key not in unique:
            unique[key] = h
            continue
        if h["Score"] > unique[key]["Score"]:
            unique[key] = h
    highlights = list(unique.values())

    # Hard filter: drop junk titles regardless of model score - enforced in
    # code since smaller models don't reliably apply exclusion rules from
    # the prompt alone. (Discovery already skips these per-line; this
    # catches what slipped through after dedup picked a still-junk winner.)
    before_filter_count = len(highlights)
    highlights = [h for h in highlights if not is_low_content_title(h["Title"])]
    filtered_count = before_filter_count - len(highlights)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} low-content candidates (single/repeated words, sound effects)")

    return highlights, part_errors

# --- Stage 8: Content verification ---------------------------------------

def verify_candidates(highlights, transcript_blocks_by_part, verify_prompt_header, stream_folder, batch_size=None):
    """A timestamp can be "real" (it exists in the transcript) while the
    title/reason describing it is still made up. This pulls the actual
    transcript text around each candidate's timestamp and asks the model to
    confirm the claim is actually supported by what was said/happened
    there. Returns the surviving (verified) highlights list.

    Returns (verified_highlights, total_items, total_verdicts_parsed). The
    last two let the caller judge whether the WHOLE stage's output is
    trustworthy: a single slow/truncated batch is normal and already
    handled per-batch below (debug dump + kept-unverified fallback), but if
    parsing fails across every batch, that same "keep unparsed items rather
    than dropping them" fallback quietly turns verification into a no-op
    for the entire run - see VERIFY_MIN_COVERAGE_RATIO / run_stage_verify().
    """
    if batch_size is None:
        batch_size = VERIFY_BATCH_SIZE

    print(f"Verifying {len(highlights)} candidates against transcript content...")

    verified_highlights = []
    total_items = 0
    total_verdicts_parsed = 0

    for batch_start in range(0, len(highlights), batch_size):
        batch = highlights[batch_start:batch_start + batch_size]
        total_items += len(batch)

        # Thinking must be OFF here: JUDGE_MODEL is thinking-capable, and
        # without this it can burn the whole num_predict budget on an
        # internal reasoning trace before emitting ItemNumber,VERDICT lines
        # - every batch silently returning 0 parsed verdicts. qwen3.5
        # handles thinking via Ollama's own renderer/parser, not the old
        # "/no_think"-in-prompt convention, and /api/generate ignores
        # think:false for it entirely (confirmed Ollama bug
        # ollama/ollama#14793) - only /api/chat + top-level think:false
        # actually disables it, so that's what's used below.
        verify_prompt = verify_prompt_header
        for i, h in enumerate(batch, start=1):
            blocks = transcript_blocks_by_part.get(h.get("SourcePart"), [])
            target_seconds = timestamp_to_seconds(h["Timestamp"])
            snippet = get_snippet(blocks, target_seconds, window=20)

            if not snippet:
                snippet = "(no transcript text found near this timestamp)"

            verify_prompt += (
                f"{i}. Title: {h['Title']} | Reason: {h['Reason']} | "
                f"Snippet: {snippet}\n"
            )

        payload = {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": verify_prompt}],
            "think": False,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": NUM_CTX,
                "num_predict": VERIFY_NUM_PREDICT
            }
        }

        try:
            response = ollama_generate(payload, timeout=1800, url=OLLAMA_CHAT_URL)
            result = response.json().get("message", {}).get("content", "")

            verdicts = {}
            for line in result.splitlines():
                line = line.strip()
                if not line:
                    continue
                m = re.match(r'^\D*(\d+)\D*?(PASS|FAIL)', line, re.IGNORECASE)
                if not m:
                    continue
                verdicts[int(m.group(1))] = m.group(2).upper()

            total_verdicts_parsed += len(verdicts)

            # Sanity check: if the model returned verdicts for far fewer
            # items than were sent, it likely ran out of output budget
            # partway through - surface this instead of hiding it.
            coverage = len(verdicts) / len(batch) if batch else 1
            if coverage < 0.8:
                print(f"     [!] Verification only returned {len(verdicts)}/{len(batch)} verdicts for this batch - response may have been cut off")
                debug_path = os.path.join(stream_folder, f"debug_verify_batch_{batch_start}.txt")
                with open(debug_path, "w", encoding="utf-8") as dbg:
                    dbg.write("PROMPT:\n")
                    dbg.write(verify_prompt)
                    dbg.write("\n\n---RESPONSE---\n")
                    dbg.write(result)

            rejected_count = 0
            for i, h in enumerate(batch, start=1):
                verdict = verdicts.get(i)
                # If the model didn't return a verdict for this item, keep
                # it rather than silently dropping it (avoid being
                # over-aggressive due to a parsing miss).
                if verdict == "FAIL":
                    rejected_count += 1
                    continue
                verified_highlights.append(h)

            if rejected_count > 0:
                print(f"     [!] Verification rejected {rejected_count} candidates as unsupported by transcript")

        except Exception as e:
            print("     [!] Verification batch failed, keeping candidates unverified")
            print("    ", e)
            verified_highlights.extend(batch)

    return verified_highlights, total_items, total_verdicts_parsed

# --- Stage 9: Judging -----------------------------------------------------

def run_judge_batch(pool, keep_n, judge_instructions):
    """Sends a batch of candidates to the judge model and returns the
    ranked subset (as highlight dicts), preserving model-assigned order.
    judge_instructions must contain a {keep_n} placeholder.

    Uses /api/chat + think:false, like verify and audio-scan titling. The
    judge originally ran on /api/generate with thinking enabled, but newer
    Ollama (>= 0.3.x) counts the reasoning against num_predict and returns
    it in a separate `thinking` JSON field instead of inline in `response` -
    the model burns the whole budget mid-thought and the CSV rows never
    appear, silently reducing judging to a score sort.
    """
    prompt = judge_instructions.format(keep_n=keep_n)

    for i, h in enumerate(pool, start=1):
        prompt += (
            f"{i}. {h['Timestamp']} | "
            f"{h['Title']} | "
            f"{h['Reason']} | "
            f"DiscoveryScore={h['Score']} | "
            f"TranscriptScore={h.get('TranscriptScore', h['Score'])} | "
            f"Emotion={h.get('Emotion', '')} {h.get('EmotionConfidence', '')} | "
            f"EmotionBoost={h.get('EmotionBoost', 0)}\n"
        )

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": NUM_CTX,
            "num_predict": 3000
        }
    }

    response = ollama_generate(payload, timeout=900, url=OLLAMA_CHAT_URL)
    result = response.json().get("message", {}).get("content", "")

    ranked = []
    seen_timestamps = set()

    for line in result.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.lower().startswith("rank"):
            continue

        pieces = line.split(",", 3)

        if len(pieces) != 4:
            continue

        rank, score, timestamp, title = pieces
        timestamp = timestamp.strip().strip('"')
        # The model sometimes quotes fields ("00:03:28") and/or emits decimal
        # scores (7.5). Strip quotes; round decimals rather than mangling
        # them to "75". An unparseable score keeps the row but leaves
        # JudgeScore unset (backfill-by-score then applies).
        try:
            score_text = re.sub(r"[^\d.]", "", str(score))
            if score_text:
                judge_score = min(max(round(float(score_text)), 1), 10)
            else:
                judge_score = None
        except Exception:
            judge_score = None

        if timestamp in seen_timestamps:
            continue

        for h in pool:
            if h["Timestamp"] == timestamp:
                ranked_highlight = dict(h)
                if judge_score is not None:
                    ranked_highlight["JudgeScore"] = judge_score
                    ranked_highlight["Score"] = judge_score
                ranked.append(ranked_highlight)
                seen_timestamps.add(timestamp)
                break

    if not ranked:
        # Total parse failure: surface it instead of silently degrading to a
        # score sort. This used to happen on every run with newer Ollama,
        # where /api/generate with thinking enabled returned an empty
        # response (the whole output budget went into a separate `thinking`
        # field) - see the JUDGE_INSTRUCTIONS comment above.
        print(f"     [!] Judge returned no parseable CSV rows for {len(pool)} candidate(s) - falling back to score order")
        print(f"     [!] Raw response: {result[:400]!r}")

    return ranked

def run_judge_tournament(judge_pool, judge_instructions, top_n, judge_batch_size=None):
    """Ranks judge_pool down to top_n. For pools bigger than one batch,
    judges in batches first (keeping the best half of each), then does a
    final ranking pass on the survivors. Falls back to a plain score-sort
    if the judge stage throws. Returns the final ranked highlights list
    (highlights not selected by the judge get backfilled by score if the
    judge returned fewer than top_n)."""
    if judge_batch_size is None:
        judge_batch_size = JUDGE_BATCH_SIZE

    highlights_by_score = sorted(judge_pool, key=lambda x: x["Score"], reverse=True)
    ranked = []

    try:
        if len(judge_pool) <= judge_batch_size:
            ranked = run_judge_batch(judge_pool, top_n, judge_instructions)
        else:
            round1_survivors = []
            num_batches = (len(judge_pool) + judge_batch_size - 1) // judge_batch_size
            print(f"     Judging in {num_batches} batch(es) of up to {judge_batch_size}...")

            for batch_start in range(0, len(judge_pool), judge_batch_size):
                batch = judge_pool[batch_start:batch_start + judge_batch_size]
                keep_n = max(1, len(batch) // 2)

                batch_ranked = run_judge_batch(batch, keep_n, judge_instructions)

                if not batch_ranked:
                    # If a batch fails to parse, fall back to its
                    # top-scored candidates rather than losing the batch.
                    batch_ranked = sorted(batch, key=lambda x: x["Score"], reverse=True)[:keep_n]

                round1_survivors.extend(batch_ranked)

            print(f"     Round 1 complete, {len(round1_survivors)} candidates advancing to final round")
            ranked = run_judge_batch(round1_survivors, top_n, judge_instructions)

    except Exception as e:
        print("Judge stage failed, falling back to score sort")
        print(e)
        return highlights_by_score[:top_n]

    if len(ranked) >= top_n:
        return ranked[:top_n]
    elif ranked:
        seen = set((h["Timestamp"], h["Title"].lower()) for h in ranked)
        backfill = [h for h in highlights_by_score if (h["Timestamp"], h["Title"].lower()) not in seen]
        return (ranked + backfill)[:top_n]
    else:
        return highlights_by_score[:top_n]

# --- Stage 10: Export -----------------------------------------------------

def final_score_cap_for_rank(rank: int) -> int:
    if rank == 1:
        return 10
    if rank <= 3:
        return 9
    if rank <= 10:
        return 8
    if rank <= 20:
        return 7
    if rank <= 35:
        return 6
    return 5

def calibrate_final_scores_by_rank(highlights):
    """Prevent score inflation: final scores are rank-calibrated for output."""
    for rank, highlight in enumerate(highlights, start=1):
        raw_score = int(highlight.get("Score", 0))
        cap = final_score_cap_for_rank(rank)
        highlight["RawScore"] = raw_score
        highlight["ScoreCap"] = cap
        highlight["Score"] = min(raw_score, cap)

def score_to_resolve_color(score):
    """Maps a highlight's score to a DaVinci Resolve marker color name.
    Resolve's 'Timeline Markers from EDL' importer reads these from a
    |C:ResolveColor<Name> comment line, not the Avid-style '* LOC:' syntax."""
    if score >= 9:
        return "Green"
    elif score == 8:
        return "Yellow"
    else:
        return "Blue"

def sanitize_marker_text(text):
    """Resolve's marker EDL parser uses '|' as a field separator and
    ignores note text starting with a digit, so strip pipes and guard
    against a leading numeral."""
    cleaned = text.replace("|", "-").strip()
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned

def write_highlights_csv(stream_folder, final_highlights, top_n):
    csv_path = os.path.join(stream_folder, f"top{top_n}_highlights.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank", "Score", "RawScore", "ScoreCap", "JudgeScore",
            "TranscriptScore", "Emotion", "EmotionConfidence", "EmotionBoost",
            "Timestamp", "Title", "Reason", "Category"
        ])
        for rank, h in enumerate(final_highlights, start=1):
            writer.writerow([
                rank,
                h["Score"],
                h.get("RawScore", h["Score"]),
                h.get("ScoreCap", ""),
                h.get("JudgeScore", ""),
                h.get("TranscriptScore", h["Score"]),
                h.get("Emotion", ""),
                h.get("EmotionConfidence", ""),
                h.get("EmotionBoost", 0),
                h["Timestamp"],
                h["Title"],
                h["Reason"],
                h.get("Category", "")
            ])
    return csv_path

def write_highlights_edl(stream_folder, final_highlights, top_n):
    edl_path = os.path.join(stream_folder, f"top{top_n}_markers.edl")

    with open(edl_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"TITLE: TOP{top_n}_HIGHLIGHTS\n")
        f.write("FCM: NON-DROP FRAME\n\n")

        for rank, h in enumerate(final_highlights, start=1):
            tc = h["Timestamp"]
            if tc.count(":") == 2:
                tc = tc + ":00"

            hh, mm, ss, ff = map(int, tc.split(":"))
            end_ff = ff + 1
            if end_ff >= 60:
                end_ff = 0
                ss += 1
            end_tc = f"{hh:02d}:{mm:02d}:{ss:02d}:{end_ff:02d}"

            title = h["Title"].replace("\n", " ")
            color = score_to_resolve_color(h["Score"])
            category = h.get("Category", "")
            category_tag = f"[{category}] " if category else ""
            marker_text = sanitize_marker_text(f"{category_tag}{title} (Score {h['Score']})")

            f.write(
                f"{rank:03d}  AX       V     C        "
                f"{tc} {end_tc} {tc} {end_tc}\n"
            )
            f.write(f" |C:ResolveColor{color} |M:{marker_text} |D:1\n\n")

    return edl_path

def write_run_info(stream_folder, final_highlights):
    """Snapshots the models/settings used and the accumulated pipeline
    stats (Ollama calls, per-stage timing) into run_info.json."""
    stats = load_pipeline_stats(stream_folder)
    run_info = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "stream_folder": stream_folder,
        "model": MODEL,
        "judge_model": JUDGE_MODEL,
        "top_n": TOP_N,
        "judge_pool_size": JUDGE_POOL_SIZE,
        "verify_pool_size": VERIFY_POOL_SIZE,
        "timestamp_tolerance_seconds": TIMESTAMP_TOLERANCE_SECONDS,
        "emotion_enabled": EMOTION_ENABLED,
        "emotion_boosts": EMOTION_BOOSTS,
        "audio_scan_enabled": AUDIO_SCAN_ENABLED,
        "audio_scan_candidates_found": stats.get("audio_scan_candidates_found", 0),
        "audio_scan_candidates_kept": stats.get("audio_scan_candidates_kept", 0),
        "final_highlight_count": len(final_highlights),
        "ollama_calls": stats.get("ollama_calls", 0),
        "ollama_retries": stats.get("ollama_retries", 0),
        "ollama_seconds": round(stats.get("ollama_seconds", 0.0), 1),
        "stage_seconds": stats.get("stage_seconds", {}),
        "total_pipeline_seconds": round(sum(stats.get("stage_seconds", {}).values()), 1),
    }
    run_info_path = os.path.join(stream_folder, RUN_INFO_FILENAME)
    try:
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)
        print(f"Run metadata saved: {run_info_path}")
    except Exception as exc:
        print(f"[!] Could not write run metadata: {exc}")
    return run_info

def record_pipeline_run_history(stream_folder):
    """Appends one row to pipeline_run_history.csv, kept next to this
    script (SCRIPT_DIR) rather than in stream_folder, so it accumulates
    timing across every VOD you process instead of just the current one.

    Only called from run_all_remaining_stages() after a normal (no
    --stage) run reaches the end of the stage list - a forced single-stage
    debug rerun (5a-5f) isn't representative of a full pipeline's timing,
    so it deliberately doesn't get logged here.

    stage_seconds (from pipeline_stats.json) holds, for each stage, the
    duration of the last time that stage actually ran - which for a
    resumed run may span several separate sessions on different days
    rather than one unbroken sitting. That's fine for what this is used
    for (rough per-step and total averages over time), just not a literal
    stopwatch time for any single session.
    """
    stats = load_pipeline_stats(stream_folder)
    stage_seconds = stats.get("stage_seconds", {})

    row = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "stream_folder": stream_folder,
    }
    for stage in STAGE_ORDER:
        row[f"{stage}_seconds"] = stage_seconds.get(stage, "")
    row["total_seconds"] = round(sum(stage_seconds.values()), 1)
    row["ollama_calls"] = stats.get("ollama_calls", 0)
    row["ollama_seconds"] = round(stats.get("ollama_seconds", 0.0), 1)
    row["ollama_retries"] = stats.get("ollama_retries", 0)

    final_highlight_count = ""
    run_info_path = os.path.join(stream_folder, RUN_INFO_FILENAME)
    if os.path.exists(run_info_path):
        try:
            with open(run_info_path, "r", encoding="utf-8") as f:
                final_highlight_count = json.load(f).get("final_highlight_count", "")
        except Exception:
            pass
    row["final_highlight_count"] = final_highlight_count

    history_path = SCRIPT_DIR / RUN_HISTORY_FILENAME
    is_new_file = not history_path.exists()
    try:
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f"Logged run timing to: {history_path}")
    except Exception as exc:
        print(f"[!] Could not write pipeline run history: {exc}")

# ============================================================================
# Discovery prompts (Emotion / Gameplay / Viral passes)
# ============================================================================
PROMPTS = [
("""/no_think
You are an expert short-form content scout for TikTok/Reels/YouTube Shorts.

CRITICAL RULE: Only return moments that are explicitly present in the
transcript text below. Every Timestamp you return MUST be copied directly
from a timestamp that appears in the transcript - never estimate, round,
or invent a timestamp. Every Title and Reason must be based on dialogue or
events that are actually written in the transcript, not assumed or imagined.
If you are not sure a moment exists at a specific timestamp, do not include it.

Find emotional moments from this Twitch transcript - reactions raw or
intense enough that someone scrolling a feed would stop and watch.

Prioritize:
- rage
- frustration
- screaming
- excitement
- laughter

ZERO-CONTEXT TEST (apply this to every candidate):
Imagine a stranger with no idea who this streamer is or what game this is.
Would the reaction itself (tone, volume, words) be entertaining within 3
seconds, even without knowing why it happened? If the reaction only makes
sense with backstory (e.g. "finally won after trying for 3 hours"), still
include it but score it lower, since it works better with a caption than
on its own.

Prefer moments where the emotion is captured in clear dialogue or audible
reaction, not subtle mood shifts that wouldn't be obvious in a short clip.

Score each moment 1-10 using this rubric:
- 9-10: Reaction alone is entertaining with zero context, instantly clip-able
- 6-8: Strong reaction, but lands better with a one-line caption for context
- 3-5: Genuine reaction, but needs backstory or stream knowledge to land
- 1-2: Only meaningful if you were already watching live

Use the FULL 1-10 range, but be harsh. Most valid moments should land in
4-7. An 8 is already a strong clip. A 9 is rare. A 10 means a near-perfect,
standalone viral moment that would be obvious to strangers with no context.
Across an entire stream, expect zero or one 10. If many moments feel like 9
or 10, lower them until only the truly exceptional outliers remain.

In the Reason field, describe SPECIFICALLY what happens at this exact
moment, in your own words, based on the actual transcript text. Do not
copy or rephrase the category descriptions or criteria from these
instructions (e.g. do not write "Setup -> twist -> payoff" or "did that
just happen moment" as the reason) - that is not a real reason, it is just
repeating the instructions. Write what is ACTUALLY said or happening.

AUTOMATIC LOW SCORE (1-2) or EXCLUDE:
- Singing, humming, or musical moments
- A single word or short phrase repeated multiple times with no other content
  (e.g. "no no no no", "what what what", screaming the same word repeatedly)
- Reactions with no actual identifiable trigger or moment behind them

Return up to 25 moments - fewer, well-chosen moments are better than
padding the list with weak ones just to reach the limit.

STRICT OUTPUT FORMAT:
- Exactly one moment per line
- Exactly 4 comma-separated fields per line: Timestamp,Score,Title,Reason
- Timestamp must be plain HH:MM:SS with NO brackets, NO extra text
- There must be a comma (not a space) immediately after the timestamp,
  and a comma (not a space) immediately after the score
- Wrap Title and Reason in double quotes since they may contain commas
- Do not add any extra commentary, headers, or explanation - CSV rows only

Example of a moment worth including (score shown is just an example, not a
target - use whatever score actually fits this specific moment):
03:17:37,6,"Uhhh, so sad","Expresses deep disappointment after a loss"

Example of a moment that should NOT appear in your output at all: a
teammate calmly says "yeah I guess that's fine" and the streamer just moves
on. No emotional intensity, no reaction a stranger would stop scrolling for
- leave moments like this out entirely rather than including them at a low
score.

Format:
Timestamp,Score,Title,Reason
""", "Emotion"),

("""/no_think
You are an expert short-form content scout for TikTok/Reels/YouTube Shorts.

CRITICAL RULE: Only return moments that are explicitly present in the
transcript text below. Every Timestamp you return MUST be copied directly
from a timestamp that appears in the transcript - never estimate, round,
or invent a timestamp. Every Title and Reason must be based on dialogue or
events that are actually written in the transcript, not assumed or imagined.
If you are not sure a moment exists at a specific timestamp, do not include it.

Find gameplay moments from this Twitch transcript that would impress or
entertain someone watching as a short clip.

Prioritize:
- clutches
- outplays
- insane aim
- huge mistakes
- game-winning plays

ZERO-CONTEXT TEST (apply this to every candidate):
Imagine a stranger with no familiarity with this specific game or its rules.
Would the moment still look impressive or funny just from what's visibly
happening (a clear win, a clear fail, a clear close call), even without
understanding the deeper mechanics? If it only impresses people who already
know the game's strategy, still include it but score it lower.

Prefer moments with an obvious, visible outcome (a kill, a win, a death, a
clear mistake) over plays that are only impressive to people who understand
matchup-specific or mechanic-specific nuance.

Score each moment 1-10 using this rubric:
- 9-10: Visually obvious and impressive/funny with zero game knowledge
- 6-8: Strong play, but lands better with a one-line caption explaining it
- 3-5: Impressive only to people who understand this game's mechanics
- 1-2: Only meaningful to viewers who were already watching live

Use the FULL 1-10 range, but be harsh. Most valid moments should land in
4-7. An 8 is already a strong clip. A 9 is rare. A 10 means a near-perfect,
standalone viral moment that would be obvious to strangers with no context.
Across an entire stream, expect zero or one 10. If many moments feel like 9
or 10, lower them until only the truly exceptional outliers remain.

In the Reason field, describe SPECIFICALLY what happens at this exact
moment, in your own words, based on the actual transcript text. Do not
copy or rephrase the category descriptions or criteria from these
instructions - that is not a real reason, it is just repeating the
instructions. Write what is ACTUALLY said or happening.

AUTOMATIC LOW SCORE (1-2) or EXCLUDE:
- Singing, humming, or musical moments
- A single word or short phrase repeated multiple times with no other content
- Moments with no clear, identifiable gameplay action behind them

Return up to 25 moments - fewer, well-chosen moments are better than
padding the list with weak ones just to reach the limit.

STRICT OUTPUT FORMAT:
- Exactly one moment per line
- Exactly 4 comma-separated fields per line: Timestamp,Score,Title,Reason
- Timestamp must be plain HH:MM:SS with NO brackets, NO extra text
- There must be a comma (not a space) immediately after the timestamp,
  and a comma (not a space) immediately after the score
- Wrap Title and Reason in double quotes since they may contain commas
- Do not add any extra commentary, headers, or explanation - CSV rows only

Example of a moment worth including (score shown is just an example, not a
target - use whatever score actually fits this specific moment):
03:17:37,6,"Clean 1v3 clutch","Wins the round alone after teammates die early"

Example of a moment that should NOT appear in your output at all: the
streamer picks up a routine kill in a lopsided fight with no real risk or
skill on display. Ordinary gameplay a stranger would scroll right past -
leave moments like this out entirely rather than including them at a low
score.

Format:
Timestamp,Score,Title,Reason
""", "Gameplay"),

("""/no_think
You are an expert short-form content scout for TikTok/Reels/YouTube Shorts.

CRITICAL RULE: Only return moments that are explicitly present in the
transcript text below. Every Timestamp you return MUST be copied directly
from a timestamp that appears in the transcript - never estimate, round,
or invent a timestamp. Every Title and Reason must be based on dialogue or
events that are actually written in the transcript, not assumed or imagined.
If you are not sure a moment exists at a specific timestamp, do not include it.

Your job is to find moments from this Twitch transcript that could go VIRAL
as a standalone clip, watched by someone who has never seen this streamer
or this game before.

A moment has viral potential if it has ONE OR MORE of these properties:
- A complete mini story in under 30 seconds (setup -> twist -> payoff)
- A reaction so extreme, unexpected, or chaotic it's funny with zero context
- Dramatic irony (the viewer can guess something bad/funny is about to happen)
- A quote that is funny or quotable as a standalone sentence, out of context
- A "did that just happen" moment - something rare, lucky, or absurd
- Pure chaos: something breaks, glitches, fails, or backfires in a comedic way

ZERO-CONTEXT TEST (apply this to every candidate):
Imagine a total stranger scrolling TikTok with no idea who this streamer is
or what game this is. Would they react (laugh, gasp, rewatch) within the
first 3 seconds? If understanding the moment requires backstory, an inside
joke, or game-specific knowledge, DO NOT include it.

Also prefer moments where the funny/shocking part is captured in dialogue or
clearly visible action, NOT moments that rely purely on tone of voice or
things happening off-screen, since clip viewers can't pick up on subtle audio cues.

Score each moment 1-10 using this rubric:
- 9-10: Shareable with zero context, instantly funny/shocking, perfect hook
- 6-8: Strong, but works best with a one-line caption for context
- 3-5: Funny to channel regulars, but needs game/streamer knowledge
- 1-2: Only really funny if you were already watching live

Use the FULL 1-10 range, but be harsh. Most valid moments should land in
4-7. An 8 is already a strong clip. A 9 is rare. A 10 means a near-perfect,
standalone viral moment that would be obvious to strangers with no context.
Across an entire stream, expect zero or one 10. If many moments feel like 9
or 10, lower them until only the truly exceptional outliers remain.

AUTOMATIC LOW SCORE (1-2) or EXCLUDE:
- Singing, humming, or musical moments
- A single word or short phrase repeated multiple times with no other content
- Moments with no clear story, trigger, or punchline behind them

Return up to 25 moments - fewer, well-chosen moments are better than
padding the list with weak ones just to reach the limit.

In the Reason field, write a SPECIFIC one-line hook/caption based on what
actually happens at this exact moment (e.g. "He didn't even see it coming"
only makes sense if someone genuinely got caught off guard - replace it
with your own hook describing the real event), followed by a short
explanation of why THIS SPECIFIC moment has viral potential. Do not reuse
the category names or criteria wording above as the reason - describe the
actual transcript content in your own words.

STRICT OUTPUT FORMAT:
- Exactly one moment per line
- Exactly 4 comma-separated fields per line: Timestamp,Score,Title,Reason
- Timestamp must be plain HH:MM:SS with NO brackets, NO extra text
- There must be a comma (not a space) immediately after the timestamp,
  and a comma (not a space) immediately after the score
- Wrap Title and Reason in double quotes since they may contain commas
- Do not add any extra commentary, headers, or explanation - CSV rows only

Example of a moment worth including (score shown is just an example, not a
target - use whatever score actually fits this specific moment):
03:17:37,6,"He had ONE job","Hook: he didn't even see it coming. Misses an open shot everyone expected him to make."

Example of a moment that should NOT appear in your output at all: the
streamer mentions offhand that they're a little tired and want a snack
soon. No story, no twist, no reaction - nothing here a stranger would ever
share. Leave moments like this out entirely rather than including them at
a low score.

Format:
Timestamp,Score,Title,Reason
""", "Viral")
]

# ============================================================================
# Verification prompt
# ============================================================================
VERIFY_PROMPT_HEADER = """You are fact-checking highlight clips against a transcript.

For each numbered item below, you are given:
- The claimed Title and Reason for a highlight
- The actual transcript text from that exact moment (the Snippet)

Decide if the Snippet plausibly supports the claimed Title/Reason. The
streamer's actual words/actions don't need to match word-for-word, but the
core claim (what supposedly happened) must be reasonably consistent with
the Snippet. If the Snippet is unrelated, contradicts the claim, or shows
nothing resembling the claimed event, mark it as a FAIL.

Example PASS: Title "Rages after dying", Reason "Screams in frustration
after an unfair death", Snippet "no way, NO WAY, that is such bullshit, I
had him dead to rights" - the core claim (frustration after dying) matches
even though the exact wording differs.

Example FAIL: Title "Clutches the round alone", Reason "Wins a 1v3 with no
teammates left", Snippet "yeah nice, gg, that was a good round" - the
Snippet shows mild acknowledgment of a round ending, not a clutch or a
1v3, and doesn't mention anything the Title/Reason specifically claims.

Respond with ONLY one line per item in this exact format:
ItemNumber,VERDICT

Where VERDICT is either PASS or FAIL. Do not add any other text.

"""

# ============================================================================
# Judge prompt
# ============================================================================
# Originally this was the only JUDGE_MODEL prompt that deliberately kept
# thinking ENABLED - judge is comparative (weighing candidates against a
# 5-factor priority order), which reasoning was supposed to help - and it
# ran on /api/generate, skipping the OLLAMA_CHAT_URL + think:false treatment
# used by verify/titling.
#
# That no longer works on current Ollama (>= 0.3.x): qwen3.5's reasoning is
# emitted into a separate `thinking` JSON field that counts against
# num_predict, so the model spends the whole 3000-token budget mid-thought
# and "response" comes back empty - the CSV rows never appear and judging
# silently degrades to a score sort. run_judge_batch() therefore uses the
# same /api/chat + think:false path as verify/titling (the qwen3.5
# think:false bug ollama/ollama#14793 only affects /api/generate).
#
# If judging comes up short on parsed items, run_judge_tournament() backfills
# by score rather than losing candidates - verify has no equivalent
# fallback, which is why it can't afford the same tradeoff.
JUDGE_INSTRUCTIONS = """You are ranking these candidates as if selecting clips for a TikTok/Reels/
Shorts account with no prior audience and no subscribers.

The following are candidate highlights already discovered by other passes.

Select the BEST moments using this priority order:
1. Works as a STANDALONE clip - no prior context, no familiarity with the
   streamer or game required to understand or enjoy it
2. Has a clear hook in the first 2-3 seconds (a viewer scrolling past would
   stop and watch)
3. Emotional impact (shock, laughter, excitement, secondhand embarrassment)
4. Memorable, quotable, or visually distinct moment
5. General entertainment/clip value

Penalize candidates that:
- Require explaining who the streamer is or what game this is to land
- Depend on tone/inflection rather than visible action or dialogue
- Are only impressive to people who already understand the game mechanics

EXCLUDE or rank LAST any candidates that are:
- Singing, humming, or musical moments
- A single word or short phrase repeated multiple times with no other content
- Vague reactions with no clear story, trigger, or punchline

When assigning Score, use the FULL 1-10 range based on relative quality,
but be stricter than the discovery passes. Most candidates should be 4-7.
An 8 is a strong clip. A 9 is exceptional. A 10 should be used only for a
near-perfect standalone viral moment; most batches should have no 10 at all.
Do not give the same top score to many items.

Return ONLY the top {keep_n} ranked as CSV, no more:

Rank,Score,Timestamp,Title

One row per line, plain values only: rank as an integer, Score as a single
integer 1-10, Timestamp exactly as given (HH:MM:SS), Title as given. No
quotes around fields, no decimals, no extra columns or commentary.

"""

# ============================================================================
# Per-stage runners - each: loads its checkpoint input (if any), does the
# work inside stage_log() (captured to a persistent per-stage log file),
# saves its checkpoint output, records timing/stats, marks itself as the
# last-completed stage. Kept thin; actual logic lives in the functions above.
# ============================================================================

def run_stage_discovery(stream_folder):
    stage = "discovery"
    with stage_log(stream_folder, stage):
        stage_start = time.time()

        parts = list_transcript_parts(stream_folder)
        if not parts:
            print("No transcript_partN.txt files found in stream folder.")
            sys.exit(1)
        print(f"Found {len(parts)} transcript part(s): {', '.join(parts)}")

        highlights, part_errors = run_discovery(stream_folder, parts, PROMPTS)

        if not highlights and part_errors > 0:
            print(
                f"Discovery produced 0 candidates and {part_errors}/{len(parts)} transcript "
                f"part(s) raised an error while calling Ollama (see the ERROR lines above, "
                f"or log_discovery.txt) - this looks like a failed run (Ollama not running, "
                f"model failed to load, connection refused, etc.), not a stream that "
                f"genuinely has nothing worth clipping."
            )
            print(
                "Not saving a checkpoint for this, so the next run retries discovery from "
                "scratch instead of treating this empty result as done forever."
            )
            sys.exit(1)

        if not highlights:
            print("No candidates found by LLM discovery (audio scan may still find some).")

        save_checkpoint(stream_folder, STAGE_CHECKPOINT_NAMES[stage], highlights)
        record_stage_stats(stream_folder, stage, time.time() - stage_start)
        print(f"Saved {len(highlights)} candidate(s)")

def run_stage_audioscan(stream_folder):
    stage = "audioscan"
    with stage_log(stream_folder, stage):
        stage_start = time.time()

        highlights = require_checkpoint(stream_folder, STAGE_CHECKPOINT_NAMES["discovery"], stage)
        transcript_blocks_by_part = build_transcript_blocks_by_part(stream_folder)

        audio_candidates = find_audio_scan_candidates(stream_folder, highlights, transcript_blocks_by_part)

        # AUDIO_SCAN_STATS["candidates_kept"] is set by find_audio_scan_candidates()
        # whenever titling is reached (0 if audio scan is disabled, no mic
        # wav, no librosa, or no DSP peaks - all legitimate). candidates_kept
        # > 0 with zero titled results is the one non-legitimate case: real
        # candidates went to the model and it produced nothing usable for
        # any of them - the field-count bug this originally caught in
        # title_audio_candidates().
        candidates_kept = AUDIO_SCAN_STATS.get("candidates_kept", 0)
        if candidates_kept > 0 and not audio_candidates:
            print(
                f"[audio-scan] Found {candidates_kept} candidate(s) worth titling, but the "
                f"titling pass returned 0 usable titles for every single one - this looks "
                f"like a broken or truncated {JUDGE_MODEL} response, not a stream with "
                f"nothing worth titling. Check any debug_audioscan_titles_batch_*.txt just "
                f"written to this folder."
            )
            print(
                "[audio-scan] Not saving a checkpoint for this, so the next run retries "
                "the audio scan from scratch."
            )
            sys.exit(1)

        if audio_candidates:
            highlights.extend(audio_candidates)

        if not highlights:
            print("No highlights found from discovery or the audio scan.")
            # Note: no record_stage_stats() here - this stage never checkpointed,
            # so it hasn't actually "completed" and shouldn't be marked as such.
            sys.exit(1)

        before_merge_count = len(highlights)
        highlights = merge_near_duplicates(highlights, time_window=10)
        merged_count = before_merge_count - len(highlights)
        if merged_count > 0:
            print(f"Merged {merged_count} near-duplicate candidates")

        save_checkpoint(stream_folder, STAGE_CHECKPOINT_NAMES[stage], highlights)
        record_stage_stats(stream_folder, stage, time.time() - stage_start)
        print(f"Saved {len(highlights)} candidate(s)")

def run_stage_emotion(stream_folder):
    stage = "emotion"
    with stage_log(stream_folder, stage):
        stage_start = time.time()

        highlights = require_checkpoint(stream_folder, STAGE_CHECKPOINT_NAMES["audioscan"], stage)

        apply_emotion_scores_to_highlights(highlights, stream_folder)
        highlights.sort(key=lambda x: x["Score"], reverse=True)

        save_checkpoint(stream_folder, STAGE_CHECKPOINT_NAMES[stage], highlights)
        record_stage_stats(stream_folder, stage, time.time() - stage_start)
        print(f"Saved {len(highlights)} candidate(s)")

def run_stage_verify(stream_folder):
    stage = "verify"
    with stage_log(stream_folder, stage):
        stage_start = time.time()

        highlights = require_checkpoint(stream_folder, STAGE_CHECKPOINT_NAMES["emotion"], stage)

        before_trim = len(highlights)
        highlights = highlights[:VERIFY_POOL_SIZE]
        if before_trim > len(highlights):
            print(f"Trimmed {before_trim - len(highlights)} low-scoring candidate(s) before verification")

        transcript_blocks_by_part = build_transcript_blocks_by_part(stream_folder)
        verified, total_items, total_parsed = verify_candidates(
            highlights, transcript_blocks_by_part, VERIFY_PROMPT_HEADER, stream_folder
        )

        coverage_ratio = (total_parsed / total_items) if total_items else 1.0
        if total_items > 0 and coverage_ratio < VERIFY_MIN_COVERAGE_RATIO:
            print(
                f"Verification only parsed {total_parsed}/{total_items} verdicts "
                f"({coverage_ratio:.0%}) across the whole stage. Since an unparsed item is "
                f"kept rather than rejected, this doesn't mean the transcript supports "
                f"almost everything - it means {JUDGE_MODEL} mostly isn't producing "
                f"PASS/FAIL lines. Check one of the debug_verify_batch_*.txt files just "
                f"written to this folder to see the raw response."
            )
            print(
                "Not saving a checkpoint for this, so the next run retries verification "
                "from scratch instead of treating an unverified pass-through as done."
            )
            sys.exit(1)

        save_checkpoint(stream_folder, STAGE_CHECKPOINT_NAMES[stage], verified)
        record_stage_stats(stream_folder, stage, time.time() - stage_start)
        print(f"Saved {len(verified)} candidate(s)")

def run_stage_judge(stream_folder):
    stage = "judge"
    with stage_log(stream_folder, stage):
        stage_start = time.time()

        highlights = require_checkpoint(stream_folder, STAGE_CHECKPOINT_NAMES["verify"], stage)

        judge_pool = highlights[:JUDGE_POOL_SIZE]
        print(f"Running judge stage with {len(judge_pool)} candidates...")

        ranked = run_judge_tournament(judge_pool, JUDGE_INSTRUCTIONS, TOP_N)

        seen = set()
        final_highlights = []
        for h in ranked:
            key = (h["Timestamp"].strip(), h["Title"].strip().lower())
            if key in seen:
                continue
            seen.add(key)
            final_highlights.append(h)

        save_checkpoint(stream_folder, STAGE_CHECKPOINT_NAMES[stage], final_highlights)
        record_stage_stats(stream_folder, stage, time.time() - stage_start)
        print(f"Saved {len(final_highlights)} final candidate(s)")

def run_stage_export(stream_folder):
    stage = "export"
    with stage_log(stream_folder, stage):
        stage_start = time.time()

        final_highlights = require_checkpoint(stream_folder, STAGE_CHECKPOINT_NAMES["judge"], stage)

        calibrate_final_scores_by_rank(final_highlights)

        csv_path = write_highlights_csv(stream_folder, final_highlights, TOP_N)
        edl_path = write_highlights_edl(stream_folder, final_highlights, TOP_N)

        if EXPORT_PREVIEW_CLIPS:
            export_preview_clips(stream_folder, final_highlights, PREVIEW_CLIP_SECONDS_BEFORE, PREVIEW_CLIP_SECONDS_AFTER)

        record_stage_stats(stream_folder, stage, time.time() - stage_start)
        write_run_info(stream_folder, final_highlights)

        print("=" * 60)
        print(f"Top {TOP_N} highlights saved:")
        print(csv_path)
        print(edl_path)
        print("=" * 60)

        stats = load_pipeline_stats(stream_folder)
        total_seconds = sum(stats.get("stage_seconds", {}).values())
        print(
            f"Total pipeline time: {total_seconds / 60:.1f} min across all stages "
            f"({stats.get('ollama_calls', 0)} Ollama call(s), "
            f"{stats.get('ollama_seconds', 0) / 60:.1f} min in Ollama, "
            f"{stats.get('ollama_retries', 0)} retry/retries)"
        )
        for s, seconds in stats.get("stage_seconds", {}).items():
            print(f"  {s}: {seconds / 60:.1f} min")

STAGE_FUNCS = {
    "discovery": run_stage_discovery,
    "audioscan": run_stage_audioscan,
    "emotion": run_stage_emotion,
    "verify": run_stage_verify,
    "judge": run_stage_judge,
    "export": run_stage_export,
}

def stage_is_done(stream_folder, stage):
    """export has no checkpoint of its own (see STAGE_CHECKPOINT_NAMES) and
    always re-runs, so it's never considered 'done' for skip purposes."""
    name = STAGE_CHECKPOINT_NAMES.get(stage)
    if name is None:
        return False
    return checkpoint_exists(stream_folder, name)

def run_all_remaining_stages(stream_folder):
    """Default mode (no --stage flag): walk every stage in order, skipping
    any whose checkpoint already exists, running the rest through export.
    This is what RunAllSteps.bat and 5_AnalyzeHighlights.bat both use - if
    the previous attempt died partway through, whichever stages already
    checkpointed are skipped and the run picks back up exactly where it
    stopped, with no separate resume step required.

    export has no checkpoint of its own and always actually runs (see
    stage_is_done()), so reaching the end of this loop means a real export
    just happened - that's the signal record_pipeline_run_history() uses to
    log this as a completed run. Any stage exiting early (sys.exit(1), e.g.
    one of the checkpoint health checks refusing to save a bad result)
    skips this entirely, which is correct - an incomplete run shouldn't
    count toward the timing averages.
    """
    for stage in STAGE_ORDER:
        if stage_is_done(stream_folder, stage):
            print(f"=== {STAGE_LABELS[stage]}: checkpoint already exists, skipping ===")
            continue
        print(f"=== Running {STAGE_LABELS[stage]} ===")
        STAGE_FUNCS[stage](stream_folder)

    record_pipeline_run_history(stream_folder)

def run_single_stage_forced(stream_folder, stage):
    """Debug mode (--stage flag, used by the 5a-5f debug bats): force this
    ONE stage to run even if its checkpoint already exists, invalidate
    every downstream checkpoint (they're now stale relative to this stage's
    fresh output), then stop - it deliberately does not cascade into later
    stages, so you can re-test a single stage's prompt/logic in isolation.
    """
    invalidate_downstream(stream_folder, stage)
    print(f"=== Force-running {STAGE_LABELS[stage]} (debug mode - downstream checkpoints cleared) ===")
    STAGE_FUNCS[stage](stream_folder)

# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Emotion-enhanced highlight analyzer. Runs discovery, "
        "audio scan, emotion scoring, verification, judging, and export in "
        "sequence, checkpointing after each so a failed/stopped run can "
        "pick back up without redoing finished work."
    )
    parser.add_argument("stream_folder", help="Folder containing transcript_partN.txt and the *_mic.wav file.")
    parser.add_argument(
        "--stage", choices=list(STAGE_ORDER), default=None,
        help="Force-run just this one stage (used by the 5a-5f debug bats), "
        "clearing any downstream checkpoints since they'd be stale. Without "
        "this flag, runs every remaining stage in order, skipping ones "
        "already checkpointed."
    )
    args = parser.parse_args()

    stream_folder = args.stream_folder.replace('"', '').rstrip("\\/")
    stream_folder = os.path.abspath(stream_folder)

    print("=" * 60)
    print("Emotion-Enhanced Highlight Analyzer")
    print("=" * 60)
    print("Stream Folder:", stream_folder)
    print()

    if args.stage:
        run_single_stage_forced(stream_folder, args.stage)
    else:
        run_all_remaining_stages(stream_folder)

if __name__ == "__main__":
    main()
