"""Isolates a streamer's voice from a single merged audio track using Demucs
source separation.

Why this exists: a locally recorded OBS VOD has game/desktop audio and mic
on separate tracks, so getting a clean voice track is just ffmpeg -map. A
downloaded Twitch VOD has everything - game audio, music, alerts, mic - all
flattened into one track, so there's no track to pull out; the voice has to
be separated out of the mix. OrganizeVODAndFixSRT_Emotion.py's
count_audio_streams() (ffprobe) detects which case a dropped video is and
only routes here for the single-track case - see
make_extract_mic_bat_singletrack() there.

Not meant to be run standalone in normal use - the auto-generated
1_ExtractMicAudio.bat calls this after extracting the full mix with ffmpeg:

  python isolate_vocals.py <mixed_audio.wav> <output_mic.wav>

<mixed_audio.wav> should be the full-quality mixed track (44.1kHz stereo is
what the .bat extracts, since downsampling before separation would hurt
separation quality). <output_mic.wav> is written at 16kHz mono with the same
noise gate the multi-track path applies, so everything downstream
(2_TranscribeAudio.bat, the emotion model, the audio-scan sidecar - all of
which just look for *_mic.wav) sees an identical format regardless of which
path produced it.

Demucs downloads its pretrained separation model (~80MB, htdemucs by
default) the first time it runs on a machine - see TORCH_CACHE_DIR below for
where it's cached. That one download needs internet access; every run after
that is fully offline. pog_engine_setup.py can also trigger this download
during setup instead of leaving it for the first real VOD - see
predownload_demucs_model() there.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import (
    NOISE_GATE_ATTACK_MS,
    NOISE_GATE_RATIO,
    NOISE_GATE_RELEASE_MS,
    NOISE_GATE_THRESHOLD_DB,
    VOCAL_ISOLATION_DEVICE,
    VOCAL_ISOLATION_MODEL,
    VOCAL_ISOLATION_SEGMENT_SECONDS,
)

# Machine-specific path - deliberately kept hardcoded here (not in
# pipeline_config.py) and patched by pog_engine_setup.py's patch_paths(),
# same pattern as EMOTION_LOCAL_MODEL_DIR in analyze_highlights_emotion.py.
# Redirects torch/demucs' model cache (normally
# %USERPROFILE%\.cache\torch\hub) into a subfolder of Pog_Engine's own
# models\ folder instead, so the separation model lives alongside
# whisper.cpp's and the emotion model's files rather than scattered in the
# user profile. A subfolder rather than models\ directly, since torch hub
# creates its own checkpoints\ subdirectory structure there that shouldn't
# mix with the flat model files check_models() in pog_engine_setup.py
# expects to see.
TORCH_CACHE_DIR = r"G:\pog_dev\models\torch_cache"
HF_CACHE_DIR = r"G:\pog_dev\models\hf_cache"



def use_local_torch_cache() -> None:
    """Points torch's hub cache AND huggingface_hub's snapshot cache at this
    script's Pog_Engine folder instead of the default user-profile location,
    so the separation model lives alongside whisper.cpp's and the emotion
    model's files rather than scattered in the user profile.

    Sets both the broad *_HOME vars AND the precise *_HUB_CACHE / TORCH_HUB_CACHE
    overrides: newer torch (>=2.4) and huggingface_hub honor the precise ones
    over the *_HOME vars, and on some installs HF_HOME alone silently fails to
    redirect the snapshot dir. Without HF_HUB_CACHE here, demucs >=4 fetches
    into ~/.cache/huggingface instead of the redirect - so the runtime and the
    setup's predownload_demucs_model() must set the same vars or they read and
    write different caches. Best effort: any OSError creating the folder falls
    back silently to torch's/HF's own defaults rather than block separation.
    """
    try:
        Path(TORCH_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", TORCH_CACHE_DIR)
        os.environ.setdefault("TORCH_HUB_CACHE", str(Path(TORCH_CACHE_DIR) / "hub"))
    except OSError as exc:
        print(f"[isolate-vocals] [!] Could not create TORCH_CACHE_DIR ({TORCH_CACHE_DIR}): {exc}")
        print("[isolate-vocals]     Falling back to torch's default cache location.")

    try:
        Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
        os.environ.setdefault("HF_HUB_CACHE", str(Path(HF_CACHE_DIR) / "huggingface" / "hub"))
    except OSError as exc:
        print(f"[isolate-vocals] [!] Could not create HF_CACHE_DIR ({HF_CACHE_DIR}): {exc}")
        print("[isolate-vocals]     Falling back to HuggingFace's default cache location.")


def detect_device() -> str:
    """VOCAL_ISOLATION_DEVICE == "auto" (the default) picks CUDA if
    available, same as the emotion model in analyze_highlights_emotion.py -
    otherwise CPU works but is much slower on a multi-hour VOD."""
    if VOCAL_ISOLATION_DEVICE != "auto":
        return VOCAL_ISOLATION_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_demucs(input_wav: Path, work_dir: Path) -> Path:
    """Runs Demucs as a subprocess (rather than importing its API directly)
    so this script's own dependencies stay minimal and its behavior matches
    running `python -m demucs` by hand for debugging. Returns the path to
    the separated vocals.wav stem."""
    device = detect_device()
    print(f"[isolate-vocals] Model: {VOCAL_ISOLATION_MODEL}  Device: {device}")
    if device == "cpu":
        print("[isolate-vocals] No GPU in use - this will be much slower on a long VOD.")

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", VOCAL_ISOLATION_MODEL,
        "--two-stems", "vocals",
        "--device", device,
        "-o", str(work_dir),
    ]
    if VOCAL_ISOLATION_SEGMENT_SECONDS:
        cmd += ["--segment", str(VOCAL_ISOLATION_SEGMENT_SECONDS)]
    cmd.append(str(input_wav))

    print("[isolate-vocals] Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"demucs exited with code {result.returncode}")

    # Demucs writes to {out}/{model_name}/{track_stem}/{stem}.wav.
    vocals_path = work_dir / VOCAL_ISOLATION_MODEL / input_wav.stem / "vocals.wav"
    if not vocals_path.exists():
        raise RuntimeError(f"demucs finished but expected output not found: {vocals_path}")
    return vocals_path


def finalize_mic_wav(vocals_wav: Path, output_wav: Path) -> None:
    """Downmixes the separated vocal stem to 16kHz mono and applies the
    pipeline's standard noise gate, matching the exact format
    2_TranscribeAudio.bat (whisper.cpp) and the emotion model expect."""
    audio_filters = (
        f"agate=threshold={NOISE_GATE_THRESHOLD_DB}dB:"
        f"ratio={NOISE_GATE_RATIO}:"
        f"attack={NOISE_GATE_ATTACK_MS}:"
        f"release={NOISE_GATE_RELEASE_MS}"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(vocals_wav),
        "-ar", "16000", "-ac", "1",
        "-af", audio_filters,
        str(output_wav),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg finalization exited with code {result.returncode}")


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python isolate_vocals.py <mixed_audio.wav> <output_mic.wav>")
        return 1

    use_local_torch_cache()

    input_wav = Path(sys.argv[1]).resolve()
    output_wav = Path(sys.argv[2]).resolve()

    if not input_wav.exists():
        print(f"ERROR: input file not found: {input_wav}")
        return 1

    try:
        import demucs  # noqa: F401
    except ImportError:
        print("ERROR: the 'demucs' package isn't installed.")
        print("       Run Install_PogEngine.bat again (it installs this), or: pip install demucs")
        return 1

    work_dir = Path(tempfile.mkdtemp(prefix="pog_demucs_"))
    try:
        vocals_wav = run_demucs(input_wav, work_dir)
        print(f"[isolate-vocals] Vocal stem separated: {vocals_wav}")
        finalize_mic_wav(vocals_wav, output_wav)
        print(f"[isolate-vocals] Saved isolated voice track: {output_wav}")
    except Exception as exc:
        print(f"ERROR: vocal isolation failed: {exc}")
        return 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
