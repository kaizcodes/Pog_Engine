"""Isolates a streamer's voice from a single merged audio track using Demucs.

A locally recorded OBS VOD has game/mic on separate tracks (plain ffmpeg
-map). A downloaded Twitch VOD flattens everything into one track, so
isolate_vocals.py runs a Demucs separation pass instead;
OrganizeVODAndFixSRT_Emotion.count_audio_streams() (ffprobe) picks the path.

Not meant to be run standalone in normal use - 1_ExtractMicAudio.bat calls:
  python isolate_vocals.py <mixed_audio.w64> <output_mic.wav>

The runner persists Demucs input chunks and trimmed mic chunks, concatenates
them into a Wave64 file, then renders <output_mic.wav> at 16kHz mono with the
shared noise gate - Step 1 finishes separation/rendering before Step 2 makes
its own Whisper chunks.

Input should stay separation-grade (44.1kHz stereo). The htdemucs weights
(~80MB) download on first use - see TORCH_CACHE_DIR below and
predownload_demucs_model() in pog_engine_setup.py.
"""

from __future__ import annotations

import argparse
import json
import math
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
    otherwise CPU works but is much slower on a multi-hour VOD. AMD needs no
    special case: ROCm torch for Windows exposes the same torch.cuda API, so
    is_available() is True on a 9070XT once the installer puts the ROCm
    build in."""
    if VOCAL_ISOLATION_DEVICE != "auto":
        return VOCAL_ISOLATION_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEMUCS_CHUNK_SECONDS = 10 * 60
DEMUCS_CHUNK_OVERLAP_SECONDS = 1


def _demucs_command(input_wav: Path, work_dir: Path, device: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", VOCAL_ISOLATION_MODEL,
        "--two-stems", "vocals",
        "--device", device,
        "-o", str(work_dir),
    ]
    if VOCAL_ISOLATION_SEGMENT_SECONDS:
        segment_seconds = int(VOCAL_ISOLATION_SEGMENT_SECONDS)
        if segment_seconds <= 0:
            raise ValueError("VOCAL_ISOLATION_SEGMENT_SECONDS must be positive")
        # HTDemucs' checkpoint was trained with a 7.8-second Transformer
        # window. Demucs rejects the CLI's integer segment values above 7;
        # omit the override so it uses the model-native 7.8-second window.
        if (
            VOCAL_ISOLATION_MODEL.lower().startswith("htdemucs")
            and segment_seconds > 7
        ):
            print(
                "[isolate-vocals] [!] Requested Demucs segment "
                f"{segment_seconds}s exceeds HTDemucs' 7.8s training window; "
                "using the model-native 7.8s window instead.",
                flush=True,
            )
        else:
            cmd += ["--segment", str(segment_seconds)]
    cmd.append(str(input_wav))
    return cmd


def _run_checked(cmd: list[str], description: str) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{description} exited with code {result.returncode}")


def _probe_duration_seconds(input_wav: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_wav),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe exited with code {result.returncode}")
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned an invalid duration: {result.stdout!r}") from exc
    if duration <= 0:
        raise RuntimeError(f"ffprobe returned a non-positive duration: {duration}")
    return duration


def _run_demucs_once(input_wav: Path, work_dir: Path, device: str) -> Path:
    cmd = _demucs_command(input_wav, work_dir, device)
    print("[isolate-vocals] Running:", " ".join(cmd), flush=True)
    _run_checked(cmd, "demucs")
    vocals_path = work_dir / VOCAL_ISOLATION_MODEL / input_wav.stem / "vocals.wav"
    if not vocals_path.exists():
        raise RuntimeError(f"demucs finished but expected output not found: {vocals_path}")
    return vocals_path




def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return candidate if isinstance(candidate, dict) else None


def _prepare_demucs_chunks(
    input_wav: Path,
    chunk_dir: Path,
    duration: float,
) -> list[dict[str, object]]:
    """Persist the full-mix windows that Demucs will receive."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = round(duration * 1_000)
    chunk_count = max(1, math.ceil(duration / DEMUCS_CHUNK_SECONDS))
    manifest_path = chunk_dir / "chunk_manifest.json"
    existing_manifest = _load_json(manifest_path)
    reusable = (
        existing_manifest is not None
        and existing_manifest.get("source_audio") == input_wav.name
        and existing_manifest.get("duration_ms") == duration_ms
        and existing_manifest.get("chunk_seconds") == DEMUCS_CHUNK_SECONDS
        and existing_manifest.get("overlap_seconds") == DEMUCS_CHUNK_OVERLAP_SECONDS
        and isinstance(existing_manifest.get("chunks"), list)
        and len(existing_manifest["chunks"]) == chunk_count
    )

    entries: list[dict[str, object]] = []
    for index in range(chunk_count):
        logical_start = index * DEMUCS_CHUNK_SECONDS
        logical_end = min(duration, logical_start + DEMUCS_CHUNK_SECONDS)
        left_overlap = DEMUCS_CHUNK_OVERLAP_SECONDS if index else 0
        right_overlap = (
            DEMUCS_CHUNK_OVERLAP_SECONDS if logical_end < duration else 0
        )
        chunk_start = max(0.0, logical_start - left_overlap)
        chunk_duration = logical_end - chunk_start + right_overlap
        chunk_path = chunk_dir / f"{input_wav.stem}_demucs_chunk_{index + 1:04d}.w64"
        if not (
            reusable
            and chunk_path.is_file()
            and chunk_path.stat().st_size > 0
        ):
            print(
                f"[isolate-vocals] Extracting Demucs chunk "
                f"{index + 1:02d}/{chunk_count:02d}: "
                f"{logical_start:.1f}s-{logical_end:.1f}s",
                flush=True,
            )
            _run_checked(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{chunk_start:.3f}", "-i", str(input_wav),
                    "-t", f"{chunk_duration:.3f}",
                    "-map", "0:a:0",
                    "-ar", "44100", "-ac", "2",
                    "-c:a", "pcm_s16le", "-f", "w64", str(chunk_path),
                ],
                "ffmpeg Demucs chunk extraction",
            )
        entries.append(
            {
                "filename": chunk_path.name,
                "logical_start_ms": round(logical_start * 1_000),
                "logical_end_ms": round(logical_end * 1_000),
                "left_overlap_ms": left_overlap * 1_000,
                "right_overlap_ms": right_overlap * 1_000,
            }
        )

    _write_json_atomic(
        manifest_path,
        {
            "source_audio": input_wav.name,
            "duration_ms": duration_ms,
            "chunk_seconds": DEMUCS_CHUNK_SECONDS,
            "overlap_seconds": DEMUCS_CHUNK_OVERLAP_SECONDS,
            "chunks": entries,
        },
    )
    print(
        f"[isolate-vocals] Demucs input chunks ready in {chunk_dir}",
        flush=True,
    )
    return entries


def _prepare_mic_chunks(
    input_wav: Path,
    demucs_chunk_dir: Path,
    mic_chunk_dir: Path,
    entries: list[dict[str, object]],
    duration: float,
    work_dir: Path,
    device: str,
) -> list[Path]:
    """Run Demucs per saved input chunk and persist trimmed vocal chunks."""
    mic_chunk_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = round(duration * 1_000)
    manifest_path = mic_chunk_dir / "chunk_manifest.json"
    existing_manifest = _load_json(manifest_path)
    reusable = (
        existing_manifest is not None
        and existing_manifest.get("source_audio") == input_wav.name
        and existing_manifest.get("duration_ms") == duration_ms
        and existing_manifest.get("chunk_seconds") == DEMUCS_CHUNK_SECONDS
        and existing_manifest.get("overlap_seconds") == DEMUCS_CHUNK_OVERLAP_SECONDS
        and isinstance(existing_manifest.get("chunks"), list)
        and len(existing_manifest["chunks"]) == len(entries)
    )

    mic_paths: list[Path] = []
    mic_entries: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        demucs_filename = entry["filename"]
        if not isinstance(demucs_filename, str):
            raise RuntimeError("Demucs manifest contains an invalid chunk filename.")
        demucs_path = demucs_chunk_dir / demucs_filename
        if not demucs_path.is_file() or demucs_path.stat().st_size <= 0:
            raise RuntimeError(f"Demucs chunk is missing or empty: {demucs_path}")

        mic_path = mic_chunk_dir / (
            f"{input_wav.stem}_mic_chunk_{index + 1:04d}.w64"
        )
        if not (
            reusable
            and mic_path.is_file()
            and mic_path.stat().st_size > 0
        ):
            chunk_work_dir = work_dir / f"demucs_{index + 1:04d}"
            chunk_vocals = _run_demucs_once(
                demucs_path,
                chunk_work_dir,
                device,
            )
            logical_start_ms = entry["logical_start_ms"]
            logical_end_ms = entry["logical_end_ms"]
            left_overlap_ms = entry["left_overlap_ms"]
            if not all(
                isinstance(value, int)
                for value in (
                    logical_start_ms,
                    logical_end_ms,
                    left_overlap_ms,
                )
            ):
                raise RuntimeError("Demucs manifest contains invalid chunk timing.")
            logical_duration_ms = logical_end_ms - logical_start_ms
            _run_checked(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(chunk_vocals),
                    "-ss", f"{left_overlap_ms / 1_000:.3f}",
                    "-t", f"{logical_duration_ms / 1_000:.3f}",
                    "-ar", "44100", "-ac", "2",
                    "-c:a", "pcm_s16le", "-f", "w64", str(mic_path),
                ],
                "ffmpeg Demucs vocal chunk trim",
            )
        mic_paths.append(mic_path)
        mic_entries.append(
            {
                "filename": mic_path.name,
                "logical_start_ms": entry["logical_start_ms"],
                "logical_end_ms": entry["logical_end_ms"],
            }
        )

    _write_json_atomic(
        manifest_path,
        {
            "source_audio": input_wav.name,
            "duration_ms": duration_ms,
            "chunk_seconds": DEMUCS_CHUNK_SECONDS,
            "overlap_seconds": DEMUCS_CHUNK_OVERLAP_SECONDS,
            "chunks": mic_entries,
        },
    )
    print(
        f"[isolate-vocals] Mic chunks ready in {mic_chunk_dir}",
        flush=True,
    )
    return mic_paths


def _combine_mic_chunks(mic_paths: list[Path], combined_path: Path) -> Path:
    """Concatenate trimmed, separated vocal chunks into one Wave64 file."""
    if not mic_paths:
        raise RuntimeError("No separated microphone chunks were produced.")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = combined_path.with_name(combined_path.name + ".concat.txt")
    concat_list.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in mic_paths),
        encoding="utf-8",
    )
    try:
        _run_checked(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c", "copy", "-f", "w64", str(combined_path),
            ],
            "ffmpeg separated vocal concatenation",
        )
    finally:
        concat_list.unlink(missing_ok=True)
    return combined_path


def run_demucs_chunked(
    input_wav: Path,
    work_dir: Path,
    *,
    demucs_chunk_dir: Path | None = None,
    mic_chunk_dir: Path | None = None,
    combined_path: Path | None = None,
) -> Path:
    """Persist Demucs input/mic chunks, then combine the trimmed mic chunks."""
    duration = _probe_duration_seconds(input_wav)
    demucs_chunk_dir = demucs_chunk_dir or input_wav.parent / (
        f"{input_wav.stem}_demucs_chunks"
    )
    mic_chunk_dir = mic_chunk_dir or input_wav.parent / (
        f"{input_wav.stem}_mic_demucs_chunks"
    )
    combined_path = combined_path or input_wav.parent / (
        f"{input_wav.stem}_mic_demucs_combined.w64"
    )

    device = detect_device()
    print(f"[isolate-vocals] Model: {VOCAL_ISOLATION_MODEL}  Device: {device}", flush=True)
    if device == "cpu":
        print("[isolate-vocals] No GPU in use - this will be much slower on a long VOD.", flush=True)

    demucs_entries = _prepare_demucs_chunks(
        input_wav,
        demucs_chunk_dir,
        duration,
    )
    mic_paths = _prepare_mic_chunks(
        input_wav,
        demucs_chunk_dir,
        mic_chunk_dir,
        demucs_entries,
        duration,
        work_dir,
        device,
    )
    combined = _combine_mic_chunks(mic_paths, combined_path)
    print(f"[isolate-vocals] Combined separated mic audio: {combined}", flush=True)
    return combined


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
    parser = argparse.ArgumentParser(
        description="Separate a mixed VOD track into a rendered microphone track."
    )
    parser.add_argument("input_wav", help="Full mixed audio, normally Wave64.")
    parser.add_argument("output_wav", help="Rendered 16kHz mono microphone WAV.")
    parser.add_argument("--demucs-chunk-dir", type=Path)
    parser.add_argument("--mic-chunk-dir", type=Path)
    parser.add_argument("--combined-path", type=Path)
    args = parser.parse_args()

    use_local_torch_cache()

    input_wav = Path(args.input_wav).resolve()
    output_wav = Path(args.output_wav).resolve()

    if not input_wav.exists():
        print(f"ERROR: input file not found: {input_wav}")
        return 1

    try:
        import demucs  # noqa: F401
    except ImportError:
        print("ERROR: the 'demucs' package isn't installed.")
        print("       Run Install_PogEngine.bat again (it installs this), or: pip install demucs")
        return 1

    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{input_wav.stem}_demucs_",
            dir=str(input_wav.parent),
        )
    )
    print(f"[isolate-vocals] Temporary Demucs workspace: {work_dir}", flush=True)
    try:
        vocals_wav = run_demucs_chunked(
            input_wav,
            work_dir,
            demucs_chunk_dir=args.demucs_chunk_dir.resolve()
            if args.demucs_chunk_dir
            else None,
            mic_chunk_dir=args.mic_chunk_dir.resolve()
            if args.mic_chunk_dir
            else None,
            combined_path=args.combined_path.resolve()
            if args.combined_path
            else None,
        )
        print(f"[isolate-vocals] Combined separated mic audio: {vocals_wav}")
        finalize_mic_wav(vocals_wav, output_wav)
        print(f"[isolate-vocals] Saved rendered voice track: {output_wav}")
    except Exception as exc:
        print(f"ERROR: vocal isolation failed: {exc}")
        return 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
