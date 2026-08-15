"""Setup / verification helper for the Pog_Engine VOD highlight pipeline.

Not meant to be run directly - Install_PogEngine.bat calls this after making
sure some version of Python exists. It:

  1. Checks the Pog_Engine folder for the 6 pipeline scripts, the 5 required
     files in the models folder, a whisper-cli.exe (whisper.cpp CUDA build),
     and ffmpeg/ffprobe on PATH.
  2. Patches the machine-specific path constants (WHISPER_CLI, WHISPER_MODEL,
     WHISPER_VAD, GALLERY_DIR, EMOTION_LOCAL_MODEL_DIR/FILE) in the two
     pipeline scripts to point at THIS machine's Pog_Engine folder, instead
     of whatever machine they were last edited on.
  3. Installs the Python packages the pipeline actually imports (requests,
     numpy, torch, demucs, transformers, librosa, soundfile, safetensors,
     Pillow), skipping anything already present - and picks a CUDA build of
     torch over a CPU-only one whenever an NVIDIA GPU is detected. demucs
     (used by isolate_vocals.py to separate a streamer's voice out of a
     single-track/Twitch-style VOD's merged audio) is installed after torch
     so it picks up that same build; its own audio I/O goes through
     ffmpeg/ffprobe directly rather than torchaudio, which is why those are
     checked separately below.
  4. Checks (does not install) Ollama and the qwen models pipeline_config.py
     expects - installing those is left to you, same as before.

Safe to re-run any time - every step only fixes/installs what's actually
missing, and re-running does not undo anything you've already fixed by hand.

Usage:
  python pog_engine_setup.py [folder]          -> opens the GUI, folder is
                                                   just a prefilled suggestion
  python pog_engine_setup.py [folder] --cli    -> console-only, no GUI
(Install_PogEngine.bat always calls this with its own folder as [folder].)
"""

# Keeps every `X | None` / `list[str]` type hint below as an unevaluated
# string, so this file still parses and runs fine even if the Python already
# on the machine turns out to be older than the 3.10 we recommend - the
# version check further down can then print a clear warning instead of the
# script just crashing on import with a SyntaxError/TypeError.
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import queue
import re
import requests
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
import wave
import zipfile
from pathlib import Path
REQUIRED_SCRIPTS = [
    "analyze_highlights_emotion.py",
    "OrganizeVODAndFixSRT_Emotion.py",
    "OrganizeVODAndFixSRT_Emotion.bat",
    "pipeline_config.py",
    # Vocal isolation for single-track (Twitch-style) VODs - see
    # count_audio_streams() / make_extract_mic_bat_singletrack() in
    # OrganizeVODAndFixSRT_Emotion.py.
    "isolate_vocals.py",
]

REQUIRED_MODEL_FILES = [
    "ggml-large-v3.bin",
    "ggml-silero-v6.2.0.bin",
    "speech-emotion-recognition-with-openai-whisper-large-v3.safetensors",
    "config.json",
    "preprocessor_config.json",
]

# module name (what `import` uses) -> pip package name (what `pip install` uses)
SIMPLE_PACKAGES = {
    "requests": "requests",
    "numpy": "numpy",
    "transformers": "transformers",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "safetensors": "safetensors",
    "PIL": "Pillow",
}

# PyTorch CUDA wheel index tags, newest first - checked against the driver's
# max supported CUDA version, so this list can just be extended over time as
# PyTorch adds new tags without changing the logic around it.
CUDA_WHEEL_TAGS = [
    (13, 0, "cu130"),
    (12, 9, "cu129"),
    (12, 8, "cu128"),
    (12, 6, "cu126"),
    (11, 8, "cu118"),
]


def mark(ok: bool) -> str:
    return "[OK]     " if ok else "[MISSING]"


def is_importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


# ============================================================================
# Reporter - the only thing that differs between --cli and the GUI. Every
# check/install function below takes one of these instead of print()-ing or
# updating widgets directly, so the exact same logic drives both.
# ============================================================================

class Reporter:
    """Console default. GuiReporter (in run_gui) overrides all of this to
    push onto a thread-safe queue instead."""

    def log(self, msg: str = "") -> None:
        print(msg)

    def status(self, key: str, value: str) -> None:
        pass  # console mode already shows this via the log() lines above

    def add_row(self, section: str, key: str, label: str) -> None:
        pass  # GUI-only: register a row not known until runtime (model names)

    def section(self, title: str) -> None:
        print()
        print("=" * 70)
        print(title)
        print("=" * 70)


# ============================================================================
# 1. Folder / file inventory
# ============================================================================

def check_scripts(pog_dir: Path, reporter: Reporter) -> list[str]:
    reporter.section("Checking for the 5 Pog_Engine scripts")
    missing = []
    for name in REQUIRED_SCRIPTS:
        ok = (pog_dir / name).is_file()
        reporter.log(f"  {mark(ok)} {name}")
        reporter.status(name, "OK" if ok else "Missing")
        if not ok:
            missing.append(name)
    reporter.log(f"  {mark(True)} pog_engine_setup.py (this script)")
    reporter.status("pog_engine_setup.py", "OK")
    return missing


def check_models(pog_dir: Path, reporter: Reporter) -> tuple[Path, list[str], Path | None]:
    reporter.section("Checking models folder")
    models_dir = pog_dir / "models"
    if not models_dir.is_dir():
        reporter.log("  [MISSING] models folder does not exist yet")
        for name in REQUIRED_MODEL_FILES:
            reporter.status(name, "Missing")
        # Also check for whisper-cli.exe in models/Release/
        reporter.status("whisper-cli.exe", "Missing")
        return models_dir, list(REQUIRED_MODEL_FILES) + ["whisper-cli.exe"], None

    missing = []
    for name in REQUIRED_MODEL_FILES:
        ok = (models_dir / name).is_file()
        reporter.log(f"  {mark(ok)} models\\{name}")
        reporter.status(name, "OK" if ok else "Missing")
        if not ok:
            missing.append(name)
    # Check for whisper-cli.exe in models/Release/
    whisper_cli_path = models_dir / "Release" / "whisper-cli.exe"
    whisper_cli_ok = whisper_cli_path.is_file()
    reporter.log(f"  {mark(whisper_cli_ok)} models\\Release\\whisper-cli.exe")
    reporter.status("whisper-cli.exe", "OK" if whisper_cli_ok else "Missing")
    if not whisper_cli_ok:
        missing.append("whisper-cli.exe")
        whisper_cli_path = None
    return models_dir, missing, whisper_cli_path


def remote_file_size(url: str) -> int | None:
    """Best-effort HEAD request for the origin's content-length in bytes.

    Hugging Face resolve URLs redirect to object storage, so redirects must
    be followed before reading Content-Length; otherwise this would measure
    the tiny redirect response body instead of the model.
    Returns None when the server doesn't answer or reports no length, so
    callers can't verify and keep the existing file -- never a false
    "re-download" from a broken HEAD.
    """
    try:
        with requests.head(url, allow_redirects=True, timeout=60) as r:
            if 200 <= r.status_code < 300:
                cl = r.headers.get("content-length")
                if cl and cl.isdigit():
                    return int(cl)
    except Exception:
        pass
    return None


_PINNED_SHA256 = {
    # GitHub release assets are immutable: the digest is published by the
    # GitHub API (releases/tags/v1.7.6 -> assets[].digest) and can never
    # change for this tag. Verified 2026-08-11.
    "https://github.com/ggml-org/whisper.cpp/releases/download/v1.7.6/whisper-cublas-12.4.0-bin-x64.zip":
        "3fc4d3ebd9a678313de50c04d9e59c43117ae190f0cb7bff602d4aeefc4efe3d",
}

_HF_SHA_CACHE: dict[tuple[str, str], dict[str, str]] = {}


def origin_sha256(url: str) -> str | None:
    """The origin's SHA-256 for a file URL, when the source publishes one.

    - huggingface.co resolve URLs: the repo tree API reports the LFS oid
      (a real sha256) per file; the response is cached per repo/revision so
      one API call serves every file of that repo. Small non-LFS files
      (e.g. config.json) have no sha256 and return None.
    - Pinned immutable GitHub release assets are known upfront in
      _PINNED_SHA256.
    - Any other origin returns None, and the caller falls back to the
      Content-Length check only.
    """
    pinned = _PINNED_SHA256.get(url)
    if pinned is not None:
        return pinned
    m = re.match(r"https://huggingface\.co/([^/]+)/([^/]+)/resolve/([^/]+)/(.+)$", url)
    if not m:
        return None
    repo, rev, path = f"{m.group(1)}/{m.group(2)}", m.group(3), m.group(4)
    key = (repo, rev)
    if key not in _HF_SHA_CACHE:
        try:
            r = requests.get(
                f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true",
                timeout=60,
            )
            r.raise_for_status()
            tree = {}
            for entry in r.json():
                lfs = entry.get("lfs") or {}
                oid = lfs.get("oid", "")
                if len(oid) == 64:
                    tree[entry["path"]] = oid
            _HF_SHA_CACHE[key] = tree
        except Exception:
            _HF_SHA_CACHE[key] = {}
    return _HF_SHA_CACHE[key].get(path)


def sha256_file(path: Path) -> str:
    """Full-file SHA-256 hex digest (chunked; a 3 GB model takes a few seconds)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_to_temp(url: str, temp_path: Path, reporter: Reporter) -> None:
    """Download one file to a temp path and verify its published size/hash."""
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        total = int(content_length) if content_length and content_length.isdigit() else 0
        if total == 0:
            total = remote_file_size(url) or 0

        downloaded = 0
        with open(temp_path, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = downloaded * 100 // total
                    if percent % 10 == 0:
                        reporter.log(
                            f"    {percent}% ({downloaded // (1024 * 1024)}MB / "
                            f"{total // (1024 * 1024)}MB)"
                        )

    if total > 0 and downloaded != total:
        raise IOError(f"download truncated: got {downloaded} bytes, expected {total}")

    expected_hash = origin_sha256(url)
    if expected_hash is not None:
        actual_hash = sha256_file(temp_path)
        if actual_hash != expected_hash:
            raise IOError(
                f"checksum mismatch: got {actual_hash[:12]}..., "
                f"expected {expected_hash[:12]}..."
            )
        reporter.log(f"    sha256 verified ({expected_hash[:12]}...)")


def download_file(url: str, dest_path: Path, reporter: Reporter, description: str) -> bool:
    """Stream download with progress, skip-if-exists, atomic rename via temp file.

    Skip-if-exists is not blind: the origin's content-length (HEAD) is
    compared against the local file, and a mismatched size re-downloads
    instead of trusting a truncated/partial file as complete. When the
    origin publishes a SHA-256 (HF LFS / pinned GitHub asset), the local
    file must match it too. Fresh downloads are likewise rejected (temp
    deleted) when the received byte count doesn't match the announced
    length or the bytes don't match the origin's sha256, so a bad
    first download can't be renamed over the destination.
    """
    if dest_path.is_file():
        local_size = dest_path.stat().st_size
        size_expected = remote_file_size(url)
        size_ok = size_expected is None or local_size == size_expected
        hash_expected = origin_sha256(url)
        hash_ok = hash_expected is None or sha256_file(dest_path) == hash_expected
        if size_ok and hash_ok:
            verified = " (sha256 verified)" if hash_expected is not None else ""
            reporter.log(f"  [OK]     {description} already exists at {dest_path}{verified}")
            reporter.status(description, "Already downloaded")
            return True
        reasons = []
        if not size_ok:
            reasons.append(f"size {local_size} != expected {size_expected}")
        if not hash_ok:
            reasons.append("sha256 mismatch")
        reporter.log(f"  [WARN]   {description} exists but {'; '.join(reasons)}, re-downloading")

    reporter.log(f"  [INFO]   Downloading {description}...")
    reporter.status(description, "Downloading...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    try:
        _download_to_temp(url, tmp_path, reporter)
        tmp_path.replace(dest_path)
        reporter.log(f"  [OK]     {description} downloaded to {dest_path}")
        reporter.status(description, "Downloaded")
        return True
    except Exception as exc:
        reporter.log(f"  [WARN] {description} download failed: {exc}")
        reporter.status(description, "Failed")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False


def download_whisper_model(models_dir: Path, reporter: Reporter) -> bool:
    return download_file(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        models_dir / "ggml-large-v3.bin",
        reporter,
        "whisper.cpp model (ggml-large-v3.bin)"
    )


def download_whisper_vad(models_dir: Path, reporter: Reporter) -> bool:
    return download_file(
        "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin",
        models_dir / "ggml-silero-v6.2.0.bin",
        reporter,
        "Whisper VAD (ggml-silero-v6.2.0.bin)"
    )


def download_emotion_model_files(models_dir: Path, reporter: Reporter) -> bool:
    base = "https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3/resolve/main/"
    ok = True
    ok &= download_file(base + "model.safetensors", 
        models_dir / "speech-emotion-recognition-with-openai-whisper-large-v3.safetensors", 
        reporter, "Emotion model (safetensors)")
    ok &= download_file(base + "config.json",
        models_dir / "config.json",
        reporter, "Emotion model config.json")
    ok &= download_file(base + "preprocessor_config.json",
        models_dir / "preprocessor_config.json",
        reporter, "Emotion model preprocessor_config.json")
    return ok


def download_whisper_cpp_cublas(models_dir: Path, reporter: Reporter) -> bool:
    """Download whisper.cpp cublas release ZIP and extract whisper-cli.exe to models/Release/"""
    dest_exe = models_dir / "Release" / "whisper-cli.exe"
    if dest_exe.is_file():
        reporter.log(f"  [OK]     whisper-cli.exe already exists at {dest_exe}")
        reporter.status("whisper-cli.exe", "Already downloaded")
        return True

    reporter.log(f"  [INFO]   Downloading whisper.cpp cublas release...")
    reporter.status("whisper.cpp cublas", "Downloading...")
    dest_exe.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ggml-org/whisper.cpp/releases/download/v1.7.6/whisper-cublas-12.4.0-bin-x64.zip"
    tmp_zip = models_dir / "whisper-cublas.zip.tmp"
    try:
        _download_to_temp(url, tmp_zip, reporter)

        reporter.log(f"  [INFO]   Extracting whisper.cpp release...")
        import shutil
        tmp_extract = models_dir / "whisper-cublas-extract.tmp"
        with zipfile.ZipFile(tmp_zip, 'r') as z:
            z.extractall(tmp_extract)
        # The ZIP contains a top-level Release/ folder; move its contents to models/Release/
        inner_release = tmp_extract / "Release"
        if inner_release.is_dir():
            for item in inner_release.iterdir():
                shutil.move(str(item), str(dest_exe.parent / item.name))
            reporter.log(f"  [OK]     Extracted whisper.cpp release to {dest_exe.parent}")
            reporter.status("whisper.cpp cublas", "Downloaded")
            tmp_zip.unlink(missing_ok=True)
            shutil.rmtree(tmp_extract, ignore_errors=True)
            return True
        else:
            reporter.log(f"  [WARN] Release folder not found in ZIP")
            reporter.status("whisper.cpp cublas", "Failed")
            return False
    except Exception as exc:
        reporter.log(f"  [WARN] whisper.cpp cublas download/extract failed: {exc}")
        reporter.status("whisper.cpp cublas", "Failed")
        if tmp_zip.exists():
            tmp_zip.unlink(missing_ok=True)
        return False


def download_all_models(models_dir: Path, reporter: Reporter) -> None:
    reporter.section("Downloading model files (skip-if-exists)")
    download_whisper_model(models_dir, reporter)
    download_whisper_vad(models_dir, reporter)
    download_emotion_model_files(models_dir, reporter)
    download_whisper_cpp_cublas(models_dir, reporter)


def find_whisper_cli(pog_dir: Path, reporter: Reporter) -> Path | None:
    reporter.section("Checking for whisper.cpp (CUDA / cublas build)")
    for candidate in pog_dir.rglob("whisper-cli.exe"):
        reporter.log(f"  {mark(True)} found: {candidate}")
        reporter.status("whisper-cli.exe", "OK")
        return candidate
    reporter.log(f"  {mark(False)} whisper-cli.exe not found anywhere under {pog_dir}")
    reporter.log("      -> download the CUDA/cublas build from the whisper.cpp GitHub")
    reporter.log("         releases page, unzip it, and place the folder inside Pog_Engine.")
    reporter.status("whisper-cli.exe", "Missing")
    return None


def check_ffmpeg(reporter: Reporter) -> bool:
    """ffmpeg is used throughout (mic extraction, transcode, preview clips)
    and was never explicitly checked before - now that ffprobe also drives
    the single-track-vs-multi-track VOD detection (see count_audio_streams()
    in OrganizeVODAndFixSRT_Emotion.py), a missing install silently falls
    back to "assume multi-track" instead of failing loudly, so it's worth
    surfacing here."""
    reporter.section("Checking for ffmpeg / ffprobe")
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    reporter.log(f"  {mark(ffmpeg_ok)} ffmpeg on PATH")
    reporter.status("ffmpeg", "OK" if ffmpeg_ok else "Missing")
    reporter.log(f"  {mark(ffprobe_ok)} ffprobe on PATH")
    reporter.status("ffprobe", "OK" if ffprobe_ok else "Missing")
    if not (ffmpeg_ok and ffprobe_ok):
        reporter.log("      -> both ship together: install ffmpeg (https://ffmpeg.org/download.html)")
        reporter.log("         and add its bin folder to PATH. ffprobe is what auto-detects whether a")
        reporter.log("         dropped VOD is a single-track Twitch-style file or a locally recorded")
        reporter.log("         file with separate game/mic tracks - without it, every VOD is assumed")
        reporter.log("         to have separate tracks (safe, but wrong for a Twitch VOD).")
    return ffmpeg_ok and ffprobe_ok


# ============================================================================
# 2. Patch machine-specific paths in the two pipeline scripts
# ============================================================================

def patch_raw_string_constant(file_path: Path, var_name: str, new_value: str,
                               path_wrapped: bool, reporter: Reporter) -> None:
    """Rewrite `VAR = r"..."` (or `VAR = Path(r"...")`) to point at new_value,
    leaving everything else in the file - including its existing line-ending
    style - untouched. Uses open() rather than Path.read_text/write_text
    since the newline="" parameter those gained is Python 3.13+ only."""
    with open(file_path, encoding="utf-8", newline="") as f:
        text = f.read()
    if path_wrapped:
        pattern = rf'^{re.escape(var_name)} = Path\(r".*?"\)'
        replacement = f'{var_name} = Path(r"{new_value}")'
    else:
        pattern = rf'^{re.escape(var_name)} = r".*?"'
        replacement = f'{var_name} = r"{new_value}"'

    # A lambda, not the plain string, as the replacement: re.sub/subn treats
    # backslashes in a *string* replacement as escape sequences (\1, \g<0>,
    # etc.), so a Windows path like G:\Pog_Engine\... blows up with "bad
    # escape \P". A callable's return value is inserted literally instead.
    new_text, count = re.subn(pattern, lambda m: replacement, text, count=1, flags=re.MULTILINE)
    if count == 0:
        reporter.log(f"  [WARN] couldn't find `{var_name}` in {file_path.name} - left it untouched")
        reporter.status(var_name, "Warn")
        return
    if new_text == text:
        reporter.log(f"  [OK]     {file_path.name}: {var_name} already correct")
    else:
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            f.write(new_text)
        reporter.log(f"  [UPDATED] {file_path.name}: {var_name} -> {new_value}")
    reporter.status(var_name, "OK")


def patch_paths(pog_dir: Path, models_dir: Path, whisper_cli: Path | None,
                 reporter: Reporter) -> None:
    reporter.section("Pointing the scripts at this machine's Pog_Engine folder")

    organize_py = pog_dir / "OrganizeVODAndFixSRT_Emotion.py"
    analyze_py = pog_dir / "analyze_highlights_emotion.py"

    if whisper_cli is not None:
        if organize_py.is_file():
            patch_raw_string_constant(organize_py, "WHISPER_CLI", str(whisper_cli), False, reporter)
    else:
        # Falls back to where download_whisper_cpp_cublas() extracts the exe
        # (models\Release\whisper-cli.exe) so the patched constant lands on a
        # path the installer can actually populate. Once you run Start Setup
        # again with the exe present, find_whisper_cli() locates it via rglob
        # and patch_paths rewrites the constant to that discovered path.
        guess = models_dir / "Release" / "whisper-cli.exe"
        reporter.log(f"  [WARN] whisper-cli.exe not found - guessing WHISPER_CLI = {guess}")
        reporter.log("         (run Start Setup again after the cublas download places it there)")
        if organize_py.is_file():
            patch_raw_string_constant(organize_py, "WHISPER_CLI", str(guess), False, reporter)
        reporter.status("WHISPER_CLI", "Warn")

    if organize_py.is_file():
        patch_raw_string_constant(organize_py, "WHISPER_MODEL", str(models_dir / "ggml-large-v3.bin"), False, reporter)
        patch_raw_string_constant(organize_py, "WHISPER_VAD", str(models_dir / "ggml-silero-v6.2.0.bin"), False, reporter)

        gallery_dir = pog_dir / "gallery" / "best of"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        patch_raw_string_constant(organize_py, "GALLERY_DIR", str(gallery_dir), True, reporter)
        reporter.log(f"  [INFO]   GALLERY_DIR defaulted to {gallery_dir} (cosmetic only - change")
        reporter.log("           it by hand in OrganizeVODAndFixSRT_Emotion.py if you'd rather")
        reporter.log("           point it at an existing folder of images.)")

    if analyze_py.is_file():
        patch_raw_string_constant(analyze_py, "EMOTION_LOCAL_MODEL_DIR", str(models_dir), False, reporter)
        patch_raw_string_constant(
            analyze_py, "EMOTION_LOCAL_MODEL_FILE",
            str(models_dir / "speech-emotion-recognition-with-openai-whisper-large-v3.safetensors"),
            False, reporter,
        )

    isolate_vocals_py = pog_dir / "isolate_vocals.py"
    if isolate_vocals_py.is_file():
        patch_raw_string_constant(isolate_vocals_py, "TORCH_CACHE_DIR", str(models_dir / "torch_cache"), False, reporter)
        patch_raw_string_constant(isolate_vocals_py, "HF_CACHE_DIR", str(models_dir / "hf_cache"), False, reporter)


# ============================================================================
# 3. Python package installation (GPU-aware, skip-if-present)
# ============================================================================

def pip_install(reporter: Reporter, *args: str, upgrade: bool = False, force: bool = False) -> bool:
    """Streams pip's own output line-by-line through the reporter (so a
    multi-GB CUDA torch download shows real progress in the GUI log, not
    just a frozen-looking wait) instead of letting it print straight to
    whatever console happens to be attached."""
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    if upgrade:
        cmd.append("--upgrade")
    if force:
        cmd.append("--force-reinstall")
    cmd.extend(args)
    reporter.log(f"    $ {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        reporter.log("    " + line.rstrip("\n"))
    return process.wait() == 0


def detect_cuda_driver_version() -> tuple[int, int] | None:
    """Max CUDA version the installed NVIDIA driver supports, or None if
    nvidia-smi isn't found (no NVIDIA GPU / driver not installed)."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    return (int(match.group(1)), int(match.group(2))) if match else None


def torch_status() -> dict | None:
    """Checks torch in a fresh subprocess (not this process) so a package we
    just pip-installed can't be shadowed by stale import state. Returns None
    if torch isn't importable at all."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch, json; print(json.dumps({'version': torch.__version__, 'cuda': torch.cuda.is_available()}))"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def ensure_torch(reporter: Reporter) -> None:
    reporter.log("  torch (PyTorch - needed for the speech-emotion model)")
    reporter.status("torch", "Checking...")
    driver_version = detect_cuda_driver_version()
    status = torch_status()

    if driver_version:
        reporter.log(f"    NVIDIA GPU detected - driver supports up to CUDA {driver_version[0]}.{driver_version[1]}")
    else:
        reporter.log("    No NVIDIA GPU detected (nvidia-smi not found) - using CPU-only PyTorch.")

    if status is not None:
        if driver_version is None or status["cuda"]:
            reporter.log(f"    [OK]     already installed: torch {status['version']} "
                         f"(CUDA available: {status['cuda']}) - skipping.")
            reporter.status("torch", "Already installed")
            return
        reporter.log(f"    torch {status['version']} is installed but CPU-only, and a GPU was "
                     f"detected - upgrading to a CUDA build...")

    reporter.status("torch", "Installing...")
    if driver_version is None:
        pip_install(reporter, "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    else:
        candidates = [tag for (maj, minr, tag) in CUDA_WHEEL_TAGS if (maj, minr) <= driver_version]
        candidates = candidates or ["cu118"]
        for tag in candidates:
            reporter.log(f"    trying PyTorch CUDA build: {tag}")
            if pip_install(reporter, "torch", "--index-url", f"https://download.pytorch.org/whl/{tag}",
                            upgrade=True, force=True):
                break
        else:
            reporter.log("    [WARN] all CUDA wheel attempts failed - falling back to CPU-only PyTorch.")
            pip_install(reporter, "torch", "--index-url", "https://download.pytorch.org/whl/cpu")

    final = torch_status()
    if final is None:
        reporter.log("    [WARN] torch still isn't importable after install - check the log above.")
        reporter.status("torch", "Failed")
    else:
        reporter.log(f"    now installed: torch {final['version']} (CUDA available: {final['cuda']})")
        reporter.status("torch", "Done" if final["cuda"] or driver_version is None else "Done (CPU only)")


def ensure_demucs(reporter: Reporter) -> None:
    """demucs (vocal isolation for single-track/Twitch-style VODs - see
    isolate_vocals.py). Installed after ensure_torch() so it picks up the
    CUDA build already installed there rather than pip resolving its own;
    demucs's separate CLI shells out to ffmpeg/ffprobe directly for audio
    I/O (see demucs/audio.py's AudioFile), not torchaudio, so there's no
    separate CUDA-wheel-matching needed the way there is for torch itself -
    see check_ffmpeg() for the ffmpeg/ffprobe check this depends on.
    """
    reporter.log("  demucs (vocal isolation, for single-track/Twitch-style VODs)")
    reporter.status("demucs", "Checking...")

    if is_importable("demucs"):
        reporter.log("    [OK]     demucs already installed - skipping.")
        reporter.status("demucs", "Already installed")
        return

    reporter.status("demucs", "Installing...")
    reporter.log("    installing demucs ...")
    if pip_install(reporter, "demucs"):
        reporter.log("    [OK]     demucs installed.")
        reporter.status("demucs", "Done")
    else:
        reporter.log("    [WARN] demucs failed to install - check the log above.")
        reporter.status("demucs", "Failed")


def _demucs_repo_dir(hub_root: Path) -> Path | None:
    """Returns the HF-Hub repo dir for adefossez/HTDemucs if present under
    hub_root, else None. Demucs >=4 fetches the htdemucs weights from this
    single HF repo (models--adefossez--HTDemucs), so its existence is a
    reliable 'already downloaded' signal - unlike matching by file extension,
    which would falsely match any other .safetensors/.bin in the cache
    (Ollama GGUF pulls, transformers weights, etc.).

    Older Demucs fetched a torch-hub bundle (htdemucs.th) instead, so also
    accept a .th under hub_root/torch/hub/checkpoints/ for back-compat.
    """
    hf_repo = hub_root / "huggingface" / "hub" / "models--adefossez--HTDemucs"
    if (hf_repo / "refs" / "main").exists():
        return hf_repo
    # Demucs' default htdemucs is HF-only on modern installs, but the original
    # torch-hub bundle (htdemucs.th) still works if a user has it from before.
    legacy = hub_root / "torch" / "hub" / "checkpoints"
    if legacy.exists():
        for f in legacy.iterdir():
            if f.suffix == ".th" and "htdemucs" in f.name.lower():
                return f
    return None


def _migrate_default_hf_htdemucs(dst_hub_root: Path, reporter: Reporter) -> bool:
    """If a previous broken run left the HTDemucs weights in the user's
    default HF cache (~/.cache/huggingface) instead of the redirect - which
    happened whenever HF_HOME silently failed to override the default cache on
    older huggingface_hub - move that repo dir into dst_hub_root/huggingface/hub
    so future installs and runs all read from one place, as originally
    intended (see the comment on TORCH_CACHE_DIR in isolate_vocals.py).

    Moves (not copies) to avoid duplicating ~80MB. Returns True if a
    migration happened, False if nothing was found to move (the usual case
    on a clean install). A leftover empty/partial models--adefossez--HTDemucs
    dir without refs/main is treated as 'not found' - predownload will
    re-download it cleanly rather than trust a truncated cache.
    """
    src = _demucs_repo_dir(Path.home() / ".cache")
    if src is None:
        return False
    try:
        dst = dst_hub_root / "huggingface" / "hub" / "models--adefossez--HTDemucs"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        reporter.log(f"    [OK]     moved existing HTDemucs cache from default location to {dst}")
        return True
    except (OSError, shutil.Error) as exc:
        reporter.log(f"    [WARN] could not move existing HTDemucs cache into the redirect ({exc}).")
        reporter.log(f"           The model stays usable from its original location; only the")
        reporter.log(f"           redirect layout remains empty. Re-run after closing any program")
        reporter.log(f"           that may hold the cache open (e.g. another demucs process).")
        return False


def predownload_demucs_model(models_dir: Path, reporter: Reporter) -> None:
    """Triggers Demucs' one-time ~80MB separation-model download now, during
    setup, instead of leaving it for whenever the first single-track/
    Twitch-style VOD gets processed. Runs demucs against a 1-second silent
    wav purely to make it load (and therefore download/cache) the model -
    the separation result itself is discarded.

    Downloads into models_dir/torch_cache (TORCH_HUB_CACHE + TORCH_HOME) and
    models_dir/hf_cache (HF_HUB_CACHE + HF_HOME), matching TORCH_CACHE_DIR in
    isolate_vocals.py (patched to the same path in patch_paths() above), so
    the model ends up in the same place isolate_vocals.py will look for it
    later instead of two different caches existing side by side.

    Two bugs this replaces:
      1) The old presence check matched ANY .safetensors/.bin/.th in any
         cache dir, so unrelated repos (Ollama GGUF pulls, transformers
         weights) in the default HF cache made it falsely report 'already
         downloaded' and skip - even when HTDemucs itself was absent.
      2) HF_HOME alone doesn't reliably redirect huggingface_hub's snapshot
         dir on newer versions (HF_HUB_CACHE takes precedence), so the
         download silently wrote to ~/.cache/huggingface instead of the
         redirect and the success log lied about the location. Setting
         HF_HUB_CACHE explicitly fixes the redirect for real.
    """
    reporter.log("  demucs separation model (~80MB, one-time download)")
    reporter.status("demucs model", "Checking...")

    if not is_importable("demucs"):
        reporter.log("    [WARN] demucs isn't installed - skipping model pre-download.")
        reporter.log("           It will be downloaded automatically the first time a single-track")
        reporter.log("           VOD is processed instead (needs internet then).")
        reporter.status("demucs model", "Skipped")
        return

    torch_cache_dir = models_dir / "torch_cache"
    hf_cache_dir = models_dir / "hf_cache"
    hf_hub_cache_dir = hf_cache_dir / "huggingface" / "hub"
    default_hub_root = Path.home() / ".cache"

    # Presence check is now anchored on the *actual* repo, not on a loose
    # file extension that any other model in the cache would satisfy.
    already_in_redirect = _demucs_repo_dir(hf_cache_dir) is not None
    already_in_default = _demucs_repo_dir(default_hub_root) is not None

    if already_in_redirect:
        reporter.log(f"    [OK]     already cached in redirect ({hf_cache_dir}) - skipping.")
        reporter.status("demucs model", "Already downloaded")
        return

    if already_in_default:
        reporter.log("    Found HTDemucs in the default user cache (left there by a previous")
        reporter.log("    install whose HF redirect didn't take). Moving it into the redirect...")
        if _migrate_default_hf_htdemucs(hf_cache_dir, reporter):
            reporter.status("demucs model", "Already downloaded")
            return
        # Migration failed (e.g. cache locked) - leave it; the pipeline still
        # works from the default location at runtime. Fall through to attempt
        # a fresh download into the redirect anyway, since the redirect being
        # empty means a future clean run is still wrong.

    reporter.status("demucs model", "Downloading...")
    torch_cache_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_cache_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pog_demucs_predownload_"))
    try:
        silent_wav = tmp_dir / "silence.wav"
        with wave.open(str(silent_wav), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(b"\x00\x00" * 8000)  # 1 second of silence, just enough to make demucs load the model

        env = os.environ.copy()
        env["TORCH_HOME"] = str(torch_cache_dir)
        env["TORCH_HUB_CACHE"] = str(torch_cache_dir / "hub")  # torch >=2.4 honors this for the checkpoints dir
        env["HF_HOME"] = str(hf_cache_dir)
        env["HF_HUB_CACHE"] = str(hf_hub_cache_dir)  # huggingface_hub honors this over HF_HOME
        cmd = [
            sys.executable, "-m", "demucs",
            "-n", "htdemucs", "--two-stems", "vocals",
            "--device", "cpu",
            "-o", str(tmp_dir / "out"),
            str(silent_wav),
        ]
        reporter.log(f"    $ {' '.join(cmd)}")
        reporter.log(f"    (TORCH_HUB_CACHE={env['TORCH_HUB_CACHE']}  HF_HUB_CACHE={env['HF_HUB_CACHE']})")
        process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            reporter.log("    " + line.rstrip("\n"))
        return_code = process.wait()

        if return_code == 0 and _demucs_repo_dir(hf_cache_dir) is not None:
            reporter.log(f"    [OK]     separation model downloaded and cached in {hf_cache_dir}")
            reporter.status("demucs model", "Downloaded")
        elif return_code == 0:
            # Demucs exited clean but, heartbreakingly, didn't land in the
            # redirect we just set. This is the regression the env-var fix
            # above is meant to prevent; if it still happens, demucs likely
            # resolved the repo from elsewhere (HF_HUB_CACHE not honored by
            # this huggingface_hub build). Surface it as a hard failure
            # rather than silently lie, then check the default cache so the
            # user at least knows where it ended up.
            if _demucs_repo_dir(default_hub_root) is not None:
                reporter.log(f"    [WARN] demucs exited OK but HTDemucs is NOT in the redirect")
                reporter.log(f"           ({hf_cache_dir}); a newer huggingface_hub ignored HF_HUB_CACHE.")
                reporter.log(f"           Found it in the default cache instead - the pipeline still")
                reporter.log(f"           works at runtime, but the redirect stayed empty.")
            else:
                reporter.log(f"    [WARN] demucs exited OK but no HTDemucs cache was created anywhere")
                reporter.log(f"           we can find. The pipeline will re-download on first use.")
            reporter.status("demucs model", "Failed")
        else:
            reporter.log(f"    [WARN] model pre-download exited with code {return_code}. It will be")
            reporter.log("           retried automatically the first time a single-track VOD actually")
            reporter.log("           needs it (needs internet then).")
            reporter.status("demucs model", "Failed")
    except Exception as exc:
        reporter.log(f"    [WARN] model pre-download failed: {exc}")
        reporter.log("           It will be retried automatically the first time a single-track VOD")
        reporter.log("           actually needs it (needs internet then).")
        reporter.status("demucs model", "Failed")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def ensure_simple_packages(reporter: Reporter) -> None:
    for module_name, pip_name in SIMPLE_PACKAGES.items():
        reporter.status(pip_name, "Checking...")
        if is_importable(module_name):
            reporter.log(f"  [OK]     {pip_name} already installed - skipping.")
            reporter.status(pip_name, "Already installed")
        else:
            reporter.log(f"  installing {pip_name} ...")
            reporter.status(pip_name, "Installing...")
            if pip_install(reporter, pip_name):
                reporter.status(pip_name, "Done")
            else:
                reporter.log(f"    [WARN] {pip_name} failed to install - check the log above.")
                reporter.status(pip_name, "Failed")


def install_dependencies(reporter: Reporter, models_dir: Path) -> None:
    reporter.section("Installing Python packages (skipping anything already present)")
    ensure_torch(reporter)
    ensure_demucs(reporter)
    predownload_demucs_model(models_dir, reporter)
    ensure_simple_packages(reporter)


# ============================================================================
# 4. Ollama + model check (verify only - installing Ollama itself is on you)
# ============================================================================

def check_ollama(pog_dir: Path, reporter: Reporter) -> bool:
    reporter.section("Checking Ollama + models")
    all_ok = True

    model_name = judge_model_name = None
    try:
        sys.path.insert(0, str(pog_dir))
        import pipeline_config as cfg  # noqa: F401  (local import, path set above)
        model_name = getattr(cfg, "MODEL", None)
        judge_model_name = getattr(cfg, "JUDGE_MODEL", None)
        ollama_url = getattr(cfg, "OLLAMA_URL", "http://localhost:11434/api/generate")
    except Exception as exc:
        reporter.log(f"  [WARN] couldn't read pipeline_config.py for model names: {exc}")
        ollama_url = "http://localhost:11434/api/generate"

    reporter.status("ollama", "Checking...")
    ollama_found = shutil.which("ollama") is not None
    reporter.log(f"  {mark(ollama_found)} ollama command available on PATH")
    reporter.status("ollama", "OK" if ollama_found else "Missing")
    if not ollama_found:
        reporter.log("      -> install it yourself from https://ollama.com/download, pull the")
        reporter.log("         models pipeline_config.py expects, then re-run this installer.")
        all_ok = False
    else:
        try:
            listing = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
            pulled_text = listing.stdout
            for name in filter(None, [model_name, judge_model_name]):
                reporter.add_row("ollama", name, f"model: {name}")
                got = name in pulled_text
                reporter.log(f"  {mark(got)} model pulled: {name}")
                reporter.status(name, "OK" if got else "Missing")
                if not got:
                    reporter.log(f"      -> run: ollama pull {name}")
                    all_ok = False
        except Exception as exc:
            reporter.log(f"  [WARN] couldn't run `ollama list`: {exc}")
            all_ok = False

    reporter.status("Ollama server", "Checking...")
    host_root = ollama_url.split("/api/")[0] if ollama_url else "http://localhost:11434"
    try:
        urllib.request.urlopen(host_root, timeout=3)
        reporter.log(f"  [OK]     Ollama server responding at {host_root}")
        reporter.status("Ollama server", "OK")
    except Exception:
        reporter.log(f"  [INFO]   Ollama server isn't responding at {host_root} right now.")
        reporter.log("           That's fine if it's just not running yet - the Ollama app or")
        reporter.log("           `ollama serve` starts it.")
        reporter.status("Ollama server", "Not running")
        all_ok = False

    return all_ok


def check_config_paths(pog_dir: Path, models_dir: Path, whisper_cli: Path | None,
                        reporter: Reporter) -> bool:
    """Check if all machine-specific paths in scripts are correctly patched."""
    reporter.section("Checking configuration (machine-specific paths)")
    all_ok = True
    
    organize_py = pog_dir / "OrganizeVODAndFixSRT_Emotion.py"
    analyze_py = pog_dir / "analyze_highlights_emotion.py"
    isolate_vocals_py = pog_dir / "isolate_vocals.py"
    
    # Resolve expected paths to absolute
    if whisper_cli is not None:
        expected_whisper_cli = str(whisper_cli.resolve())
    else:
        # Matches where download_whisper_cpp_cublas() actually extracts the
        # exe (models\Release\whisper-cli.exe, see download_whisper_cpp_cublas
        # and check_models). The old guess pointed at a side folder named
        # "whisper cublas 12.4.0\Release\" - the installer never writes there,
        # so the constant could never match even after a successful download.
        expected_whisper_cli = str((models_dir / "Release" / "whisper-cli.exe").resolve())
    expected_whisper_model = str((models_dir / "ggml-large-v3.bin").resolve())
    expected_whisper_vad = str((models_dir / "ggml-silero-v6.2.0.bin").resolve())
    expected_gallery_dir = str((pog_dir / "gallery" / "best of").resolve())
    expected_emotion_model_dir = str(models_dir.resolve())
    expected_emotion_model_file = str((models_dir / "speech-emotion-recognition-with-openai-whisper-large-v3.safetensors").resolve())
    expected_torch_cache = str((models_dir / "torch_cache").resolve())
    
    def check_var(content: str, var_name: str, expected: str) -> bool:
        # Check for both raw string and regular string formats
        patterns = [
            f'{var_name} = r"{expected}"',
            f'{var_name} = "{expected}"',
            f'{var_name} = Path(r"{expected}")',
        ]
        return any(p in content for p in patterns)
    
    if organize_py.is_file():
        content = organize_py.read_text(encoding="utf-8")
        
        for var_name, expected in [
            ("WHISPER_CLI", expected_whisper_cli),
            ("WHISPER_MODEL", expected_whisper_model),
            ("WHISPER_VAD", expected_whisper_vad),
            ("GALLERY_DIR", expected_gallery_dir),
        ]:
            if check_var(content, var_name, expected):
                reporter.log(f"  [OK]     {organize_py.name}: {var_name} already correct")
                reporter.status(var_name, "OK")
            else:
                reporter.log(f"  [WARN] {organize_py.name}: {var_name} needs patching")
                reporter.status(var_name, "Warn")
                all_ok = False
    
    if analyze_py.is_file():
        content = analyze_py.read_text(encoding="utf-8")
        
        for var_name, expected in [
            ("EMOTION_LOCAL_MODEL_DIR", expected_emotion_model_dir),
            ("EMOTION_LOCAL_MODEL_FILE", expected_emotion_model_file),
        ]:
            if check_var(content, var_name, expected):
                reporter.log(f"  [OK]     {analyze_py.name}: {var_name} already correct")
                reporter.status(var_name, "OK")
            else:
                reporter.log(f"  [WARN] {analyze_py.name}: {var_name} needs patching")
                reporter.status(var_name, "Warn")
                all_ok = False
    
    if isolate_vocals_py.is_file():
        content = isolate_vocals_py.read_text(encoding="utf-8")
        
        if check_var(content, "TORCH_CACHE_DIR", expected_torch_cache):
            reporter.log(f"  [OK]     {isolate_vocals_py.name}: TORCH_CACHE_DIR already correct")
            reporter.status("TORCH_CACHE_DIR", "OK")
        else:
            reporter.log(f"  [WARN] {isolate_vocals_py.name}: TORCH_CACHE_DIR needs patching")
            reporter.status("TORCH_CACHE_DIR", "Warn")
            all_ok = False
    
    return all_ok


def _verify_python_packages(models_dir: Path, reporter: Reporter,
                            installed_label: str = "installed") -> bool:
    """Re-import every required package and report its real status.

    The old install branch of check_python_packages returned True
    unconditionally after install_dependencies(), so the Summary could claim
    "everything ready" - and create the drag-and-drop shortcut - even when
    torch or demucs had silently failed to install. Both install and
    check-only modes now route through this single verifier so they can never
    disagree about what's actually importable. `installed_label` only changes
    the OK status word shown in the GUI ("OK" vs "Already installed").
    """
    all_ok = True

    reporter.log("  torch (PyTorch - needed for the speech-emotion model)")
    reporter.status("torch", "Checking...")
    if is_importable("torch"):
        import torch
        cuda_available = torch.cuda.is_available()
        reporter.log(f"    [OK]     {installed_label}: torch {torch.__version__} (CUDA available: {cuda_available})")
        reporter.status("torch", "Already installed" if installed_label.startswith("already") else "OK")
    else:
        reporter.log("    [MISSING] torch not installed")
        reporter.status("torch", "Missing")
        all_ok = False

    reporter.log("  demucs (vocal isolation, for single-track/Twitch-style VODs)")
    reporter.status("demucs", "Checking...")
    if is_importable("demucs"):
        reporter.log(f"    [OK]     demucs {installed_label}")
        reporter.status("demucs", "Already installed" if installed_label.startswith("already") else "OK")
    else:
        reporter.log("    [MISSING] demucs not installed")
        reporter.status("demucs", "Missing")
        all_ok = False

    predownload_demucs_model(models_dir, reporter)  # own check/reporting

    for module_name, pip_name in SIMPLE_PACKAGES.items():
        reporter.status(pip_name, "Checking...")
        if is_importable(module_name):
            reporter.log(f"  [OK]     {pip_name} {installed_label}")
            reporter.status(pip_name, "Already installed" if installed_label.startswith("already") else "OK")
        else:
            reporter.log(f"  [MISSING] {pip_name} not installed")
            reporter.status(pip_name, "Missing")
            all_ok = False

    return all_ok


def check_python_packages(models_dir: Path, reporter: Reporter, install: bool) -> bool:
    """Install packages if install=True, then verify what's actually importable.

    Verification runs in BOTH modes (previously install-mode returned True
    unconditionally, masking failed installs). The install branch first runs
    install_dependencies() - which emits its own "Installing..." section and
    reports per-package failures - then calls _verify_python_packages() under
    a "Verifying Python packages" section so the final per-row status reflects
    the on-disk truth, not pip's exit code.
    """
    if install:
        install_dependencies(reporter, models_dir)
        reporter.section("Verifying Python packages")
        return _verify_python_packages(models_dir, reporter, installed_label="installed")

    reporter.section("Checking Python packages")
    return _verify_python_packages(models_dir, reporter, installed_label="already installed")

# ============================================================================
# Drag-and-drop shortcut
# ============================================================================

def create_drag_shortcut(pog_dir: Path, reporter: Reporter) -> None:
    """Create a Windows shortcut to OrganizeVODAndFixSRT_Emotion.bat named
    "Drag MP4 on me.lnk" inside the Pog folder.

    The shortcut is the user-facing entry point: copy it into a VOD folder
    and drop any .mkv/mp4 onto it. Windows passes the dropped file as %1 to
    the target .bat, which runs the full pipeline against that VOD. The .bat
    uses %~dp0 (its own dir) to locate the .py alongside it, so the shortcut
    works from anywhere - %~dp0 still resolves to the real Pog folder, not
    the folder the .lnk was copied into.

    Builds the .lnk via PowerShell New-Object -ComObject WScript.Shell, which
    ships with every Windows install and avoids adding a pywin32 dependency.
    Idempotent: re-running Start Setup overwrites the .lnk in place, so moving
    the Pog folder and re-running Setup rewrites TargetPath to the new home.
    """
    reporter.section("Creating drag-and-drop shortcut")
    bat = pog_dir / "OrganizeVODAndFixSRT_Emotion.bat"
    if not bat.is_file():
        reporter.log(f"  [WARN] OrganizeVODAndFixSRT_Emotion.bat not found in {pog_dir} - skipping shortcut")
        reporter.status("Drag MP4 on me", "Warn")
        return

    lnk = pog_dir / "Drag MP4 on me.lnk"
    # TargetPath/WorkingDirectory/Shortcut full path are absolute Windows paths
    # inlined straight into the PowerShell string. WScript.Shell.CreateShortcut
    # takes a single string (the .lnk path); we pass it literally to avoid
    # PowerShell's space-sensitive parsing of Join-Path inside a method call.
    ps = (
        f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}') ;"
        f"$s.TargetPath = '{bat}' ;"
        f"$s.WorkingDirectory = '{pog_dir}' ;"
        f"$s.Description = 'Drag an .mp4/.mkv onto me to run Pog_Engine' ;"
        f"$s.IconLocation = '{bat},0' ;"
        "$s.Save()"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True, text=True,
    )
    if proc.returncode == 0 and lnk.is_file():
        reporter.log(f"  [OK]     created {lnk.name} in the Pog folder")
        reporter.log("           Copy it into your VOD folder and drag any video onto it.")
        reporter.status("Drag MP4 on me", "OK")
    else:
        reporter.log(f"  [WARN] shortcut creation failed (exit {proc.returncode})")
        if proc.stderr.strip():
            reporter.log(f"           powershell: {proc.stderr.strip()}")
        reporter.log("           You can still run the pipeline by dragging a video onto")
        reporter.log(f"           {bat.name} directly.")
        reporter.status("Drag MP4 on me", "Warn")


def run_all_checks(pog_dir: Path, reporter: Reporter, install: bool = True) -> bool:
    reporter.log(f"Pog_Engine folder: {pog_dir}")
    reporter.log(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        reporter.log("[WARN] PyTorch and some other dependencies need Python 3.10+.")
        reporter.log("       Consider installing a newer Python and re-running this installer.")

    missing_scripts = check_scripts(pog_dir, reporter)
    models_dir, missing_models, whisper_cli = check_models(pog_dir, reporter)
    
    if install:
        download_all_models(models_dir, reporter)
        # Re-check models after downloads so summary is accurate
        _, missing_models, whisper_cli = check_models(pog_dir, reporter)
    
    ffmpeg_ok = check_ffmpeg(reporter)

    # Repoint the machine-specific path constants in the three pipeline scripts
    # to THIS machine's pog_dir / models_dir. check_config_paths below then
    # verifies the patched result. Previously patch_paths() was defined but
    # never called - so Start Setup downloaded models and installed packages
    # but left WHISPER_CLI / WHISPER_MODEL / GALLERY_DIR / etc. pointing at
    # wherever the repo was copied from, and the config check warned forever.
    # Skipped in check-only mode so a read-only Re-check never rewrites source.
    if install:
        patch_paths(pog_dir, models_dir, whisper_cli, reporter)

    config_ok = check_config_paths(pog_dir, models_dir, whisper_cli, reporter)

    packages_ok = check_python_packages(models_dir, reporter, install)

    ollama_ok = check_ollama(pog_dir, reporter)

    reporter.section("Summary")
    all_good = (not missing_scripts and not missing_models and
                whisper_cli is not None and ffmpeg_ok and
                config_ok and packages_ok and ollama_ok)
    if all_good:
        reporter.log("  Everything needed is in place. You're ready to run the pipeline.")
        if install:
            # Drop a drag-and-drop shortcut in the Pog folder so the user never
            # has to touch code again: copy "Drag MP4 on me.lnk" into a VOD
            # folder and drop any .mp4 on it. The shortcut targets the real
            # OrganizeVODAndFixSRT_Emotion.bat, so %~dp0 inside it still resolves
            # to this Pog folder and %~1 receives the dropped file. Only created
            # on a fully-green install, never during check-only or a failed run.
            create_drag_shortcut(pog_dir, reporter)
    else:
        reporter.log("  Still needed before the pipeline will run end-to-end:")
        for name in missing_scripts:
            reporter.log(f"    - {name} (copy it into {pog_dir})")
        for name in missing_models:
            reporter.log(f"    - models\\{name}")
        if whisper_cli is None:
            reporter.log("    - whisper.cpp CUDA/cublas build (whisper-cli.exe in models\\Release\\)")
        if not ffmpeg_ok:
            reporter.log("    - ffmpeg/ffprobe on PATH")
        if not config_ok:
            reporter.log("    - configuration paths need patching (run Start Setup)")
        if not packages_ok:
            reporter.log("    - Python packages missing (run Start Setup)")
        if not ollama_ok:
            reporter.log("    - Ollama models/server not ready (run Start Setup)")
    reporter.log("")
    reporter.log("  Re-run this installer any time after adding files - it only")
    reporter.log("  installs/fixes what's still missing.")
    return all_good


# ============================================================================
# --cli mode
# ============================================================================

def run_cli(pog_dir_str: str) -> int:
    pog_dir = Path(pog_dir_str).expanduser().resolve()
    if not pog_dir.is_dir():
        print(f"ERROR: '{pog_dir}' is not a folder that exists.")
        return 1
    run_all_checks(pog_dir, Reporter())
    return 0


# ============================================================================
# GUI mode (default)
# ============================================================================

def run_gui(default_dir_str: str) -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Pog_Engine Installer")
    root.geometry("880x760")
    root.configure(bg="#121212")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background="#121212", foreground="#f2f2f2", fieldbackground="#1e1e1e")
    style.configure("TFrame", background="#121212")
    style.configure("TLabelframe", background="#121212", foreground="#f2f2f2")
    style.configure("TLabelframe.Label", background="#121212", foreground="#f2f2f2", font=("Segoe UI", 10, "bold"))
    style.configure("TLabel", background="#121212", foreground="#f2f2f2")
    style.configure("TButton", background="#242424", foreground="#f2f2f2", bordercolor="#3a3a3a")
    style.map("TButton", background=[("active", "#333333")], foreground=[("disabled", "#666666")])
    style.configure("TEntry", fieldbackground="#1e1e1e", foreground="#f2f2f2")

    STATUS_COLORS = {
        "Waiting": "#777777",
        "Checking...": "#f2f2f2",
        "Installing...": "#ff8c00",
        "OK": "#4caf50",
        "Done": "#4caf50",
        "Done (CPU only)": "#8bc34a",
        "Already installed": "#4caf50",
        "Already downloaded": "#4caf50",
        "Missing": "#e05252",
        "Failed": "#e05252",
        "Warn": "#e0a852",
        "Not running": "#e0a852",
    }

    # ---- top: folder picker -------------------------------------------------
    top_frame = ttk.Frame(root, padding=(12, 12, 12, 6))
    top_frame.pack(fill="x")
    ttk.Label(top_frame, text="Pog_Engine Installer", font=("Segoe UI", 15, "bold")).pack(anchor="w")
    ttk.Label(top_frame, text="Checks your folder, points the scripts at it, and installs "
                              "the Python packages the pipeline needs.").pack(anchor="w", pady=(2, 8))

    path_row = ttk.Frame(top_frame)
    path_row.pack(fill="x")
    path_row.columnconfigure(0, weight=1)
    folder_var = tk.StringVar(value=default_dir_str)
    folder_entry = ttk.Entry(path_row, textvariable=folder_var)
    folder_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))

    def browse_folder() -> None:
        chosen = filedialog.askdirectory(initialdir=folder_var.get() or ".", title="Select your Pog_Engine folder")
        if chosen:
            folder_var.set(chosen)

    browse_button = ttk.Button(path_row, text="Browse...", command=browse_folder)
    browse_button.grid(row=0, column=1)
    start_button = ttk.Button(path_row, text="Start Setup")
    start_button.grid(row=0, column=2, padx=(6, 0))

    status_var = tk.StringVar(value="Ready to start.")
    ttk.Label(top_frame, textvariable=status_var).pack(anchor="w", pady=(8, 0))

    # ---- middle: scrollable checklist --------------------------------------
    middle_container = ttk.Frame(root, padding=(12, 6))
    middle_container.pack(fill="both", expand=True)
    canvas = tk.Canvas(middle_container, bg="#121212", highlightthickness=0)
    scrollbar = ttk.Scrollbar(middle_container, orient="vertical", command=canvas.yview)
    checklist_frame = ttk.Frame(canvas)
    checklist_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=checklist_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def on_mousewheel(event) -> None:
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", on_mousewheel)

    row_widgets: dict[str, tk.StringVar] = {}
    section_frames: dict[str, ttk.LabelFrame] = {}
    section_next_row: dict[str, int] = {}

    def make_section(key: str, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(checklist_frame, text=title, padding=(10, 6))
        frame.pack(fill="x", pady=(0, 8))
        frame.columnconfigure(0, weight=1)
        section_frames[key] = frame
        section_next_row[key] = 0
        return frame

    def add_row_widget(section_key: str, item_key: str, label_text: str) -> None:
        frame = section_frames[section_key]
        row = section_next_row[section_key]
        section_next_row[section_key] = row + 1
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w", pady=2)
        status_string = tk.StringVar(value="Waiting")
        status_label = ttk.Label(frame, textvariable=status_string, width=18, anchor="e",
                                  foreground=STATUS_COLORS["Waiting"])
        status_label.grid(row=row, column=1, sticky="e", pady=2)
        row_widgets[item_key] = status_string
        row_widgets[item_key + "__label"] = status_label  # type: ignore[assignment]

    scripts_frame = make_section("scripts", "Pog_Engine Scripts")
    for name in REQUIRED_SCRIPTS:
        add_row_widget("scripts", name, name)
    add_row_widget("scripts", "pog_engine_setup.py", "pog_engine_setup.py (this script)")

    models_frame = make_section("models", "Model Files (in models\\)")
    for name in REQUIRED_MODEL_FILES:
        add_row_widget("models", name, name)
    add_row_widget("models", "whisper-cli.exe", "whisper-cli.exe (in models\\Release\\)")

    ffmpeg_frame = make_section("ffmpeg", "ffmpeg / ffprobe")
    add_row_widget("ffmpeg", "ffmpeg", "ffmpeg on PATH")
    add_row_widget("ffmpeg", "ffprobe", "ffprobe on PATH (VOD track-count detection)")

    config_frame = make_section("config", "Configuration (machine-specific paths)")
    for var_name in ["WHISPER_CLI", "WHISPER_MODEL", "WHISPER_VAD", "GALLERY_DIR",
                     "EMOTION_LOCAL_MODEL_DIR", "EMOTION_LOCAL_MODEL_FILE", "TORCH_CACHE_DIR"]:
        add_row_widget("config", var_name, var_name)

    packages_frame = make_section("packages", "Python Packages")
    add_row_widget("packages", "torch", "torch (PyTorch, GPU if available)")
    add_row_widget("packages", "demucs", "demucs (vocal isolation, single-track VODs)")
    add_row_widget("packages", "demucs model", "demucs separation model (~80MB)")
    for pip_name in SIMPLE_PACKAGES.values():
        add_row_widget("packages", pip_name, pip_name)

    ollama_frame = make_section("ollama", "Ollama")
    add_row_widget("ollama", "ollama", "ollama command on PATH")
    add_row_widget("ollama", "Ollama server", "Ollama server reachable")

    # ---- bottom: log box -----------------------------------------------------
    bottom_frame = ttk.Frame(root, padding=(12, 6, 12, 12))
    bottom_frame.pack(fill="both", expand=False)
    ttk.Label(bottom_frame, text="Log").pack(anchor="w")
    log_box = tk.Text(bottom_frame, height=10, wrap="word", state="disabled",
                       bg="#0f0f0f", fg="#f2f2f2", insertbackground="#f2f2f2")
    log_box.pack(fill="both", expand=True)

    def append_log(text: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", text + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def set_row_status(key: str, value: str) -> None:
        if key not in row_widgets:
            return
        row_widgets[key].set(value)
        label_widget = row_widgets.get(key + "__label")
        if label_widget is not None:
            label_widget.configure(foreground=STATUS_COLORS.get(value, "#f2f2f2"))  # type: ignore[union-attr]

    # ---- worker thread + queue -------------------------------------------------
    event_queue: queue.Queue[tuple[str, object]] = queue.Queue()

    class GuiReporter(Reporter):
        def log(self, msg: str = "") -> None:
            event_queue.put(("log", msg))

        def status(self, key: str, value: str) -> None:
            event_queue.put(("status", (key, value)))

        def add_row(self, section: str, key: str, label: str) -> None:
            event_queue.put(("add_row", (section, key, label)))

        def section(self, title: str) -> None:
            event_queue.put(("log", ""))
            event_queue.put(("log", "=" * 60))
            event_queue.put(("log", title))
            event_queue.put(("log", "=" * 60))

    def check_only_worker(pog_dir_str: str) -> None:
        try:
            pog_dir = Path(pog_dir_str).expanduser().resolve()
            if not pog_dir.is_dir():
                event_queue.put(("log", f"ERROR: '{pog_dir}' is not a folder that exists."))
                event_queue.put(("check_finished", False))
                return
            ok = run_all_checks(pog_dir, GuiReporter(), install=False)
            event_queue.put(("check_finished", ok))
        except Exception as exc:
            event_queue.put(("log", f"ERROR: unexpected failure: {exc}"))
            event_queue.put(("check_finished", False))

    def worker(pog_dir_str: str) -> None:
        try:
            pog_dir = Path(pog_dir_str).expanduser().resolve()
            if not pog_dir.is_dir():
                event_queue.put(("log", f"ERROR: '{pog_dir}' is not a folder that exists."))
                event_queue.put(("finished", False))
                return
            ok = run_all_checks(pog_dir, GuiReporter(), install=True)
            event_queue.put(("finished", ok))
        except Exception as exc:
            event_queue.put(("log", f"ERROR: unexpected failure: {exc}"))
            event_queue.put(("finished", False))

    def on_finished(all_good: bool) -> None:
        start_button.configure(state="normal", text="Re-check")
        folder_entry.configure(state="normal")
        browse_button.configure(state="normal")
        status_var.set("Finished - everything needed is in place." if all_good
                        else "Finished - see the checklist above for what's still missing.")

    def on_check_finished(all_good: bool) -> None:
        start_button.configure(state="normal", text="Start Setup")
        folder_entry.configure(state="normal")
        browse_button.configure(state="normal")
        status_var.set("Check complete. Click Start Setup to install missing components." if not all_good
                        else "Everything needed is in place. You're ready to run the pipeline.")

    def start_setup() -> None:
        pog_dir_str = folder_var.get().strip()
        if not pog_dir_str:
            messagebox.showerror("Pog_Engine Installer", "Enter or browse to your Pog_Engine folder first.")
            return
        if not Path(pog_dir_str).expanduser().is_dir():
            messagebox.showerror("Pog_Engine Installer", f"'{pog_dir_str}' is not a folder that exists.")
            return

        btn_text = start_button.cget("text")
        if btn_text == "Re-check":
            # Run check-only
            start_button.configure(state="disabled", text="Checking...")
            folder_entry.configure(state="disabled")
            browse_button.configure(state="disabled")
            status_var.set("Re-checking...")
            threading.Thread(target=check_only_worker, args=(pog_dir_str,), daemon=True).start()
        else:
            # Run full install
            start_button.configure(state="disabled", text="Running...")
            folder_entry.configure(state="disabled")
            browse_button.configure(state="disabled")
            status_var.set("Running - this can take a while the first time (PyTorch is a big download).")
            threading.Thread(target=worker, args=(pog_dir_str,), daemon=True).start()

    start_button.configure(command=start_setup)

    def drain_events() -> None:
        if not root.winfo_exists():
            return
        while True:
            try:
                kind, payload = event_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                append_log(str(payload))
            elif kind == "status":
                key, value = payload  # type: ignore[misc]
                set_row_status(key, value)
            elif kind == "add_row":
                section, key, label = payload  # type: ignore[misc]
                if key not in row_widgets:
                    add_row_widget(section, key, label)
            elif kind == "finished":
                on_finished(bool(payload))
            elif kind == "check_finished":
                on_check_finished(bool(payload))
        root.after(100, drain_events)

    # Auto-check on startup
    threading.Thread(target=check_only_worker, args=(default_dir_str,), daemon=True).start()

    root.after(100, drain_events)
    root.mainloop()
    return 0


# ============================================================================
# main
# ============================================================================

def main(argv: list[str]) -> int:
    args = argv[1:]
    cli_mode = "--cli" in args
    args = [a for a in args if a != "--cli"]
    default_dir = args[0] if args else str(Path(__file__).resolve().parent)

    if cli_mode:
        return run_cli(default_dir)

    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("[INFO] tkinter isn't available on this Python install - falling back to console mode.")
        return run_cli(default_dir)

    return run_gui(default_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
