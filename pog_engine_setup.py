"""Setup / verification helper for the Pog_Engine VOD highlight pipeline.

Not meant to be run directly - Install_PogEngine.bat calls this after making
sure some version of Python exists. It:

  1. Checks the Pog_Engine folder for the 5 pipeline scripts, the 5 required
     files in the models folder, and a whisper-cli.exe (whisper.cpp CUDA build).
  2. Patches the machine-specific path constants (WHISPER_CLI, WHISPER_MODEL,
     WHISPER_VAD, GALLERY_DIR, EMOTION_LOCAL_MODEL_DIR/FILE) in the two
     pipeline scripts to point at THIS machine's Pog_Engine folder, instead
     of whatever machine they were last edited on.
  3. Installs the Python packages the pipeline actually imports (requests,
     numpy, torch, transformers, librosa, soundfile, safetensors, Pillow),
     skipping anything already present - and picks a CUDA build of torch
     over a CPU-only one whenever an NVIDIA GPU is detected.
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

import importlib.util
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

REQUIRED_SCRIPTS = [
    "analyze_highlights_emotion.py",
    "OrganizeVODAndFixSRT_Emotion.py",
    "OrganizeVODAndFixSRT_Emotion.bat",
    "pipeline_config.py",
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

TOTAL_SECTIONS = 6  # scripts, models, whisper, configuration, packages, ollama


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


def check_models(pog_dir: Path, reporter: Reporter) -> tuple[Path, list[str]]:
    reporter.section("Checking models folder")
    models_dir = pog_dir / "models"
    if not models_dir.is_dir():
        reporter.log("  [MISSING] models folder does not exist yet")
        for name in REQUIRED_MODEL_FILES:
            reporter.status(name, "Missing")
        return models_dir, list(REQUIRED_MODEL_FILES)

    missing = []
    for name in REQUIRED_MODEL_FILES:
        ok = (models_dir / name).is_file()
        reporter.log(f"  {mark(ok)} models\\{name}")
        reporter.status(name, "OK" if ok else "Missing")
        if not ok:
            missing.append(name)
    return models_dir, missing


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
        # Best-guess default matching the folder name from the setup instructions,
        # so the constant at least points somewhere sensible once you add it.
        guess = pog_dir / "whisper cublas 12.4.0" / "Release" / "whisper-cli.exe"
        reporter.log(f"  [WARN] whisper-cli.exe not found - guessing WHISPER_CLI = {guess}")
        reporter.log("         (fix this by hand later if your folder is named differently)")
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


def install_dependencies(reporter: Reporter) -> None:
    reporter.section("Installing Python packages (skipping anything already present)")
    ensure_torch(reporter)
    ensure_simple_packages(reporter)


# ============================================================================
# 4. Ollama + model check (verify only - installing Ollama itself is on you)
# ============================================================================

def check_ollama(pog_dir: Path, reporter: Reporter) -> None:
    reporter.section("Checking Ollama + models")

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
        except Exception as exc:
            reporter.log(f"  [WARN] couldn't run `ollama list`: {exc}")

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


# ============================================================================
# Orchestration shared by both --cli and the GUI
# ============================================================================

def run_all_checks(pog_dir: Path, reporter: Reporter) -> bool:
    reporter.log(f"Pog_Engine folder: {pog_dir}")
    reporter.log(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        reporter.log("[WARN] PyTorch and some other dependencies need Python 3.10+.")
        reporter.log("       Consider installing a newer Python and re-running this installer.")

    missing_scripts = check_scripts(pog_dir, reporter)
    models_dir, missing_models = check_models(pog_dir, reporter)
    whisper_cli = find_whisper_cli(pog_dir, reporter)

    patch_paths(pog_dir, models_dir, whisper_cli, reporter)
    install_dependencies(reporter)
    check_ollama(pog_dir, reporter)

    reporter.section("Summary")
    all_good = not missing_scripts and not missing_models and whisper_cli is not None
    if all_good:
        reporter.log("  Everything needed is in place. You're ready to run the pipeline.")
    else:
        reporter.log("  Still needed before the pipeline will run end-to-end:")
        for name in missing_scripts:
            reporter.log(f"    - {name} (copy it into {pog_dir})")
        for name in missing_models:
            reporter.log(f"    - models\\{name}")
        if whisper_cli is None:
            reporter.log("    - whisper.cpp CUDA/cublas build (whisper-cli.exe)")
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
    style.configure(
        "Orange.Horizontal.TProgressbar",
        troughcolor="#2a2a2a", background="#ff8c00", lightcolor="#ff8c00",
        darkcolor="#ff8c00", bordercolor="#2a2a2a",
    )

    STATUS_COLORS = {
        "Waiting": "#777777",
        "Checking...": "#f2f2f2",
        "Installing...": "#ff8c00",
        "OK": "#4caf50",
        "Done": "#4caf50",
        "Done (CPU only)": "#8bc34a",
        "Already installed": "#4caf50",
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
    progress_var = tk.DoubleVar(value=0)
    progress = ttk.Progressbar(top_frame, variable=progress_var, maximum=TOTAL_SECTIONS,
                                style="Orange.Horizontal.TProgressbar")
    progress.pack(fill="x", pady=(4, 0))

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

    whisper_frame = make_section("whisper", "Whisper.cpp (CUDA / cublas build)")
    add_row_widget("whisper", "whisper-cli.exe", "whisper-cli.exe")

    config_frame = make_section("config", "Configuration (machine-specific paths)")
    for var_name in ["WHISPER_CLI", "WHISPER_MODEL", "WHISPER_VAD", "GALLERY_DIR",
                     "EMOTION_LOCAL_MODEL_DIR", "EMOTION_LOCAL_MODEL_FILE"]:
        add_row_widget("config", var_name, var_name)

    packages_frame = make_section("packages", "Python Packages")
    add_row_widget("packages", "torch", "torch (PyTorch, GPU if available)")
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
    section_progress = {"value": 0}

    class GuiReporter(Reporter):
        def log(self, msg: str = "") -> None:
            event_queue.put(("log", msg))

        def status(self, key: str, value: str) -> None:
            event_queue.put(("status", (key, value)))

        def add_row(self, section: str, key: str, label: str) -> None:
            event_queue.put(("add_row", (section, key, label)))

        def section(self, title: str) -> None:
            section_progress["value"] += 1
            event_queue.put(("progress", section_progress["value"]))
            event_queue.put(("log", ""))
            event_queue.put(("log", "=" * 60))
            event_queue.put(("log", title))
            event_queue.put(("log", "=" * 60))

    def worker(pog_dir_str: str) -> None:
        try:
            pog_dir = Path(pog_dir_str).expanduser().resolve()
            if not pog_dir.is_dir():
                event_queue.put(("log", f"ERROR: '{pog_dir}' is not a folder that exists."))
                event_queue.put(("finished", False))
                return
            ok = run_all_checks(pog_dir, GuiReporter())
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
            elif kind == "progress":
                progress_var.set(payload)
            elif kind == "finished":
                on_finished(bool(payload))
        root.after(100, drain_events)

    def start_setup() -> None:
        pog_dir_str = folder_var.get().strip()
        if not pog_dir_str:
            messagebox.showerror("Pog_Engine Installer", "Enter or browse to your Pog_Engine folder first.")
            return
        if not Path(pog_dir_str).expanduser().is_dir():
            messagebox.showerror("Pog_Engine Installer", f"'{pog_dir_str}' is not a folder that exists.")
            return
        start_button.configure(state="disabled", text="Running...")
        folder_entry.configure(state="disabled")
        browse_button.configure(state="disabled")
        status_var.set("Running - this can take a while the first time (PyTorch is a big download).")
        progress_var.set(0)
        section_progress["value"] = 0
        threading.Thread(target=worker, args=(pog_dir_str,), daemon=True).start()

    start_button.configure(command=start_setup)

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
