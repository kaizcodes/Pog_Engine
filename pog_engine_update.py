"""Differential updater for an installed Pog Engine.

Users run this (via Update_PogEngine.bat) to bring their install folder up to
the latest public GitHub release. It downloads the release's ``PogEngine.zip``
asset, compares every managed file against the installed copy by SHA-256, and
replaces ONLY the files that actually changed. Untouched files keep their
bytes and timestamps; nothing outside the managed list is ever modified.

Two installed customizations survive an update and are re-applied to the new
files automatically:

1. Config tunings - values the user saved through ConfigurePogEngine
   (EDITABLE_PARAMS defaults rewritten into pipeline_config.py) are carried
   from the old file into the new one via the new file's own
   apply_config_values(). New params keep new defaults; removed ones drop.
2. Machine paths - WHISPER_CLI / WHISPER_MODEL / WHISPER_VAD / GALLERY_DIR /
   EMOTION_LOCAL_MODEL_DIR / _FILE / TORCH_CACHE_DIR / HF_CACHE_DIR point at
   THIS machine's folders (rewritten by the installer). The old values are
   read from the backup and written into the new files, so a non-default
   install location keeps working without re-running setup.

Replaced originals are kept in update_backup_<tag>_<timestamp>/ next to the
scripts, so any update can be undone by copying the backup back.

Release contract (maintainer side): every GitHub release on
kaizcodes/Pog_Engine must attach a ``PogEngine.zip`` asset holding the
managed files below at the zip root (a single top-level folder is also
accepted), and bump POG_ENGINE_VERSION in pipeline_config.py to the release
number (tag = "v" + version, e.g. v2.0.0).

Usage:
    python pog_engine_update.py [--check-only] [--ref TAG] [--yes] [--force]

    --check-only  print installed vs latest-release versions and exit.
    --ref TAG     update to a specific release tag instead of latest
                  (e.g. --ref v2.0.1). Version-direction check is skipped.
    --yes         apply without asking (for scripts).
    --force       re-apply even when the versions already match.

Network: stdlib urllib only, so this runs on a bare Python with no
third-party packages. GITHUB_TOKEN env is honored to raise the API rate
limit (60 req/h anonymous). Exit 0 on success / already-up-to-date,
1 on any failure.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

GITHUB_REPO = os.environ.get("POG_ENGINE_REPO", "kaizcodes/Pog_Engine")
API_BASE = "https://api.github.com/repos"
USER_AGENT = "PogEngine-Updater/1.0"

# Allowlist: the only files this tool will ever create or overwrite. Models,
# VOD folders, gallery, CSV histories, logs, checkpoints, and per-VOD bats
# are user data and are never touched.
MANAGED_FILES = [
    "analyze_highlights_emotion.py",
    "OrganizeVODAndFixSRT_Emotion.py",
    "OrganizeVODAndFixSRT_Emotion.bat",
    "pipeline_config.py",
    "isolate_vocals.py",
    "configure_models.py",
    "pog_engine_setup.py",
    "pog_engine_update.py",
    "View_Pipeline_Duration_History.py",
    "View_Pipeline_Duration_History.bat",
    "Install_PogEngine.bat",
    "ConfigurePogEngine.bat",
    "Update_PogEngine.bat",
    "README.md",
    "LICENSE",
]

# (file, variable, path_wrapped) - machine-specific constants the installer
# rewrites per machine; carried verbatim from the old file into the new one.
# path_wrapped mirrors pog_engine_setup.patch_raw_string_constant.
PATH_CONSTANTS = [
    ("OrganizeVODAndFixSRT_Emotion.py", "WHISPER_CLI", False),
    ("OrganizeVODAndFixSRT_Emotion.py", "WHISPER_MODEL", False),
    ("OrganizeVODAndFixSRT_Emotion.py", "WHISPER_VAD", False),
    ("OrganizeVODAndFixSRT_Emotion.py", "GALLERY_DIR", True),
    ("analyze_highlights_emotion.py", "EMOTION_LOCAL_MODEL_DIR", False),
    ("analyze_highlights_emotion.py", "EMOTION_LOCAL_MODEL_FILE", False),
    ("isolate_vocals.py", "TORCH_CACHE_DIR", False),
    ("isolate_vocals.py", "HF_CACHE_DIR", False),
]

VERSION_RE = re.compile(r'^POG_ENGINE_VERSION\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_local_version(install_dir: Path) -> str | None:
    """Installed version literal, or None when the file predates versioning."""
    config = install_dir / "pipeline_config.py"
    if not config.is_file():
        return None
    match = VERSION_RE.search(config.read_text(encoding="utf-8", errors="replace"))
    return match.group(1).strip() if match else None


def normalize_version(tag: str) -> tuple[int, ...] | None:
    """'v.1.2.0' -> (1, 2, 0). None when the tag carries no version numbers
    (e.g. the legacy 'Release' tag), which is treated as incomparable."""
    cleaned = tag.strip().lstrip("vV").lstrip(".")
    parts = cleaned.split(".")
    numbers: list[int] = []
    for part in parts:
        match = re.match(r"(\d+)", part)
        if match is None:
            return None if not numbers else tuple(numbers)
        numbers.append(int(match.group(1)))
    return tuple(numbers) if numbers else None


def api_get(url: str, timeout: int = 30) -> dict:
    """GET a GitHub API URL and return the decoded JSON object."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    )
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"GitHub API {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach api.github.com ({exc.reason}). "
            "Check your internet connection and try again."
        ) from exc


def get_release(ref: str | None) -> dict:
    if ref:
        return api_get(f"{API_BASE}/{GITHUB_REPO}/releases/tags/{ref}")
    return api_get(f"{API_BASE}/{GITHUB_REPO}/releases/latest")


def pick_zip_asset(release: dict) -> dict:
    """The release's PogEngine.zip asset; any other .zip on fallback."""
    assets = release.get("assets") or []
    zips = [a for a in assets if str(a.get("name", "")).lower().endswith(".zip")]
    if not zips:
        names = ", ".join(str(a.get("name", "?")) for a in assets) or "no assets at all"
        raise RuntimeError(
            f"Release {release.get('tag_name')} has no .zip asset ({names}). "
            "Each release must attach PogEngine.zip - see pog_engine_update.py "
            "header for the release contract."
        )
    for asset in zips:
        if asset["name"].lower() == "pogengine.zip":
            return asset
    for asset in zips:
        if "pog" in asset["name"].lower():
            return asset
    return zips[0]


def download(url: str, dest: Path) -> None:
    """Stream a URL to dest atomically, with MB progress on one line."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as f:
            total = response.getheader("Content-Length")
            total_mb = int(total) / 1e6 if total and total.isdigit() else 0.0
            done = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total_mb:
                    print(f"\r  ... {done / 1e6:.1f} / {total_mb:.1f} MB", end="", flush=True)
            print()
        os.replace(tmp, dest)
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed ({exc.reason}). Try again.") from exc
    finally:
        tmp.unlink(missing_ok=True)


def find_zip_root(zip_path: Path) -> str:
    """Prefix to strip so managed files resolve: '' for a flat zip, or the
    single top-level folder for source-style zipballs."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    for depth in (0, 1):
        for name in names:
            parts = name.replace("\\", "/").split("/")
            if len(parts) == depth + 1 and parts[-1] == "pipeline_config.py":
                parent = parts[:-1]
                return ("/".join(parent) + "/") if parent else ""
    raise RuntimeError(
        "PogEngine.zip does not contain pipeline_config.py at its root "
        "(or one folder deep). The release asset is mis-packaged."
    )


def load_config_module(path: Path, label: str):
    """Import a pipeline_config.py from an arbitrary path without polluting
    sys.modules. Raises ImportError (e.g. no requests installed)."""
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[label] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(label, None)
    return module


def carry_config_tunings(old_config: Path, new_target: Path) -> bool:
    """Carry the user's saved EDITABLE_PARAMS defaults from the old config
    into the freshly installed new one, using the NEW file's own registry
    and writer. Returns True on success; False (with a warning) when the
    old file can't be imported - the backup then holds the old tunings."""
    try:
        new_mod = load_config_module(new_target, "pog_update_new_cfg")
    except ImportError as exc:
        log(f"  [WARN] could not read new pipeline_config.py ({exc}) - tunings not carried.")
        return False
    new_params = getattr(new_mod, "EDITABLE_PARAMS", [])
    new_envs = {p["env"] for p in new_params if "env" in p}

    # Scrub the registry's env vars so getattr() returns the SAVED file
    # defaults, not this console's runtime overrides.
    saved_env: dict[str, str] = {}
    try:
        old_mod = load_config_module(old_config, "pog_update_old_cfg_probe")
        old_envs = {p["env"] for p in getattr(old_mod, "EDITABLE_PARAMS", []) if "env" in p}
    except ImportError:
        old_envs = set()
    for name in new_envs | old_envs:
        if name in os.environ:
            saved_env[name] = os.environ.pop(name)
    try:
        try:
            old_mod = load_config_module(old_config, "pog_update_old_cfg")
        except ImportError as exc:
            log(f"  [WARN] could not import old pipeline_config.py ({exc}).")
            log("         Your tunings stay in the backup; re-run ConfigurePogEngine to re-apply them.")
            return False
        carried: dict = {}
        for param in new_params:
            key = param["key"]
            if key == "HYPE_PHRASES":
                value = getattr(old_mod, "DEFAULT_HYPE_PHRASES", None)
                if value is None:
                    value = getattr(old_mod, "HYPE_PHRASES", None)
            else:
                value = getattr(old_mod, key, None)
            if value is not None:
                carried[key] = value
        if not carried:
            return True
        ok, message = new_mod.apply_config_values(carried, str(new_target))
        log(f"  [..]     tuning carry-over: {message}")
        return ok
    finally:
        os.environ.update(saved_env)


def extract_path_constant(source: str, var_name: str) -> str | None:
    """Value of `VAR = r"..."` / `VAR = "..."` / `VAR = Path(r"...")`."""
    match = re.search(
        rf'^{re.escape(var_name)}\s*=\s*(?:Path\()?r?"(.*?)"\)?\s*(?:\r?\n|$)',
        source,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def carry_machine_paths(backup_dir: Path, install_dir: Path, updated: list[str]) -> None:
    """Re-apply this machine's path constants from each backup copy into the
    newly installed file. Same substitution shape as the installer's
    patch_raw_string_constant (callable replacement - Windows backslashes
    must not pass through re escape processing)."""
    for filename, var_name, wrapped in PATH_CONSTANTS:
        if filename not in updated:
            continue
        old_file = backup_dir / filename
        new_file = install_dir / filename
        if not old_file.is_file() or not new_file.is_file():
            continue
        old_value = extract_path_constant(
            old_file.read_text(encoding="utf-8", errors="replace"), var_name
        )
        if old_value is None:
            continue
        with open(new_file, encoding="utf-8", newline="") as f:
            text = f.read()
        if wrapped:
            pattern = rf'^{re.escape(var_name)} = Path\(r".*?"\)'
            replacement = f'{var_name} = Path(r"{old_value}")'
        else:
            pattern = rf'^{re.escape(var_name)} = r".*?"'
            replacement = f'{var_name} = r"{old_value}"'
        new_text, count = re.subn(pattern, lambda m: replacement, text, count=1, flags=re.MULTILINE)
        if count == 0:
            log(f"  [WARN] {filename}: `{var_name}` not in r-string form in the new release - left as shipped.")
            continue
        if new_text != text:
            with open(new_file, "w", encoding="utf-8", newline="") as f:
                f.write(new_text)
            log(f"  [OK]       {filename}: {var_name} kept for this machine")


def clear_pycache(install_dir: Path, filename: str) -> None:
    cache = install_dir / "__pycache__"
    if not cache.is_dir():
        return
    stem = Path(filename).stem
    for stale in cache.glob(f"{stem}.*.pyc"):
        stale.unlink(missing_ok=True)


def atomic_replace(src: Path, dest: Path) -> bool:
    """os.replace; on a sharing violation (e.g. the updater's own running
    .bat) stage a .new file and let the user finish it by hand."""
    try:
        os.replace(src, dest)
        return True
    except OSError as exc:
        staged = dest.with_name(dest.name + ".new")
        try:
            shutil.copy2(src, staged)
            log(f"  [WARN] {dest.name} is locked ({exc}); new version staged as {staged.name} -")
            log(f"         close this window, rename it over {dest.name}, and re-run.")
        except OSError:
            log(f"  [WARN] {dest.name} could not be replaced ({exc}) - left unchanged.")
        return False


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update this Pog Engine install to the latest public release "
        "(only changed files are replaced)."
    )
    parser.add_argument("--check-only", action="store_true",
                        help="print installed vs latest versions and exit.")
    parser.add_argument("--ref", metavar="TAG", default=None,
                        help="update to a specific release tag instead of latest.")
    parser.add_argument("--yes", action="store_true", help="apply without asking.")
    parser.add_argument("--force", action="store_true",
                        help="re-apply even when the versions already match.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    install_dir = Path(__file__).resolve().parent
    log(f"Pog Engine folder: {install_dir}")

    local_version = read_local_version(install_dir)
    log(f"Installed version: {local_version or 'unknown (predates versioning)'}")

    try:
        release = get_release(args.ref)
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        return 1
    tag = str(release.get("tag_name", ""))
    log(f"Latest release:  {tag}")

    release_ver = normalize_version(tag)
    local_ver = normalize_version(local_version) if local_version else None

    if args.check_only:
        if local_ver is not None and release_ver is not None:
            if release_ver > local_ver:
                log(f"Update available: {local_version} -> {tag}. Run without --check-only to apply.")
            else:
                log("Already up to date.")
        else:
            log("Could not compare versions (unparseable tag or unversioned install).")
            log("Run without --check-only (add --force if needed) to diff the files directly.")
        return 0

    if not args.force and not args.ref and local_ver is not None and release_ver is not None:
        if release_ver <= local_ver:
            log("Already up to date - nothing to do.")
            return 0
    if release_ver is None and not args.force and not args.ref:
        log("WARNING: release tag carries no version number - cannot tell if it is newer.")
        log("         Re-run with --force to update anyway, or --ref <tag> for a specific release.")
        return 0

    try:
        asset = pick_zip_asset(release)
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        return 1
    log(f"Release asset:   {asset['name']} ({int(asset.get('size', 0)) / 1e6:.1f} MB)")

    workdir = Path(tempfile.mkdtemp(prefix="pog_update_"))
    zip_path = workdir / "release.zip"
    try:
        log("Downloading...")
        try:
            download(str(asset["browser_download_url"]), zip_path)
        except RuntimeError as exc:
            log(f"ERROR: {exc}")
            return 1

        try:
            prefix = find_zip_root(zip_path)
        except RuntimeError as exc:
            log(f"ERROR: {exc}")
            return 1
        extract_dir = workdir / "extracted"
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            log("ERROR: downloaded file is not a valid zip. Try again.")
            return 1

        # Diff first: stage every changed file, verify it compiles, THEN touch
        # the install. A failure anywhere below leaves the install pristine.
        staged: list[tuple[str, Path]] = []  # (filename, staged source)
        identical: list[str] = []
        missing: list[str] = []
        for filename in MANAGED_FILES:
            staged_src = extract_dir / (prefix + filename)
            if not staged_src.is_file():
                missing.append(filename)
                continue
            target = install_dir / filename
            if target.is_file() and sha256_file(staged_src) == sha256_file(target):
                identical.append(filename)
                continue
            staged.append((filename, staged_src))

        for filename in missing:
            log(f"  [WARN] {filename} not in this release - left unchanged.")

        if not staged:
            log("Every managed file already matches the release - nothing to do.")
            return 0

        log(f"{len(staged)} file(s) differ; verifying the new code compiles...")
        for filename, staged_src in staged:
            if staged_src.suffix.lower() == ".py":
                try:
                    py_compile.compile(str(staged_src), doraise=True)
                except py_compile.PyCompileError as exc:
                    log(f"ERROR: {filename} in the release does not compile: {exc}")
                    log("Aborted - your install was NOT modified.")
                    return 1
        staged_version = None
        staged_cfg = extract_dir / (prefix + "pipeline_config.py")
        if staged_cfg.is_file():
            match = VERSION_RE.search(staged_cfg.read_text(encoding="utf-8", errors="replace"))
            staged_version = match.group(1).strip() if match else None

        log("Changed files:")
        for filename, _ in staged:
            marker = " (new)" if not (install_dir / filename).is_file() else ""
            log(f"  [..]     {filename}{marker}")
        if not args.yes:
            try:
                answer = input("Apply this update? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in ("y", "yes"):
                log("Cancelled - nothing was changed.")
                return 0

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag) or "untagged"
        backup_dir = install_dir / f"update_backup_{safe_tag}_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        updated: list[str] = []
        for filename, staged_src in staged:
            target = install_dir / filename
            if target.is_file():
                shutil.copy2(target, backup_dir / filename)
            if atomic_replace(staged_src, target):
                updated.append(filename)
                clear_pycache(install_dir, filename)

        if not updated:
            log("No files were replaced.")
            return 1

        # pipeline_config.py first (tunings), then machine paths - both read
        # the pre-update originals from the backup, never the new defaults.
        if "pipeline_config.py" in updated and (backup_dir / "pipeline_config.py").is_file():
            log("Carrying your config tunings into the new pipeline_config.py...")
            carry_config_tunings(backup_dir / "pipeline_config.py", install_dir / "pipeline_config.py")
        carry_machine_paths(backup_dir, install_dir, updated)

        log("")
        log(f"Updated {len(updated)}/{len(MANAGED_FILES)} managed file(s) "
            f"to {tag} ({len(identical)} already matched).")
        if staged_version:
            log(f"New installed version: {staged_version}")
        log(f"Backup of the replaced originals: {backup_dir.name}/")
        log("Undo: copy the backup files back over the install.")
        if "pog_engine_update.py" in updated or "Update_PogEngine.bat" in updated:
            log("Note: the updater itself changed - next run uses the new one.")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
