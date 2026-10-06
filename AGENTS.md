# Repository Guidelines

## Project Overview

**Pog Engine** is an AI-powered, Windows-only local VOD processing pipeline for streamers. Drop an OBS `.mp4` onto a launcher; the pipeline transcribes it (whisper.cpp), isolates the streamer's voice, detects viral highlight candidates using local LLMs (Ollama **or** llama.cpp via a `LLM_BACKEND` toggle) plus a speech-emotion model (PyTorch), a model-free audio scan, and configurable hype-phrase scoring, verifies/ranks the candidates, and exports ranked highlights as a CSV and a DaVinci Resolve EDL marker file. The discovery-through-judge analysis stages are checkpointed so a killed/interrupted run resumes exactly where it stopped.

The repository is a **flat layout** — all Python sources, the shared config, and the three top-level `.bat` launchers (`OrganizeVODAndFixSRT_Emotion.bat`, `Install_PogEngine.bat`, `ConfigurePogEngine.bat`) sit at the root. There is no `requirements.txt`/`pyproject.toml`; setup is done by `pog_engine_setup.py` (driven from `Install_PogEngine.bat`). There is no test suite.

## Architecture & Data Flow

### Top-level orchestration: the "6 steps"
For a normal VOD run, users touch two launchers. `OrganizeVODAndFixSRT_Emotion.bat` runs `OrganizeVODAndFixSRT_Emotion.py <video.mp4>`, which is the **organizer**: it creates a `<video_stem>/` folder next to the VOD, moves related files in, probes track layout, and **writes 12 generated `.bat` files** into that folder (steps 1–6 plus debug sub-steps 5a–5f). The optional third launcher, `ConfigurePogEngine.bat`, opens the model/configuration GUI. The user then double-clicks `6_RunAllSteps.bat`, which opens the redesigned Run All window — a canvas-drawn pipeline map (see `run_all_gui()` and the **RunAll GUI** section under Code Conventions) — that runs steps 1–5 in order:

1. **`1_ExtractMicAudio.bat`** — produce `*_mic.wav`. Two paths, auto-selected by `count_audio_streams()` (ffprobe):
   - *Multi-track* (local OBS): `ffmpeg -map 0:a:1` extracts the separate mic track directly, then Step 1 prepares the overlapping Whisper chunks.
   - *Single-track* (Twitch download): the full mix is extracted to a retained Wave64 `.w64` intermediate. Step 1 then persists 10-minute overlapping Demucs input chunks, runs Demucs on each chunk, saves the trimmed separated mic chunks, concatenates them into a Wave64 file, renders `*_mic.wav` as 16 kHz mono with the shared noise gate, and only then prepares the separate overlapping Whisper chunks. Downstream stages see an identical `*_mic.wav` regardless of path.
2. **`2_TranscribeAudio.bat`** — `whisper-cli.exe` (whisper.cpp CUDA build) writes a raw `.srt` by running one fresh process per prepared Whisper chunk, then shifting timestamps and deduplicating overlap captions after every chunk succeeds.
  - **Chunked transcription:** Step 1 prepares 30-minute audio windows with 10 seconds of overlap; Step 2 runs a fresh whisper.cpp process per saved window, shifts timestamps into the full-audio clock, removes duplicate overlap captions, and writes the stitched raw SRT. Tune `TRANSCRIPTION_CHUNK_MINUTES` and `TRANSCRIPTION_CHUNK_OVERLAP_SECONDS` in `pipeline_config.py` or via environment variables.
3. **`3_FixSRT.bat`** → `fix_srt()` — repair bad Whisper timestamps, collapse adjacent duplicate sentences/blocks, renumber, and preserve each retained caption as one SRT block (display wrapping only).
4. **`4_SplitSRT.bat`** → `split_srt_into_chunks()` — regroup captions into fuller "thoughts" (`TRANSCRIPT_MERGE_TARGET_WORDS=30`) and write `transcript_partN.txt` files in `DEFAULT_CHUNK_MINUTES=25` chunks.
5. **`5_AnalyzeHighlights.bat`** → `analyze_highlights_emotion.py <folder>` — the heavy stage (see below).
6. *(no stage-6 bat)* — `6_RunAllSteps.bat` itself opens `run_all_gui()`, the pipeline-map runner that orchestrates steps 1–5.

### RunAll GUI (redesigned pipeline map, EVA-UI-REDESGN 2026-09-05)

`run_all_gui()` was rebuilt into a canvas-drawn "circuit board" pipeline map per the pinned
`GUI REDESIGN CONCEPT/` brief (black `#181818` ground, `#f9f9f9` panels, `#ff8b1b` wiring,
Helvetica Compressed/Regular — resolved at runtime via `_resolve_font_family`). Design tokens and
rules live in `DESIGN.md`; product context in `PRODUCT.md`.

- **Mini-step registry is the single source of truth.** `BIG_STEP_MINI_STAGES` +
  `SINGLE_TRACK_STEP_MINI_STAGES` (Twitch route swaps Step 1 to 1a–1g) drive the map layout, the
  hover tooltips (`MINI_DESCRIPTIONS`), and the output parser together — future mini steps append
  by editing the registries only.
- **Route convergence.** The routes differ only in Step 1: `_vod_is_single_track()` call sites are
  confined to Step 1's registry/parser branch, the Step 1 tooltip text, and the header label
  (LOCAL/TWITCH VOD). Step 1's exit contract (`*_mic.wav` + `prepared_chunks`) is where both
  routes meet — Steps 2–5, their bats, parser branches, and tooltips are route-agnostic.
- **State model.** `step_state` / `mini_state` / `sync_state` / `finish_lit` feed
  `_cell_fill_color()`: neon `#00ff7e` done, progress green `#1d874d` working, pure red `#ff0000`
  failed, dark red `#751313` waiting, orange `#ff8b1b` stopped. Waiting cells never glow.
- **Glow** = pre-rendered Gaussian-blur RGBA sprites (PIL) in the cell's own silhouette, cross-faded
  through per-state 16-step intensity tables on a 120 ms clock (~1.9 s breath); the in-progress cell also glides
  Progress Green ↔ Neon Green through 16 distinct fills and settles solid Neon on finish. Sprites build lazily one per
  event-loop tick; without PIL the glow degrades silently (no crash).
- **Interaction.** Hovering a cell draws an orange 2 px outline and opens a styled tooltip
  (260 ms delay); ghost buttons (`_GhostButton`) have wrapper-frame outlines and orange hover fill;
  the live console text area is pure `#000000`; console and status panels stretch to equal heights.
- **Responsive contract.** The window opens maximized (`zoomed`); the IMAGE GALLERY panel shows only
  when maximized AND width ≥ 1500 px, otherwise it is replaced by a SCREEN SAVER panel — all map
  columns stay visible from `minsize(1000, 660)` upward. Opening plays a staged fade-in
  (window alpha → panels → cells).
- **Preserved contracts.** Stop (taskkill tree + Ollama unload + `last_stop_state.json`), duration
  recording to `View_Pipeline_Duration_History.csv`, checkpoint-skip reporting, auto-sleep, and the
  exit-code contract are unchanged from the pre-redesign GUI.

### Step 5 internal pipeline (`analyze_highlights_emotion.py`)
Step 5 is one external process from the GUI's view but internally runs **6 sub-stages** in `STAGE_ORDER = ["discovery", "audioscan", "emotion", "verify", "judge", "export"]`; the first five produce checkpoints and export consumes the judge checkpoint without writing one of its own:

- **discovery** — for each `transcript_partN.txt`, run multiple LLM passes (`PROMPTS`: Emotion/Gameplay/Viral) over the transcript via Ollama (`MODEL`/`/api/chat` with `think:false`, same qwen3.5 treatment as the judge-side stages). Returns raw candidates with Timestamp/Title/Reason. `is_low_content_title()` then hard-filters junk (single-word, repeated phrase) titles. **Anti-hallucination**: claimed timestamps are checked against real transcript timestamps (`TIMESTAMP_TOLERANCE_SECONDS=15`).
- **audioscan** — model-free DSP pass over the whole mic track (`compute_audio_arousal_series`: loudness + speech-rate proxy per hop, z-scored against the stream's own baseline). Peaks (`pick_energy_peaks`, non-max suppression) not already near an LLM candidate become *new* candidates, titled in batch via `title_audio_candidates()` → `MODEL`/`/api/chat` (`think:false`), already resident from discovery. The parser tolerates the model's doubled ItemNumber (`1,1,"Title","Reason"`). Closes the gap where a wordless reaction (scream, laughter) has no transcript text to discover.
- **emotion** — `classify_candidate_emotions()` attempts the local speech-emotion model (`firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`, safetensors) around each candidate timestamp → `emotion_scores.csv`; disabled, missing, or failed optional scoring soft-falls back without aborting the stage. `apply_emotion_scores_to_highlights()` merges scores; `emotion_boost()` adds its bounded score boost. `apply_hype_phrase_boost()` then searches transcript blocks within ±`HYPE_PHRASE_WINDOW_SECONDS`, requiring `HYPE_PHRASE_MIN_MATCHES`, and adds a configured boost capped at 2.0 points. It records `HypePhraseBoost`/`HypePhrases` in the CSV and phrase settings in `run_info.json`. Ollama models are unloaded (`unload_ollama_model`) when the emotion model is loaded to free CUDA VRAM.
- **verify** — `verify_candidates()` sends batches (`VERIFY_BATCH_SIZE`) to `JUDGE_MODEL` via `/api/chat` with `think:false` (the qwen3.5 `/api/generate` bug, ollama/ollama#14793, ignores `think:false`). Filter hallucinated/unsupported timestamps. Batches that fail to parse keep candidates unverified; if parsed-verdict ratio drops below `VERIFY_MIN_COVERAGE_RATIO=0.5`, the stage is untrusted and **not** checkpointed.
- **judge** — `run_judge_tournament()` → `run_judge_batch()` limits input to `JUDGE_POOL_SIZE` candidates and ranks the pool down to `TOP_N` by a 5-factor priority order. Uses the same `/api/chat` + `think:false` treatment as verify — thinking ON on `/api/generate` used to be deliberate, but newer Ollama emits qwen3.5's reasoning into a separate `thinking` field that counts against `num_predict` and returns an empty `response` (see Fix History 2026-08-05). Short parses backfill by score; a total parse failure prints a warning instead of silently degrading to a score sort.
- **export** — `calibrate_final_scores_by_rank()` (rank caps to fight inflation), `score_to_resolve_color()` (score→Resolve marker color), then `write_highlights_csv()` → `top<N>_highlights.csv`, `write_highlights_edl()` → `top<N>_markers.edl` (Resolve EDL), `write_run_info()` → `run_info.json`. Optionally `export_preview_clips()` (off by default; `EXPORT_PREVIEW_CLIPS`).

The discovery, audioscan, emotion, verify, and judge stages each write `checkpoint_<name>.json` atomically (`.tmp` then `os.replace`); export has no checkpoint of its own and always runs from the judge checkpoint. `run_all_remaining_stages()` skips completed checkpointed stages, so a crashed run picks up where it stopped. Each stage appends console output to `log_<stage>.txt`; `record_pipeline_run_history()` appends a row to `pipeline_run_history.csv` (kept next to the scripts, not per-VOD). `--stage <name>` forces one stage and invalidates downstream checkpoints (used by `5a`–`5f` debug bats).

### Two-tier LLM design
`MODEL` and `JUDGE_MODEL` (both default `qwen3.5:9b-q4_K_M`, ~6.6 GB at Q4_K_M - fits a 10 GB card in either role). Discovery reads long transcript chunks and proposes candidates; verify/judge need careful structured reasoning over shorter prompts. Audio-scan titling (short-title writing, no ranking) runs on `MODEL`, already resident from discovery, so a split-role setup loads the 35B judge GGUF only for verify/judge. Context budgets are split per role — `DISCOVERY_NUM_CTX` and `JUDGE_NUM_CTX`, both default 8192 (down from 32768) — fits a 10 GB 3080 at Q4_K_M after weights + KV cache. The split lets a partial-CPU-offload judge model shrink only its own window (the `qwen3.6:35b-a3b` preset uses `JUDGE_NUM_CTX=6144`); both are switchable at runtime via the config GUI, which rewrites `pipeline_config.py` defaults (see `configure_models.py` below). All LLM calls go through one `ollama_generate()` with retry (`OLLAMA_RETRIES`, backoff), centralized call/timing stats, and — on a connection-level failure — automatic Ollama boot via `ensure_ollama_running()` (spawns a detached `ollama serve` for a local `OLLAMA_URL`, waits for `/api/version`, then retries; see config comment + Fix History 2026-08-05).

**llama.cpp alternative (`LLM_BACKEND="llamacpp"`):** the single toggle collapses or hot-swaps the two-tier split. `LLAMA_SERVER_URL` (default `http://localhost:8080`, editable) + `LLAMA_DISCOVERY_MODEL_PATH`/`LLAMA_JUDGE_MODEL_PATH` (fallback `LLAMA_MODEL_PATH` for single-GGUF mode) drive a **managed `llama-server`**. `analyze_highlights_emotion.py` owns the process (`_LLAMA_PROC`): `discovery` starts the discovery GGUF, `emotion` stops the server to free VRAM for the PyTorch model (mirroring Ollama's `keep_alive:0` unload), `verify`/`judge` restart with the judge GGUF if different (`audioscan` reuses the already-running discovery GGUF). `ollama_generate()` translates Ollama-shaped payloads to the OpenAI-compatible `/v1/chat/completions` (chat vs prompt, `chat_template_kwargs:{"enable_thinking":false}`, `max_tokens`, `<think>` stripping, shim response) so the six call sites stay unchanged. `llm_is_reachable`/`llm_not_ready_message` dispatch on `LLM_BACKEND` (`/health` vs `/api/version`). `LLAMA_CONTEXT_SIZE` (`-c`) is the single server-launch context (per-request `num_ctx` does not exist on llama.cpp). `OrganizeVODAndFixSRT_Emotion.py` generates `Start_LlamaServer.bat` (manual launch doc; pipeline manages hot-swap) and the RunAll preflight checks GGUF existence, not `/health`. See `pipeline_config.py` LLM backend toggle block for rationale.
## Key Directories

Flat repo — no subdirectories. Source files at root:

| File | Role |
|---|---|
| `pipeline_config.py` | Shared tunable constants, including the hype-phrase signal settings; all are overridable via env vars. Imported by every other Python file. Comments here are the best architecture doc. |
| `OrganizeVODAndFixSRT_Emotion.py` | VOD organizer + SRT fixing/transcript chunking + RunAll Tkinter GUI. Entry for the whole workflow. |
| `analyze_highlights_emotion.py` | Core highlight engine: discovery, audio scan, emotion/hype scoring, verify, judge, export. The largest, most complex file. |
| `isolate_vocals.py` | Demucs vocal separation for single-track VODs; subprocess target of `1_ExtractMicAudio.bat`. |
| `pog_engine_setup.py` | Setup/verification installer (GUI + `--cli`): checks dependencies, downloads models, patches machine-specific paths. |
| `configure_models.py` | Tk GUI for choosing Ollama models / pipeline defaults. Renders one form from the `EDITABLE_PARAMS` registry, applies `PRESETS`, and saves via `pipeline_config.apply_config_values()`, which rewrites that file's own default literals. Opened via `ConfigurePogEngine.bat` or the organizer's `--config`. |
| `OrganizeVODAndFixSRT_Emotion.bat` | Launcher: `python OrganizeVODAndFixSRT_Emotion.py "%~1"` (drag-drop a VOD onto a shortcut to it). |
| `Install_PogEngine.bat` | Bootstrap launcher: enforces exactly Python 3.12 (auto-installs 3.12.10 when missing or any other version, prompts old-version uninstall), then runs `pog_engine_setup.py <folder>`. |
| `ConfigurePogEngine.bat` | Launcher: `python OrganizeVODAndFixSRT_Emotion.py --config --no-pause` (opens the config GUI; no video needed). |
| `plan.md` | Test-run journal of the single-track sample VOD: bug discoveries (#1 bat paren crash, #8 stats inflation, #9 judge silent no-op) with repros, fixes, and verification notes. |

**Generated per-VOD** (inside `<video_stem>/`, not in repo): `1_ExtractMicAudio.bat` … `6_RunAllSteps.bat`, debug `5a`–`5f` bats, `vod_audio_info.json` (ffprobe audio-stream count + which extraction path was chosen, written by `organize_video()`), `<name>_mixed_full.w64` retained for single-track processing, `<name>_demucs_chunks/` with persisted full-mix Demucs input chunks and manifest, `<name>_mic_demucs_chunks/` with persisted trimmed separated mic chunks and manifest, `<name>_mic_demucs_combined.w64`, `<name>_mic.wav`, `<name>_mic_transcription_chunks/` with persisted Whisper `.wav` chunks and raw `.srt` files, `*.srt`, `transcript_partN.txt`, `checkpoint_<stage>.json`, `log_<stage>.txt`, `emotion_scores.csv`, `pipeline_stats.json`, `run_info.json`, `top<N>_highlights.csv`, `top<N>_markers.edl`, `step6_run_*.log`.

## Development Commands

There is no package manager / build step. Workflows:

```
# First-time install (Windows):
Install_PogEngine.bat              # GUI installer; locates/installs Python, deps, models, patches paths
python pog_engine_setup.py --cli <folder>  # same logic, console-only

# Process a VOD:
#   1. Drag the .mp4 onto a shortcut to OrganizeVODAndFixSRT_Emotion.bat  (creates the VOD folder + step bats)
#   2. Double-click the VOD folder's 6_RunAllSteps.bat                   (pipeline-map GUI runs steps 1→5)
# Transcribe audio independently to reset Whisper context on long VODs:
python OrganizeVODAndFixSRT_Emotion.py --transcribe-audio <mic.wav>
python OrganizeVODAndFixSRT_Emotion.py --transcribe-audio <mic.wav> --transcription-chunk-minutes 15 --transcription-overlap-seconds 10

# Run only step 5 (after steps 1-4 already produced transcript_part files):
python analyze_highlights_emotion.py "<VOD_folder>"

# Re-run one internal stage in isolation (clears downstream checkpoints):
python analyze_highlights_emotion.py "<VOD_folder>" --stage verify    # discovery|audioscan|emotion|verify|judge|export

# SRT utilities without organizing a video:
python OrganizeVODAndFixSRT_Emotion.py --fix-srt <in.srt> [--chunk-minutes 25]
python OrganizeVODAndFixSRT_Emotion.py --split-srt <fixed.srt>
python OrganizeVODAndFixSRT_Emotion.py --run-all-gui "<VOD_folder>" [--base-name NAME]

# Model / default configuration GUI (rewrites pipeline_config.py defaults; env vars still override at runtime):
ConfigurePogEngine.bat                        # Tk GUI; equivalently: python OrganizeVODAndFixSRT_Emotion.py --config

# Vocal isolation directly:
python isolate_vocals.py <mixed_audio.wav-or-w64> <output_mic.wav>
```

No `pytest`, `lint`, `format`, or `test` commands exist.

## Code Conventions & Common Patterns

- **Python 3.12**, source kept modern but defensive: `from __future__ import annotations` (so PEP 604 `X | None` type hints don't crash on an older interpreter than the targeted 3.12 — the setup script then prints a clear version warning instead of `SyntaxError`).
- **External tools via subprocess, never libraries** — ffmpeg/ffprobe, whisper-cli.exe, and Demucs are invoked as subprocesses (Demucs is run as a CLI specifically rather than via its Python API). LLM generation uses the centralized `requests.post` wrapper `ollama_generate()` which dispatches on `LLM_BACKEND` (Ollama `/api/generate|/api/chat` vs llama.cpp `/v1/chat/completions` translation + `llm_is_reachable` health dispatch); Ollama `ollama serve` is started with `subprocess.Popen` only for the local connection-failure recovery path, while llama.cpp's server is managed (`_LLAMA_PROC`, `taskkill /F /T` with a terminate fallback) by the analyzer itself.
- **Checkpointing / resume** — discovery, audioscan, emotion, verify, and judge each use `checkpoint_exists()` → `run_stage_*()` inside `with stage_log(folder, stage):` (a `_Tee`-backed context manager duplicating stdout to `log_<stage>.txt`), then save a checkpoint and record stage stats. Export consumes the judge checkpoint and always runs. `invalidate_downstream()` deletes this stage's and all later checkpoints when forcing a debug rerun.
- **Error handling** — stage functions can `sys.exit(1)` on a checkpoint health-check failure (e.g. verify refusing to save an untrusted result), which intentionally skips `record_pipeline_run_history()` so incomplete runs don't corrupt timing averages. `ollama_generate()` retries with linear backoff; transient failures no longer silently drop candidate coverage. `Organize…main()` wraps everything in `try/except → return 1` and a `pause_if_needed()` finally block (the `.bat`'s `pause`).
- **Atomic file writes** protect config, checkpoints, pipeline stats, and installer downloads (`.tmp` then `os.replace`/`Path.replace`). CSV, EDL, `run_info.json`, and `vod_audio_info.json` are direct output writes rather than atomic state updates.
- **SRT handling** — `transcribe_audio_in_chunks()` preserves overlapping Whisper windows in `<audio_stem>_transcription_chunks/`, reuses existing chunk audio on reruns, and starts a fresh Whisper process for each chunk; `stitch_transcription_chunks()` shifts timestamps and removes exact overlap duplicates before `fix_srt()` repairs timestamps, removes adjacent duplicate sentences/blocks, renumbers captions, and emits each retained caption as one SRT block. Long text is wrapped across display lines but is not split into sentence/word-sized blocks. `split_srt_into_chunks()` remains the separate transcript-part grouping step.
- **No DI / no stateful domain classes** — pipelines are module-level functions; the value objects are small frozen `@dataclass` classes (`SubtitleEntry`, `RunAllStep`), `_Tee` is a small log adapter with stream references, and the setup script's `Reporter`/nested `GuiReporter` share console/GUI reporting.
- **Threading model** — `run_all_gui()` runs each step's `.bat` via `subprocess.Popen` on a worker thread, with a `queue.Queue` ferrying line events to the Tk UI; a **Stop** button kills `current_process["popen"]`. `isolate_vocals.py` is a pure synchronous subprocess target.
- **Tkinter for GUIs** — `pog_engine_setup.run_gui()` is plain Tk/ttk (`clam` theme); the redesigned
  RunAll window is a `tk.Canvas`-drawn pipeline map with its own design tokens (see `DESIGN.md`) and no ttk
  styling. PIL is required for the glow sprites and the gallery; without it they degrade gracefully
  (no glow, empty gallery) and everything else still works.

## Important Files

- **`pipeline_config.py`** — start here. Every constant has an inline comment explaining the *why* (model sizing vs VRAM, qwen3.5 `/api/generate` bug anti-hallucination tolerance, multitrack vs single-track routing, chunked Whisper context reset, hype-phrase matching). Reading this file alone conveys most of the architecture. `EDITABLE_PARAMS` (`:312`) + `PRESETS` (`:585`) + `apply_config_values()` (`:690`) back the config GUI's save-back.
- **`analyze_highlights_emotion.py:352`** — `STAGE_ORDER`; **`:2837`** — `STAGE_FUNCS` dispatch; **`:2900`** — `main()` / `--stage`; **`:1105`/`:1113`** — hype phrase matching/boost; **`:2363`** — `PROMPTS` (the discovery prompt family); **`:2523`/`:2561`** — `VERIFY_PROMPT_HEADER` / `JUDGE_INSTRUCTIONS`.
- **`OrganizeVODAndFixSRT_Emotion.py:1096`** — `build_run_all_steps()` (the canonical 5-step list with input/expected file contracts);
  **`:1145`/`:1176`/`:1472`** — `BIG_STEP_MINI_STAGES` / `SINGLE_TRACK_STEP_MINI_STAGES` / `MINI_DESCRIPTIONS` (the mini-step registry trio — edit these together);
  **`:1641`** — `run_all_gui()` (the redesigned pipeline-map runner); **`:3763`** — `organize_video()` (folder setup + the generated `.bat` files + `vod_audio_info.json`); **`:3861`** — `parse_args()` (subcommands incl. `--config`).
- **`pog_engine_setup.py`** — `patch_paths()` (`:513`) rewrites the `r"G:\pog_dev\…"` raw-string constants (incl. `TORCH_CACHE_DIR` *and* `HF_CACHE_DIR` in `isolate_vocals.py`) to the chosen install folder; **`:688`** `_demucs_repo_dir()` presence-check anchored on the real `models--adefossez--HTDemucs` HF repo (not a loose file extension); **`:712`** `_migrate_default_hf_htdemucs()` moves an orphaned copy from the default user cache into the redirect; **`:743`** `predownload_demucs_model()` sets `TORCH_HUB_CACHE`/`HF_HUB_CACHE` (not just `*_HOME`) so the redirect actually holds; **`:1163`** `run_all_checks()` orders the install phases; **`:1524`** `main()` (`--cli` vs GUI).
- **`configure_models.py`** — config GUI (opened by `ConfigurePogEngine.bat` or the organizer's `--config`, which lazy-imports it so a config edit never loads the heavyweight pipeline imports). Renders one form from `EDITABLE_PARAMS`, applies `PRESETS`, saves via `cfg.apply_config_values()`.
- **`Install_PogEngine.bat`** — prerequisite Python bootstrap: locates `python`/`py`, installs python-3.12.10-amd64.exe silently when absent or not exactly 3.12 (other installs left untouched, uninstall recommended via prompt), then hands to `pog_engine_setup.py`.

## Runtime/Tooling Preferences

- **OS: Windows only** — generates `.bat` files and uses Windows `pause` prompts. Installation is driven by `Install_PogEngine.bat`, which can bootstrap Python through a downloaded `.exe`; Pog Engine itself is not a packaged executable.
- **Python 3.12 exactly (3.12.10 via the installer bootstrap)** — required and enforced by the launcher (anything but 3.12 triggers an automatic 3.12.10 install; no other version is ever used). `pog_engine_setup.py` still warns instead of failing on other versions for direct-script runs, with ROCm torch the hard 3.12 requirement.
- **No virtualenv is created** by setup; packages install into the interpreter that ran the setup. The legacy `.gitignore` mentions `2.0/.venv/` and `2.0/pog_engine.egg-info/` (a previous packaging layout, since abandoned).
- **External binaries the pipeline shells out to** (must be on PATH or referenced by the patched path constants):
  - **Ollama** (default, `LLM_BACKEND="ollama"`) — must be pre-installed by the user, running model `qwen3.5:9b-q4_K_M` at `OLLAMA_URL`/`OLLAMA_CHAT_URL` (both pipeline roles default to it; `qwen3:8b` remains a supported alternative in the tuning tables) (default `http://localhost:11434`). `qwen3.5` thinking is disabled via `/api/chat` `think:false` to dodge ollama/ollama#14793 (verify/judge use it on `JUDGE_MODEL`; discovery/titling use it on `MODEL`). The setup verifies the models are pulled. *Alternative:* **llama.cpp** (`LLM_BACKEND="llamacpp"`) — `llama-server` + per-role GGUFs (`LLAMA_DISCOVERY_MODEL_PATH`/`LLAMA_JUDGE_MODEL_PATH`, fallback `LLAMA_MODEL_PATH`) at `LLAMA_SERVER_URL` (default `http://localhost:8080`). The analyzer manages `llama-server` and hot-swaps GGUFs (discovery→stop for emotion→judge) so no manual `Start_LlamaServer.bat` is required; that bat is kept only for manual testing. `LLAMA_CONTEXT_SIZE` (`-c`) replaces per-request `num_ctx`. Setup verifies `llama-server` on PATH and GGUF existence; RunAll checks GGUF existence instead of `/health`.
  - **whisper.cpp** CUDA/cublas build (`whisper-cli.exe`) + `ggml-large-v3.bin` — downloaded + extracted by setup.
  - **FFmpeg + ffprobe** — checked by setup; used across mic extraction, transcode, preview clips, and vocal finalization.
- **Python packages installed by setup** (`SIMPLE_PACKAGES`): `requests`, `numpy`, `transformers`, `librosa`, `soundfile`, `safetensors`, `Pillow`. Plus **PyTorch** (GPU-aware: matched against the driver's max CUDA via `CUDA_WHEEL_TAGS` `cu130`→`cu118`; falls back to CPU) and **demucs** (vocal separation). Demucs' ~80 MB `htdemucs` weights are pre-downloaded during setup by `predownload_demucs_model()` into `<models>\hf_cache\huggingface\hub\models--adefossez--HTDemucs\` (redirect via `HF_HUB_CACHE`; legacy torch-hub `*.th` also accepted). See "Vocal isolation model cache" below — the weights are **not** in `models\` itself and must not be looked for by loose file extension.
- **Speech-emotion model** (`firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3`) is pinned; setup downloads `config.json`, `preprocessor_config.json`, the safetensors weights, and points `EMOTION_LOCAL_MODEL_DIR` at them.
- **Discrete GPU strongly preferred** — Demucs and the emotion model auto-select a GPU backend when `…_DEVICE == "auto"` (default): CUDA on NVIDIA, ROCm HIP on AMD 7000/9000-series (needs Python 3.12 + recent Adrenalin; CPU otherwise, much slower on multi-hour VODs). Transcription is GPU-accelerated on NVIDIA only (no Vulkan whisper.cpp Windows build; CPU-bound on AMD). LLM stages run on either vendor via Ollama-Vulkan / llama.cpp-Vulkan. A 10 GB card (e.g. 3080) is the design target for VRAM sizing.

## Testing & QA

- **No automated tests exist** — no `tests/`, no `pytest`, no `unittest`, no test files anywhere in the tree (confirmed via search). There is no CI configuration in the repo.
- **Resilience instead of tests** — the pipeline's correctness strategy is **checkpointing + anti-hallucination guards + health checks**, not unit tests: per-stage JSON checkpoints, atomic writes, timestamp-tolerance filtering, min-coverage verification gate, low-content-title rejection, and final score rank-capping to fight inflation.
- **Manual QA path** — `pog_engine_setup.py --cli` runs the full dependency/model/path verification suite (the same `Reporter`-driven checks the GUI uses) and reports `[OK]`/`[MISSING]`/`[Failed]` per item; the `5a`–`5f` debug `.bat` files force-rerun one analysis sub-stage each so a stage's prompt/logic can be exercised in isolation (clearing stale downstream checkpoints automatically).
- **GUI QA rig** — `.impeccable/review/harness.py` (+ `wincap.py`) drives `run_all_gui()` against a temp VOD folder with a fake `subprocess.Popen` that streams scripted step output, so every GUI state (working/failed/complete/Twitch route/restored window) renders in seconds without a real pipeline. Captures via `PrintWindow` (occlusion-proof). Run: `python .impeccable/review/harness.py <working|failed|complete|complete_small|twitch_working|twitch_complete|hover|stay>`. Dev-only; not part of the user install.
- **No coverage tooling** is configured.

### Notes for an AI assistant
- **RunAll GUI changes must move as one unit**: the mini-step registries (`BIG_STEP_MINI_STAGES`, `SINGLE_TRACK_STEP_MINI_STAGES`),
  the hover descriptions (`MINI_DESCRIPTIONS`), and the output parser (`update_step_from_output`) all encode the same phases — update
  them together and keep parser keywords matching the real echo strings of the bats/`isolate_vocals.py` (probe before editing).
- **Route awareness ends at Step 1**: keep `_vod_is_single_track()` call sites confined to Step 1's registry/parser branch and the header;
  Steps 2–5 are route-agnostic by design (both routes converge on Step 1's `*_mic.wav` + `prepared_chunks` contract).
- **Sync workflow after GUI/pipeline changes**: private `main` (push) → copy the changed scripts to the public repo
  (`E:\VIAL\Pog_Engine`, commit + push) → copy to the installed `G:\pog_dev`. The installed copy is a snapshot; the generated
  per-VOD bats call installed scripts by absolute path, so no bat regeneration is needed for script-only changes.
- **Glow/pulse internals**: sprites cache per (kind, size, color, phase, mirror, state_tag); intensity tables are per state
  (`GLOW_PHASES_DEFAULT/RUNNING/FAILED`); the in-progress cell's fill color-cycles green-to-green via `_running_fill()`. Without PIL the glow
  silently disappears (no crash) — check PIL is installed when users report "no glow".
- The richest source of architectural intent is the **comments inside `pipeline_config.py`** — read it first; it documents decisions the code only implies.
- Three hardcoded `r"G:\pog_dev\…"` path clusters (`analyze_highlights_emotion.py`, `isolate_vocals.py`, `OrganizeVODAndFixSRT_Emotion.py`) are rewritten by `pog_engine_setup.patch_paths()`; editing them by hand is fragile — prefer re-running setup.
- `ollama_generate()` is the centralized LLM generation call; `ollama_is_reachable()` probes `/api/version` and the best-effort unload helpers issue direct POSTs to release models. **With `LLM_BACKEND="llamacpp"` it dispatches to `llama-server`'s `/v1/chat/completions` (translation + `<think>` stripping + shim) and `llm_is_reachable()` probes `/health`.** All four Ollama call sites (discovery, titling, verify, judge) route to `/api/chat` with top-level `think:false` by passing `url=OLLAMA_CHAT_URL`; generate-style payloads (flat `prompt`) are still translated for robustness, and llama.cpp always uses the OpenAI chat endpoint with `chat_template_kwargs:{"enable_thinking":false}`. The analyzer owns `llama-server`'s lifecycle (`_LLAMA_PROC`, `_start/_stop`, `taskkill`) and hot-swaps GGUFs (`discovery`→`judge`, stop for `emotion` to free VRAM). Keep this distinction when adding LLM-driven stages. The analyzer and RunAll GUI require the LLM server to be active before Step 5 for Ollama; for llama.cpp the server is managed per-stage and the RunAll preflight checks GGUF existence instead. If unavailable, they instruct the user to close RunAll, launch the required server manually, and rerun.
- **Vocal-isolation model cache — the load-bearing invariant.** Demucs' `htdemucs` weights live in HF-Hub layout: `<models>\hf_cache\huggingface\hub\models--adefossez--HTDemucs\` (snapshot → blob symlinks). Two things must stay in sync or this silently breaks:
  1. `predownload_demucs_model()` in `pog_engine_setup.py` sets `HF_HUB_CACHE` **and** `TORCH_HUB_CACHE` (not just `*_HOME` — newer `huggingface_hub`/torch honor the precise vars and ignore `HF_HOME` alone, which is what left the weights silently in the default `~/.cache/huggingface` on older installs). The presence-check (`_demucs_repo_dir`) looks for the **actual repo dir** by name; do NOT revert to matching by `.safetensors`/`.bin` extension — unrelated Ollama GGUF pulls in the default cache will make it falsely report "already downloaded."
  2. `use_local_torch_cache()` in `isolate_vocals.py` sets the **same two env vars** so the runtime reads from the same redirect setup writes to. If you add a stage that loads demucs/another HF model, mirror these two var pairs (`TORCH_HOME`+`TORCH_HUB_CACHE`, `HF_HOME`+`HF_HUB_CACHE`) or the two sides will read and write different caches.
- **`patch_paths()` rewrites only the install-folder copy of the scripts, not the repo source** — and the install folder is a snapshot from whenever the user last ran setup, so a repo fix to `pog_engine_setup.py`/`isolate_vocals.py`/etc. does **not** propagate to existing installs. To apply a script fix to an already-installed Pog Engine, copy the patched repo source into the install folder before re-running `Install_PogEngine.bat`/`pog_engine_setup.py --cli`, or the installer will patch stale scripts.
- When changing a `STAGE_*` checkpoint name or `STAGE_ORDER`, the RunAll GUI and `5a`–`5f` debug bats are generated from `build_run_all_steps()` / `make_debug_stage_bat()` — regenerate the VOD folder's bats by re-running `OrganizeVODAndFixSRT_Emotion.py` against the video.
- **Long-VOD transcription invariant** — do not restore one uninterrupted Whisper invocation for multi-hour audio. `2_TranscribeAudio.bat` must call `--transcribe-audio`; each window must start a fresh decoder context, retain overlap, shift timestamps, and preserve chunk SRTs for diagnosis. If changing chunk defaults, update both `pipeline_config.py` and the generated-launcher behavior in `OrganizeVODAndFixSRT_Emotion.py`; regenerate existing VOD helper bats when needed.
- **Config save-back mechanics** — `configure_models.py` writes no env or state files; it rewrites the *default literals* in `pipeline_config.py`'s source (`apply_config_values()`: regex anchored on the `os.environ.get|_env_int|_env_float|_env_bool` helper call, atomic `.tmp` + `os.replace`). Adding a new editable parameter requires BOTH the definition AND an `EDITABLE_PARAMS` entry whose `env` name matches it. The `HYPE_PHRASES` list has a dedicated rewrite branch that normalizes/deduplicates phrases case-insensitively. Two definitions (`EMOTION_ENABLED`, `VOCAL_ISOLATION_SEGMENT_SECONDS`) skip the helper-call form and have dedicated rewrite branches in `apply_config_values()` — change their shape and update the matching special case. Runtime env vars still override any saved default.
- **Hype phrase signal** is scoring-only: `apply_hype_phrase_boost()` runs in the emotion stage after emotion scoring, matches configured phrases case-insensitively inside the candidate window, and cannot create candidates. The audio scan remains the candidate source for wordless reactions. Keep the CSV/run-info metadata fields when changing this signal.
- **`README.md`** is the user-facing workflow and configuration overview. Keep it aligned with stage behavior and tunable names; `pipeline_config.py` remains authoritative for configuration rationale and save-back mechanics.

## Fix History: WAV 4GB overflow on long single-track VODs (2026-08-03)

### Symptom
A Twitch-download VOD (`G:\pog_dev\twitchvod\sample.mp4`, **7h34m**) halted at Step 1 (`1_ExtractMicAudio.bat`) inside the RunAll GUI. The `step6_run_*.log` showed two compounding failures:
1. ffmpeg's RIFF/WAV muxer warned the output exceeded the 4 GB ceiling: `Filesize 4810436686 invalid for wav, output file will be broken` (the stereo s16le @ 44.1 kHz PCM mix is ~4.81 GB — past the 32-bit size field).
2. Demucs then crashed loading the structurally-broken `*_mixed_full.wav`: `memory allocation of 4831838208 bytes failed`, exit code `3221226505` (`0xC0000409` = `STATUS_STACK_BUFFER_OVERRUN`). Step 1 reported `Failed` and the RunAll GUI stopped.

### Root cause
`make_extract_mic_bat_singletrack()` in `OrganizeVODAndFixSRT_Emotion.py` extracted the full single-track mix to a standard **`.wav`** intermediate before handing it to `isolate_vocals.py`/Demucs. The RIFF/WAV container caps audio data at 4 GB (32-bit size field); any single-track VOD longer than ~6h15m at 44.1 kHz stereo s16le overflows it. ffmpeg keeps writing bytes after the warning (it's a header-validity issue, not a hard stop), producing a non-conformant file Demucs' loader (soundfile/libsndfile) can't open.

### Fix (generator, not the generated bat)
Changed the single-track intermediate container from `.wav` to **Wave64 (`.w64`)** — same signed 16-bit PCM payload, but a 64-bit size field with no 4 GB ceiling. This preserves Demucs' separation-grade input (44.1 kHz stereo) exactly; only the container changes, so separation quality is untouched. Demucs reads W64 via the same soundfile/libsndfile backend with no code change in `isolate_vocals.py`.

Edits in `make_extract_mic_bat_singletrack()` (`OrganizeVODAndFixSRT_Emotion.py`):
- `mixed_wav_name`: `{base}_mixed_full.wav` → `{base}_mixed_full.w64`.
- ffmpeg extraction command: added `-f w64` (`ffmpeg -y -i … -map 0:a:0 -ar 44100 -ac 2 -f w64 "…mixed_full.w64"`).
- Docstring rewritten to record the 4 GB WAV limit as the reason for Wave64.

`isolate_vocals.py` requires **no change** — it already takes the mixed path as a CLI arg and Demucs/soundfile read W64 transparently.

### Applied to two locations
- `E:\VIAL\Pog_Engine_priv\OrganizeVODAndFixSRT_Emotion.py` — repo/source copy.
- `G:\pog_dev\OrganizeVODAndFixSRT_Emotion.py` — **installed** copy the machine actually runs. Per "Notes for an AI assistant" above, `patch_paths()` snapshots scripts at install time, so source fixes don't auto-propagate; the install copy was stale (un-patched pre-fix) and got the same edit. Both copies now match byte-for-byte in the edited region and compile clean.

### Verification performed
- ffmpeg `w64` (Sony Wave64) muxer confirmed present; default codec `pcm_s16le`.
- soundfile/libsndfile 1.2.2 confirmed to support W64 + `PCM_16` (Demucs' reader).
- End-to-end smoke test against the real VOD: `ffmpeg … -f w64 sample.mp4 → _w64_smoke2.w64` succeeded; `soundfile.read` returned 15.0 s / 44.1 kHz / stereo — the exact separation-grade input Demucs expects.
- Regenerated `G:\pog_dev\twitchvod\sample\1_ExtractMicAudio.bat` from the install copy: uses `G:\pog_dev\isolate_vocals.py` (matching bats 3/4/5) and `.w64` for extract/isolate/del consistently. Deleted the stale broken `sample_mixed_full.wav`.

### Residual risk (not addressed by this fix)
`VOCAL_ISOLATION_SEGMENT_SECONDS` is unset by default. The 4 GB container bug is gone, but a 7.5 h unsegmented file may still **CUDA OOM inside Demucs** on a 10 GB card. If Step 1 fails again with a CUDA out-of-memory error (distinct from the WAV/crash error fixed here), set it before re-running: `set VOCAL_ISOLATION_SEGMENT_SECONDS=20` then `6_RunAllSteps.bat` — Demucs then processes in bounded 20 s windows with only minor boundary-quality leakage (the downstream emotion/verify/judge stages don't care about that).

### Key takeaway
Single-track (Twitch-download) VODs are the only path that writes an intermediate full-mix audio file; multi-track (local OBS) extracts the mic directly and never hits this. The Wave64 fix is specific to `make_extract_mic_bat_singletrack()` and its generated bat; the multi-track builder (`make_extract_mic_bat_multitrack`) is unaffected.
## Fix History: judge stage silent no-op + inflated pipeline stats (2026-08-05)

### Symptom 1: judge silently did nothing on installed Ollama
`checkpoint_5_judged.json` / `top50_highlights.csv` had **no `JudgeScore` on any row**; the final order matched a plain score sort. `run_judge_batch()` ran on `/api/generate` with thinking **enabled** (deliberate — comparative ranking was supposed to benefit from reasoning) and parsed only the `response` field.

**Root cause:** Ollama ≥ 0.3.x (installed: **0.32.5**) splits qwen3.5's reasoning into a separate `thinking` JSON field that **counts against `num_predict`** and is no longer inlined in `response`. The model burned the entire 3000-token budget mid-thought (`done_reason: length`, `response` empty — reproduced with 3000 and 6000), so the CSV rows never appeared and `run_judge_batch` returned `[]`, silently falling back to `highlights_by_score[:top_n]`. The config comment about ollama/ollama#14793 (`/api/generate` ignores `think:false`) describes the same empty-response failure for calls that DO pass `think:false`; the judge hit it by *not* passing it.

**Fix** (`analyze_highlights_emotion.py`, both source + installed copies):
- `run_judge_batch()` now uses `OLLAMA_CHAT_URL` + top-level `"think": False` + `messages` payload — the exact path verify/titling already used successfully on this model. Judge response came back in 4–6 s with clean CSV rows (`done_reason: stop`).
- Parse hardened: strip `"` quotes around timestamp fields; scores parsed as floats and rounded (the model emitted `7.5`) instead of the old `int(re.sub(r"[^\d-]", ...))` which mangled `7.5` → `75` → clamp 10.
- Total parse failure now prints a `[!] Judge returned no parseable CSV rows` warning with a response excerpt instead of silently degrading to a score sort.
- `JUDGE_INSTRUCTIONS` tail tightened: "plain values only ... no quotes around fields, no decimals".
- Comments updated: the judge-prompt design block, `run_judge_batch` docstring, and the `OLLAMA_CHAT_URL` note in `pipeline_config.py` (judge joined the `/api/chat` list).

**Verified:** fresh full run on `sample1_cut` — judge finished in 4.2 s (was ~35 s of burned reasoning), 10 candidates ranked with the judge's order driving the top rows and `JudgeScore` populated on model-ranked rows (backfill per design fills the rest).

### Symptom 2: pipeline_stats.json / run_info.json counts inflated in single-process runs
Run 2's stats recorded audioscan **35 found / 20 kept** when the log said **7 / 4** (exactly 7×5, 4×5), and **31 Ollama calls / 416 s** when only **7 calls / ~67 s** happened.

**Root cause:** `record_stage_stats()` did `+=` the process-global `CALL_STATS`/`AUDIO_SCAN_STATS` into the persisted `pipeline_stats.json` at **every** stage boundary. In a single-process run (the normal `5_AnalyzeHighlights.bat`/RunAll path) each later stage re-added all earlier stages' counts (cumulative adds 3+4+4+6+7+7 = 31 — exact match). Only multi-process debug runs (5a–5f) got correct numbers by accident.

**Fix** (`record_stage_stats()`): reset `CALL_STATS`/`AUDIO_SCAN_STATS` to zero after folding, so each call records only the delta since the last successful stage end. Inflated the run summary, `run_info.json`, and the `pipeline_run_history.csv` timing averages.

**Verified:** fresh full run prints `7 Ollama call(s), 1.1 min in Ollama` (exactly the 3 discovery + 1 titling + 2 verify + 1 judge calls); `pipeline_stats.json` shows `audio_scan_candidates_found: 7, kept: 4`. A later `--stage judge` rerun added exactly +1 call (delta semantics hold across processes too).

### Historical feature (superseded 2026-08-21): stage 5 auto-started a stopped Ollama
This behavior was removed after startup timing/model readiness problems. Current behavior is a preflight health check: Step 5 refuses to start until Ollama answers `/api/version`, then instructs the user to close RunAll, launch Ollama manually, and rerun.

## Fix History: installer treated truncated/wrong-size downloads as complete (2026-08-11)

### Symptom
`download_file()`'s skip-if-exists only tested `dest_path.is_file()` and a fresh stream was renamed over the destination without comparing received bytes to the announced `Content-Length`. A failed/partial download that left a wrong-size file on disk was reported `[OK] already exists` forever; an early-closed connection could also be renamed as a "complete" file. (User hit exactly this in another project's installer.)

### Fix (`pog_engine_setup.py`, both source + installed copies)
- New `remote_file_size(url)` helper: best-effort HEAD for the origin's `content-length`; returns `None` (no re-download, no failure) when the server doesn't answer or reports no length — never a false positive from a broken HEAD.
- `download_file()` skip-if-exists now HEAD-checks the origin: size mismatch logs `[WARN] ... exists but size N != expected M, re-downloading`; matching size (or unverifiable origin) skips as before.
- Both `download_file()` and the whisper.cpp ZIP download in `download_whisper_cpp_cublas()` reject a fresh download when `downloaded != total` with an `IOError` (temp deleted; stale dest preserved for resume). Missing `Content-Length` on the GET falls back to `remote_file_size()`; non-numeric header values are ignored instead of crashing the download.
- The extracted `whisper-cli.exe` skip-if-exists is unchanged — the exe has no direct origin URL (it lives inside the ZIP), and `zipfile.extractall` CRC-checks its contents anyway.

### Verification
Local-HTTP-server test against the actual `download_file()`: fresh download exact-size ✓; silent truncation (server lies about `Content-Length`, closes early) → `False` + no dest file ✓; wrong-size existing file + verifiable HEAD → re-downloaded to exact size with `[WARN] re-downloading` logged ✓; correct-size existing file skipped ✓; unverifiable HEAD keeps file ✓. 7/7 assertions passed on Python 3.11.9 + requests 2.34.2. Both copies compile clean; edited regions byte-identical.

## Fix History: installer downloads get SHA-256 origin verification (2026-08-11)

### Symptom
Size-only verification couldn't catch a file that was the right byte count but wrong content (corrupted mid-transfer, upstream re-publish, disk bit rot).

### Fix (`pog_engine_setup.py`, both source + installed copies)
- `sha256_file(path)` — chunked full-file digest.
- `origin_sha256(url)` — the *origin's* published SHA-256, fetched at runtime instead of hardcoded, so moving upstream files can never brick the installer on a stale pin:
  - huggingface.co `resolve/…` URLs → HF tree API (`/api/models/<repo>/tree/<rev>?recursive=true`), whose `lfs.oid` is the true blob sha256 for LFS files. Response cached per repo/revision — one API call serves all files of a repo. Small non-LFS files (emotion model's `config.json`, `preprocessor_config.json`) have no sha256 → `None`, size check only.
  - The whisper.cpp ZIP is a pinned, immutable GitHub release asset → digest from the GitHub API (`assets[].digest`) hardcoded in `_PINNED_SHA256` with the verification date.
  - Anything else → `None`, size check only. Lookup failures are swallowed, never crash or block a download.
- `download_file()` skip-if-exists: local file must match BOTH size and (when available) sha256; mismatch logs the reason (`size N != expected M` / `sha256 mismatch`) and re-downloads. Fresh downloads: after the byte-count check, the temp file is hashed against the origin's digest before the atomic rename — mismatch raises, temp is deleted, stale dest preserved.
- Same hash gate on the whisper.cpp ZIP download before extraction (in addition to `zipfile` CRC checks).
- Logs: `(sha256 verified)` on skip, `sha256 verified (abc123…)` after a fresh download.


### Verification
- Live origin lookups (7/7): `ggml-large-v3.bin` → `64d182b4…`, `ggml-silero-v6.2.0.bin` → `2aa269b7…`, emotion `model.safetensors` → `d5a5dfcd…`, both JSON configs → `None` (non-LFS), pinned ZIP → `3fc4d3eb…`, foreign URL → `None`.
- Local-server functional tests (9/9): fresh matching-hash download accepted + verified logged; fresh mismatching-hash download rejected, no dest file; correct existing file skipped; **same-size-different-bytes tampered file caught by sha256 and re-downloaded**; no-hash origin falls back to size-only; `sha256_file` matches `hashlib` directly.
- Both copies compile clean; edited regions byte-identical. Note: hashing a 3 GB model on the skip path adds a few seconds per setup re-run — the price of the guarantee.
## Fix History: long-VOD transcription context collapse (2026-08-23)

### Symptom
A long VOD had audible speech after the one-hour mark, but the raw Whisper SRT repeated the same sentence for roughly three hours. `fix_srt()` removed the repeated blocks, so the fixed SRT appeared to stop at one hour; it could not recover speech that Whisper never transcribed.

### Root cause
`2_TranscribeAudio.bat` ran one uninterrupted whisper.cpp process over the entire audio file with unlimited decoder context (`-mc -1`). Once decoding entered a repeated-text loop, the long-running context could reinforce that hallucination for the rest of the VOD.

### Fix
`OrganizeVODAndFixSRT_Emotion.py` now implements the transcription path as two
strict phases:
- **Step 1:** `1_ExtractMicAudio.bat` extracts the mic track, then
  `prepare_audio_chunks()` extracts every overlapping window and saves all
  chunk `.wav` files plus `chunk_manifest.json` in
  `<audio_stem>_transcription_chunks/`.
- **Step 2:** `2_TranscribeAudio.bat` runs a fresh whisper.cpp process on each
  saved chunk. It does not extract audio. Only after every chunk SRT succeeds,
  `transcribe_audio_in_chunks()` shifts timestamps into the full-audio clock,
  removes overlap duplicates, and writes the stitched raw SRT.

`TRANSCRIPTION_CHUNK_MINUTES` and `TRANSCRIPTION_CHUNK_OVERLAP_SECONDS` are
environment-overridable and exposed through the configuration GUI. The existing
fix and transcript-part stages remain downstream of the stitched raw SRT.

### Applied
- Repository source: `OrganizeVODAndFixSRT_Emotion.py`, `pipeline_config.py`.
- Installed copy: `G:\pog_dev\OrganizeVODAndFixSRT_Emotion.py`, `G:\pog_dev\pipeline_config.py`.
- Existing generated launcher: `G:\pog_dev\twitchvod\sample1_cut\2_TranscribeAudio.bat`.

### Verification
Both source and installed Python copies compile with `py_compile`. The installed CLI exposes `--transcribe-audio`; a synthetic stitching check confirmed timestamp shifting and overlap duplicate removal.

### Operational note
The full multi-hour Whisper run still requires the real VOD, whisper.cpp executable, model, VAD model, ffmpeg, and available GPU/CPU runtime. If repetition remains, reduce the chunk length to 15 minutes before changing VAD thresholds or model settings.
## Fix History: explicit Twitch Demucs and Whisper phases (2026-08-25)

### Required Twitch order
For a single-audio-stream Twitch VOD, the generated launchers must complete
these boundaries in order:

1. `1a`: extract the full mixed track to retained Wave64 `<name>_mixed_full.w64`.
2. `1b`: split the full mix into persisted 10-minute Demucs input chunks.
3. `1c`: run Demucs once per input chunk.
4. `1d`: save each trimmed separated vocal chunk under `<name>_mic_demucs_chunks/`.
5. `1e`: concatenate the separated chunks into `<name>_mic_demucs_combined.w64`.
6. `1f`: render the combined vocal audio to `<name>_mic.wav` at 16 kHz mono with the shared noise gate.
7. `1g`: split the rendered mic WAV into the separate persisted Whisper chunk set.
8. `2a`: run one fresh Whisper process per rendered mic chunk.
9. After every chunk SRT succeeds, shift timestamps, deduplicate overlap captions, and write the stitched raw SRT.

Whisper must never receive the full mixed VOD or the full unchunked isolated
track. Step 1 must finish Demucs separation, rendering, and Whisper-chunk
preparation before Step 2 starts. Matching manifests and non-empty chunk files
are reused on reruns.

### Implementation
- `isolate_vocals.py` owns persistent Demucs input chunks, per-chunk separation,
  trimmed mic chunks, concatenation, and final mic rendering.
- `make_extract_mic_bat_singletrack()` passes explicit Demucs chunk, mic chunk,
  and combined-audio directories into `isolate_vocals.py`.
- `prepare_audio_chunks()` remains the Whisper chunk preparation module.
- `transcribe_audio_in_chunks()` remains the Whisper runner and SRT stitcher.

### Applied
- Repository source: `OrganizeVODAndFixSRT_Emotion.py`, `isolate_vocals.py`.
- Installed copy: `G:\pog_dev\OrganizeVODAndFixSRT_Emotion.py`,
  `G:\pog_dev\isolate_vocals.py`.
- Existing generated launchers must be regenerated by rerunning the organizer;
  generated per-VOD `.bat` files are snapshots.
The RunAll GUI now detects `single_track_mode` from `vod_audio_info.json` and
shows the Twitch-specific `1a` through `1g` mini-process cards. It updates
those cards from the live Step 1 output, including Demucs input preparation,
per-chunk separation, mic-chunk persistence, concatenation, final rendering,
and Whisper-chunk preparation. Local multi-track VODs retain the original
three-card Step 1 display.
The RunAll GUI forwards combined stdout/stderr from every generated batch and
child Python process into the Live Console with line-buffered subprocess reads;
the parent `[STATUS] 1. Extract mic audio: Running` message is only the outer
Step 1 state, while the subsequent `Step 1a` through `Step 1g` output shows the
active Twitch sub-step.

### Verification
Source and installed Python copies compile with `py_compile`. Synthetic smoke
checks confirmed persisted Demucs and mic chunk counts, combined-audio creation,
generated launcher ordering, Whisper chunk reuse, timestamp shifting, and
overlap duplicate removal. A real multi-hour Demucs/Whisper run still requires
the VOD, ffmpeg, Demucs model, whisper.cpp, VAD model, and available GPU/CPU.


## Fix History: preserved chunk audio reuse (2026-08-24)

### Change
Chunked transcription now keeps each extracted audio window in
`<audio_stem>_transcription_chunks/` after Whisper finishes. A later
transcription run reuses an existing non-empty chunk `.wav` instead of
extracting that window again from the multi-hour source. Whisper still starts a
fresh process for every preserved chunk, and the final stitch remains the only
input to downstream SRT repair.
### Applied
- Repository source and installed copy: `OrganizeVODAndFixSRT_Emotion.py`.
- Existing generated launcher: `F:\OBS VOD\2026-08-16 11-50-24\2_TranscribeAudio.bat`.

### Invariant
Whisper receives only individual chunk audio files; it never receives the full multi-hour `_mic.wav`. The full audio is used only by ffmpeg to create missing chunks and by ffprobe to read duration.


## Fix History: strict local-VOD extraction/transcription phases (2026-08-24)

### Required local-VOD order
For a VOD with separate game and mic tracks, the generated launchers must
complete these boundaries in order:

1. `1a`: extract mic audio with `ffmpeg -map 0:a:1`.
2. `1b`: split the completed mic WAV into all overlapping audio chunks.
3. `1c`: save every chunk and `chunk_manifest.json` under
   `<audio_stem>_transcription_chunks/`.
4. `2a`: run one fresh Whisper process per saved chunk.
5. After every chunk SRT succeeds, stitch them into the full-audio timeline.

Whisper must never start while Step 1 is still extracting chunks. The
`--transcribe-audio` path requires the manifest and every listed non-empty chunk;
missing preparation data is an error that tells the user to rerun Step 1.

### Implementation
- `prepare_audio_chunks()` owns extraction, overlap calculation, persistence,
  and manifest creation. The manifest records each chunk's full-audio
  `offset_ms`, nominal range, source name, duration, chunk length, and overlap.
- `transcribe_audio_in_chunks()` only loads the prepared manifest, runs Whisper
  once per chunk, then calls `stitch_transcription_chunks()`.
- `stitch_transcription_chunks()` shifts each chunk SRT by its manifest offset,
  sorts captions by connected timecode, removes duplicate overlap captions,
  renumbers the result, and rejects invalid timestamp ranges.
- RunAll treats Step 1 as complete only when the mic WAV and the complete
  prepared chunk set validate successfully.
- Regenerate existing per-VOD `.bat` files by rerunning the organizer; generated
  launchers are snapshots and do not update when repository source changes.

### Verification
`py_compile` passes. Synthetic checks cover complete chunk preparation before
Whisper, overlap deduplication, timestamp shifting, generated launcher order,
and the RunAll output contract. A real VOD run still requires ffmpeg,
ffprobe, whisper.cpp, model files, and the local GPU/CPU runtime.

## Fix History: model-load visibility + end-of-run Ollama unload (2026-08-26)

### Symptom
RunAll log `step6_run_20260826_160237.log` showed three problems: after 5f
export finished, the judge model (`qwen3.6:35b-a3b`) stayed resident in the
Ollama server; step 5e's first batch silently stalled on a cold model load
(the visible `[!] Ollama call failed (500 ...)` retry), with nothing in the
live console or GUI saying a load was happening; and every mini-process card
read "Waiting for Step N" instead of naming its actual predecessor.

### Root causes
- Nothing ever asked Ollama to unload models at the end of a full run; only
  the emotion stage unloads them (to free VRAM mid-run). The Ollama server
  outlives the pipeline process, so weights stayed in VRAM until keep_alive
  expiry.
- Current Ollama reliably returns **500 on the FIRST request that triggers a
  cold load** of the partially-offloaded 35B judge model (reproduced
  standalone: attempt 1 → 500, retry after backoff → loads fine). The real
  verify/judge batches were only surviving this via `ollama_generate()`
  retries, invisibly.
- `set_mini_stage()` in the RunAll GUI accepted a `detail` argument but never
  applied it to `detail_var` — card detail text could never leave its initial
  "Waiting for ..." value — and no longer updated the status label color.

### Fix
- `analyze_highlights_emotion.py`: new `ollama_model_loaded()` (GET `/api/ps`)
  and `ensure_ollama_model_ready(model, url, stage_label)` — prints
  `[stage] Loading Ollama model ...` when cold, warms the model up with a
  1-token request through its own retry loop (not `ollama_generate()`, so
  `CALL_STATS` stay exact), then prints ready-in-Ns. Called at the start of
  discovery (MODEL) and audioscan/verify/judge (JUDGE_MODEL).
- New `unload_all_ollama_models()` runs at the end of
  `run_all_remaining_stages()` (i.e. every full run that reaches export),
  unloading MODEL + JUDGE_MODEL; `unload_ollama_model()` now targets
  `OLLAMA_URL` instead of a hardcoded localhost URL and takes a label.
  Debug `--stage` runs deliberately do not unload.
- `OrganizeVODAndFixSRT_Emotion.py` RunAll GUI: mini-card waiting labels are
  chained ("Waiting for Step 5" → "Waiting for 5a" → …); `set_mini_stage()`
  applies its detail text and status color again; analyzer loading/ready lines
  are parsed into the active mini-process card so a cold judge-model load is
  visible instead of looking like a hang.

### Applied
- Repository source and installed copies (`E:\VIAL\Pog_Engine_priv\`,
  `G:\pog_dev\`) of both files, byte-identical. Generated per-VOD bats are
  unaffected: they invoke the installed Python scripts directly.

### Verification
- Live-Ollama test of the helpers: cold → announce → 500 → retry → loaded
  (`/api/ps` confirms) → warm path reports "already loaded" without warming;
  `unload_all_ollama_models()` leaves `/api/ps` empty.
- Real-GUI harness (auto-closing `mainloop`, faked `subprocess.Popen` feeding
  scripted analyzer output): all 15 waiting labels correct per dependency;
  `[judge]/[verify]/[discovery]` loading + ready lines land on the right
  cards; event-drain loop survives (an earlier draft used `set_step_mini`,
  which mutates `state["stage"]` and crashed the drain with an
  `ANALYSIS_STAGE_BY_LABEL` KeyError — do not reintroduce).

## Fix History: llama.cpp backend with managed hot-swap (2026-09-02)

### Symptom
Ollama is the only LLM backend. Users asked for a toggle to `llama.cpp`
without changing pipeline shape: Ollama uses two models sequentially
(`qwen3:8b` for discovery, then unload, then `qwen3.5:9b` for
verify/judge/titling); llama.cpp's `llama-server` hosts exactly one GGUF per
process and has no API to swap models in-place.

### Design
`LLM_BACKEND` (`"ollama"` default, `"llamacpp"` alternative) in
`pipeline_config.py` drives the whole stack. `ollama_generate()` is the single
dispatch point: llama paths are translated to the OpenAI-compatible
`/v1/chat/completions` (`messages` vs `prompt` wrapping, `chat_template_kwargs`
`{"enable_thinking":false}` + `reasoning_budget:0` for `think:false`,
`max_tokens` from `num_predict`, `<think>` stripping, shim
`{"message":{"content":…}}`/`{"response":…}` so the six call sites stay
Ollama-shaped. `llm_base_url`/`llm_is_reachable`/`llm_not_ready_message`
branch on `LLM_BACKEND` (`/health` vs `/api/version`).

Managed hot-swap (Option 2, `llama-cpp-support` branch): `LLAMA_SERVER_URL`
(`http://localhost:8080`, now editable) + `LLAMA_DISCOVERY_MODEL_PATH` /
`LLAMA_JUDGE_MODEL_PATH` (fallback `LLAMA_MODEL_PATH` for single-GGUF mode) +
`LLAMA_CONTEXT_SIZE` (`-c` at launch, replaces per-request `num_ctx`).
`analyze_highlights_emotion.py` owns `llama-server` (`_LLAMA_PROC`,
`atexit`, `taskkill /F /T`): `discovery` starts discovery GGUF,
`emotion` stops the server to free VRAM before the PyTorch model (mirrors
Ollama's `keep_alive:0`), `audioscan`/`verify`/`judge` restart with the judge
GGUF if different (deduped when both roles resolve to same path). `main()`
preflight for llamacpp checks GGUF existence, not `/health`. Final
`unload_all_ollama_models()` now stops the managed server. `Start_LlamaServer.bat`
is kept as a manual doc launcher (shows both GGUFs, discovery GGUF by default)
but is optional — the pipeline no longer requires the user to pre-launch the
server.

### Applied
- `pipeline_config.py`: `LLM_BACKEND`, `LLAMA_SERVER_URL`, per-role GGUFs,
  `LLAMA_CONTEXT_SIZE`, `_coerce_for_write` `backend` kind,
  `EDITABLE_PARAMS` LLM backend stage (6 entries), llama.cpp preset.
- `analyze_highlights_emotion.py`: imports, `require_ollama_ready` /
  `ollama_model_loaded` / `ensure_ollama_model_ready` now llm-aware, global
  `_LLAMA_PROC` + `_llama_resolve_model` / `_parse_host_port` /
  `_wait_for_health` / `_stop` / `_start` / `_ensure_llama_server_for_role`
  + per-stage wiring and `main()` GGUF preflight.
- `OrganizeVODAndFixSRT_Emotion.py`: imports per-role paths, preflight checks
  GGUF existence, `make_llama_server_bat` now documents both GGUFs and
  managed hot-swap, `make_llama_server_bat` added to `organize_video` output.
- `configure_models.py`: LLM backend readonly combobox, header update, backend
  validation, new stage renders automatically.
- `pog_engine_setup.py`: `check_ollama` now branches — llamacpp verifies
  `llama-server[.exe]` on PATH and both unique GGUFs + `/health` INFO.

### Verification
- `py_compile` all, per-role resolve fallback, host/port parse, bat two-model
  doc, and lifecycle mock (discovery start → judge stop+start → same-judge reuse
  → stop via taskkill) — all mocked, no real `llama-server` needed.
- Earlier regressions fixed in same sweep: restored `_phrase_defaults_source`
  (HYPE save-back), restored `running judge stage` mini-card, fixed
  `llm_base_url` misuse for llamacpp helpers and `ollama_generate` error
  message base URL.


## Fix History: dead-code and comment cleanup (2026-09-05)

### Removed
- `import atexit` in `analyze_highlights_emotion.py` — dead since the llama-server stop logic moved to direct taskkill with a terminate/wait fallback (5ff2153 removed the `atexit.register(_stop_llama_server)` hook).
- `llm_base_url` in `pipeline_config.py` plus its imports in the organizer and the analyzer — defined and imported but never called anywhere.
- `llm_is_reachable` / `llm_not_ready_message` imports in `OrganizeVODAndFixSRT_Emotion.py` — unused in that file (the analyzer uses them).
- `mark_box_mode` intermediate local in the RunAll GUI's failed-event handler.

### Updated
- AGENTS.md llama.cpp sections now describe the current stop logic (taskkill with terminate fallback, no atexit hook) and no longer reference `llm_base_url`.
- Comment pass: markdown-style emphasis removed from the audio-scan intro; the RunAll GUI docstring and glow comments tightened. No behavior changes; all modules py_compile clean.

## Fix History: RunAll GUI redesign + dead-code cleanup (EVA-UI-REDESGN, 2026-09-05)

### What changed
- `run_all_gui()` rebuilt into a canvas-drawn pipeline map per the pinned `GUI REDESIGN CONCEPT/` brief: state-colored ribbon cells
  (alternating slant per column, inside-mounted orange chips, orange code labels outside), ST bars chained along the top, trunk
  brackets for the code labels, spines entering sync bars at mid height, a 5F→SYNC:4-5 elbow, chamfered 45° routing, sync drops into
  the finish bar. Sync bars light when one step finishes AND the next starts; POG ENGINE FINISH turns neon only when every step and
  sync finished.
- Animation: staged fade-in (window alpha → panels → cells), breathing glow from pre-rendered Gaussian-blur RGBA sprites
  (two/three stacked layers per cell, per-state intensity tables, in-progress cell color-cycles Progress Green ↔ Neon Green and
  settles solid Neon on finish), all built lazily one sprite per event-loop tick so the UI never freezes.
- Interaction: hover tooltips (withdraw→geometry→deiconify to avoid the default-position Leave race), orange hover outlines,
  ghost buttons with wrapper-frame outlines + hover fill, pure `#000000` console text area, equal-height console/status panels.
- Responsive: gallery panel only while maximized AND ≥ 1500 px wide; otherwise a SCREEN SAVER panel replaces it; map scales
  0.55×–1.5×, all columns visible from minsize 1000×660.
- Dead code removed: `import atexit` + `llm_base_url` (function + imports) + unused organizer imports + a leftover intermediate
  local in the failed-event handler. Comment pass removed markdown-style emphasis from comments.

### Bugs fixed along the way
- Sync-to-sync connector loop ran one iteration too many (`range(n-1)` → `range(n-2)`), drawing a stray stub past SYNC:4-5
  (survived several rounds because earlier fixes removed the wrong neighbors).
- `mark_box_mode` intermediate local; tooltip Toplevel mapped at its default position under the pointer, firing a canvas `<Leave>`
  that killed the tooltip before it showed (fixed via withdraw → geometry → deiconify).

### Verification
- QA rig (`.impeccable/review/harness.py`) drove every state against a fake pipeline: working (3B hold), failed (2B), complete,
  complete-small (restored window), Twitch working/complete, hover tooltips/outlines; captures via `PrintWindow`.
- `py_compile` on all modules; public repo + `G:\pog_dev` synced and compile-verified.

## Fix History: ffmpeg download falls back to GitHub mirrors (2026-09-05)

### Symptom
A user reported ffmpeg "failing to download" during setup while the same installer worked on other machines.

### Root cause
`download_ffmpeg()` downloaded the ~106 MB essentials bundle from `www.gyan.dev` only — a single personal host behind Cloudflare that fails for some users (region/ISP interference, bot-protection 403s, antivirus HTTPS filtering), with no retry source and no progress feedback during the download.

### Fix
`download_ffmpeg()` now walks `FFMPEG_SOURCES` (module constant): gyan.dev primary → `GyanD/codexffmpeg` GitHub release (byte-identical gyan build on GitHub's CDN) → BtbN `FFmpeg-Builds` latest win64-gpl (last resort; different build layout, handled by the existing `rglob` extraction). Each failed source is logged with its reason (`[WARN] <source> failed: <exc>; trying the next mirror...`), temp artifacts are cleaned between attempts, and the final warning reports the last error if every source fails. Skip-if-exists, truncation check, registry PATH persistence, and non-blocking failure behavior are unchanged.

### Verification
Functional tests against a local HTTP server: 404 first source falls through to a working mirror (bundle installed); skip-if-exists makes no HTTP requests; a lying Content-Length truncated download is rejected (IncompleteRead surfaced, no bundle installed, no `.tmp` leftovers). Live HEAD checks: gyan.dev 200 (~106 MB → ffmpeg 9.0.1), GyanD mirror 200 (111.3 MB), BtbN 200 (170.8 MB).

### phi4:14b model option (2026-09-06)
`DISCOVERY_MODEL_TUNINGS["phi4:14b"] = {"DISCOVERY_NUM_CTX": 6144}` and
`JUDGE_MODEL_TUNINGS["phi4:14b"] = dict(_JUDGE_HEAVY)` — a dense 14B judge/discovery option
(Q4_K_M ~ 9.1 GB fits a 10 GB card only with the trimmed context; same VRAM-pressure profile as
the 35B MoE entries: smaller batches, more retries). Users must `ollama pull phi4:14b` first; the
configurator's Scan auto-detects it and applies the tuning on selection. Known-good test target for
the judge role (see the 2026-09-05 model scan).

### qwen3.8-27b-mtp:latest model option (2026-09-16)
`DISCOVERY_MODEL_TUNINGS["qwen3.8-27b-mtp:latest"] = {"DISCOVERY_NUM_CTX": 6144}` and
`JUDGE_MODEL_TUNINGS["qwen3.8-27b-mtp:latest"] = dict(_JUDGE_HEAVY)` — a dense 27B judge/discovery
option (14.54 GB at IQ4_XS per `/api/tags`; always partial-offloads a 10 GB card and runs all
27.3B params per token, so slower than the MoE 35B tags under offload). Same VRAM-pressure
profile as the 35B MoE / phi4 entries: trimmed context, smaller batches, more retries. The
configurator's Scan auto-detects it and applies the tuning on selection.

## Fix History: Ollama RAM pre-flight before model loads (2026-09-06)

### Symptom
qwen3.6:35b-a3b (23 GB MoE judge) intermittently failed at load with a cryptic memory error (`CUDA error: out of memory` / "not enough memory to allocate") depending on what else was using RAM/VRAM at that moment.

### Root cause
The load needs ~7.5 GB free VRAM + ~16 GB free system RAM (23.4 GB model minus the VRAM share, plus KV/compute buffers). With the discovery model resident or RAM-hungry apps open, the pre-flight math failed and the load died mid-warmup with no actionable message.

### Fix
`ensure_ollama_model_ready()` (which every Ollama-calling stage already calls: discovery, audioscan, verify, judge) now pre-flights memory before warming: model size from `/api/tags`, free RAM via `GlobalMemoryStatusEx`, free VRAM via `nvidia-smi`. Required = weights + 2 GB KV/compute headroom; available = free RAM + free VRAM. A shortfall prints a clear WARNING with the numbers and the fix (close apps) BEFORE the load is attempted; the load is still attempted (mmap/page file can save it), and `ollama_generate()`'s retry chain recovers from Ollama's own first-attempt 500s.

### Verification
Reproduced the exact failure under simulated memory pressure (8 GB hog + unload): the CUDA OOM from Ollama's inner llama-server startup appeared, preceded by the new WARNING with correct numbers; the retry chain then loaded successfully. Healthy-headroom case prints no warning. All Ollama-calling stages are covered via the shared helper.

## Fix History: VOD folder reorganization into per-step folders (2026-09-06)

### Change
The VOD folder no longer accumulates every artifact at the root. Each step's
outputs (including every mini step's) live in that step's own folder:

- `step1_extract_mic_audio/` - mic WAV, Wave64 intermediates, Demucs chunk sets
- `step2_transcribe_audio/` - raw stitched SRT (+ the prepared transcription
  chunk set stays in `step1_extract_mic_audio/`, where step 1 generated it)
- `step4_split_srt/` - transcript_partN.txt files
- `step5_analyze_highlights/` - checkpoints, per-sub-stage logs, emotion
  scores, highlights CSV, run_info.json, pipeline_stats.json, debug dumps,
  preview clips, and the 5a-5f debug bats (generated there)

Staying at the VOD root: the runner bat (renamed **6_RunAllSteps.bat →
Run_Pog_Engine.bat**), the step bats 1-5, `step6_run_*.log` (the big
combined run log), `vod_audio_info.json`, `last_stop_state.json`, and the
two DaVinci hand-off files (the fixed SRT and the marker EDL).

### Implementation notes
- Shared helper `step_subdir(stream_folder, step)` in `pipeline_config.py`
  (STEP_FOLDER_NAMES); organizer, analyzer, and GUI all resolve through it.
- `fix_srt()` / `split_srt_into_chunks()` take an optional `out_dir`; the
  generated 3/4 bats bake `--out-dir` (drag-drop use without it still writes
  next to the input). The analyzer's export writes the EDL to the root and
  the CSV to step 5.
- The analyzer's scans (`list_transcript_parts`, `find_mic_wav`) check the
  step folder first and fall back to the legacy root, so pre-reorganization
  VOD folders still resume correctly.
- `migrate_legacy_vod_folder()` (called at RunAll GUI startup) moves legacy
  root artifacts into the step folders, renames 6_RunAllSteps.bat, and logs
  each move into the live console.
- Steps 1-4 get `log_stepN.txt` files fed from the GUI's captured subprocess
  output (step 5 keeps its per-sub-stage analyzer logs).

### Verification
QA harness updated for the new layout (fake pipeline writes per-step
outputs); working/complete/failed/Twitch scenarios verified the folder
structure, root placement of the fixed SRT + EDL + big log, and skip/resume
behavior across the new paths.

## Fix History: discovery empty responses under qwen3.5 + Path-refactor stragglers (2026-09-06)

### Symptom
Real-VOD run: discovery loaded the model fine, but **every pass of every
transcript part parsed 0 candidates** (`Response received in ~50-70 s` then
`Added 0`), with the dumped raw response empty. Separately, the analyzer
crashed with `NameError: name 'part' is not defined` in `run_discovery`
(debug-dump path and the exception handler around it) and
`TypeError: sequence item 0: expected str instance, WindowsPath found`
when printing the transcript-part list.

### Root causes
1. Discovery still ran `/api/generate` with thinking ON - deliberate under
   the old `qwen3:8b` default, but the discovery default is now
   `qwen3.5:9b-q4_K_M`, and qwen3.5's reasoning lands in Ollama's separate
   `thinking` field and exhausts the generation budget, leaving `response`
   empty (the same failure the judge stage hit 2026-08-05).
2. The VOD-folder reorganization changed `list_transcript_parts()` to return
   `Path` objects and renamed the loop variable to `part_path`, but three
   references inside `run_discovery` (`SourcePart`, the debug filename, the
   error print) and one in `build_transcript_blocks_by_part` (the dict key)
   still used the old `part` name.

### Fix
- Discovery payload switched to `messages` + top-level `think:false` +
  `url=OLLAMA_CHAT_URL` (identical to titling/verify/judge); result is read
  from `message.content`. All four LLM call sites now use `/api/chat`.
- New `DISCOVERY_NUM_PREDICT` (`HIGHLIGHT_DISCOVERY_NUM_PREDICT`, default
  2000, GUI-editable via `EDITABLE_PARAMS`) bounds a runaway pass.
- Stale `part` references replaced with `part_path.name` / `.stem`;
  `transcript_blocks_by_part` keys by `part_path.name`, which
  `SourcePart`/`part_for_timestamp`/verify's snippet lookup all agree on.
- `ensure_ollama_model_ready` for discovery warms via `OLLAMA_CHAT_URL`.

### Applied
Repository source, installed copy (`G:\pog_dev`), and public repo, all
byte-identical in the edited regions.

### Verification
Fresh full run on the 2026-08-25 12-21-38 VOD: discovery passes parse real
candidate rows again (debug dumps for empty responses no longer produced);
the pre-fix run had 6/6 passes on parts 1-2 return empty.

## Fix History: judge score-field redundancy + cross-pass discovery dedup (2026-09-07)

### Change
Redundancy-audit trims in `analyze_highlights_emotion.py`; no stage/checkpoint
format changes, existing checkpoints stay compatible.

- **Judge prompt fields (`run_judge_batch`)** — the prompt used to send
  `DiscoveryScore={Score}` (which by judge time already includes emotion +
  hype boosts) *and* `TranscriptScore={pre-boost score}`: the same underlying
  number twice under misleading names. It now sends one honest decomposition:
  `BaseScore={TranscriptScore}` (the raw pre-boost score; the field name is
  kept because the CSV column persists it) plus the explicit
  `EmotionBoost` and `HypePhraseBoost` fields. `JUDGE_INSTRUCTIONS` never
  referenced the input field names, so no prompt-text change was needed.
  The CSV columns are untouched.
- **Discovery dedup is timestamp-only** — `run_discovery` previously
  deduplicated on `(timestamp, title.lower())`, so the same moment proposed
  by two passes with differently-worded titles survived discovery and only
  merged later (audioscan's `merge_near_duplicates`, which requires title
  similarity ≥0.75 on top of the ±10 s window). The key is now the timestamp
  alone, keeping the highest-scored proposal: candidates anchor to real
  transcript block timestamps, so an exact-second collision across passes is
  the same moment. `merge_near_duplicates()` stays in audioscan — it still
  handles drifted-second collisions and audio-scan vs LLM overlaps.
- **Judge-stage final guard aligned** — `run_stage_judge`'s post-tournament
  dedup key is now timestamp-only too (the pool is timestamp-unique after the
  discovery change, so the old title component was dead weight).

### Applied
Repository source (`E:\VIAL\Pog_Engine_priv`) only; public repo and
`G:\pog_dev` install copy pending the usual sync.

### Verification
`py_compile` clean. Synthetic smoke tests (fake `ollama_generate`, no server):
judge prompt contains `BaseScore=` from the pre-boost value, no
`DiscoveryScore=`/`TranscriptScore=` labels, explicit `HypePhraseBoost=`;
discovery feed with two same-second different-title candidates + one adjacent
second collapses to the score-9 row and keeps the adjacent second (exact-key
semantics preserved).

## Fix History: Demucs-combined rename + false memory warnings (2026-09-07)

### Rename: `<name>_mic_combined.w64` → `<name>_mic_demucs_combined.w64`
The old name read as if the file were the final mic track; it is actually the
concatenated Demucs-separated (vocals-only) chunks, pre-render — `*_mic.wav`
is the later 16 kHz mono ffmpeg re-render Whisper consumes. Changed in
`make_extract_mic_bat_singletrack()` (generator + Step 1e echo),
`isolate_vocals.py`'s `--combined-path` default, the RunAll `1e` mini-step
description, the QA harness script line, and AGENTS.md. The output parser is
unaffected (it keys on isolate_vocals' "Combined separated mic audio:" print,
unchanged). **Compatibility:** generated bats pass `--combined-path`
explicitly, so already-generated Step 1 bats keep writing the old name and
still work; `migrate_legacy_vod_folder()` moves BOTH globs into the step 1
folder. Re-run the organizer to regenerate bats with the new name.

### False memory warnings: model sizing picked the wrong sibling
`[verify] WARNING: qwen3.5:9b-q4_K_M needs ~24.2 GB (22.2 GB weights)...`
fired on a 6 GB model. Root cause: `_ollama_model_size_gb()` returned on the
FIRST `/api/tags` entry matching `name == model or same family base`, and the
machine lists `qwen3.5:35b-a3b-q4_K_M` (22.2 GB) before `qwen3.5:9b-q4_K_M`
— iteration order decided which sibling answered. Fix: collect all sizes
into a dict, then one exact case-insensitive lookup (untagged names match
their `:latest`); no family fallback at all.

The memory pre-flight helpers moved to `pipeline_config.py`
(`memory_status_gb`, `free_vram_gb`, `ollama_model_size_gb`,
`model_memory_fit`, `MODEL_LOAD_HEADROOM_GB=2.0`) so the analyzer
(`ensure_ollama_model_ready`) and the configurator share one implementation;
the analyzer's private copies are gone.

### Configurator memory warning
`configure_models.py` now logs a memory-fit warning (same numbers/format as
the stage pre-flight) in its log box on "Scan models" and on every
MODEL/JUDGE_MODEL selection — ollama backend only, silent when the size is
unknown. An undersized pick now surfaces at selection time instead of as a
mid-run load failure.

### Applied
Repository source (`E:\VIAL\Pog_Engine_priv`): `pipeline_config.py`,
`analyze_highlights_emotion.py`, `configure_models.py`,
`OrganizeVODAndFixSRT_Emotion.py`, `isolate_vocals.py`, QA harness. Installed
copy (`G:\pog_dev`) synced for the five pipeline files.

### Verification
`py_compile` all six. Synthetic: sibling-listed-first returns 6.1 not 22.2,
case variants match, `q4_km` ≠ `q4_k_m` (different tag → None), no family
fallback, `:latest` resolution, fit math, genuine 35B shortfall still warns.
Live against the real Ollama: `JUDGE_MODEL` sizes 6.14 GB, required 8.1 GB vs
27.2 GB free → no warning; `qwen3.5:35b-a3b-q4_K_M` still sizes 22.2 GB.
Headless imports of analyzer + configurator clean.

## Fix History: AMD GPU support, 9070XT via ROCm torch + Vulkan LLM (2026-09-08)

### What 9070XT support means per stage (priv repo only, not synced)
No `pog_dev`/public sync yet, per request — all below is `E:\VIAL\Pog_Engine_priv` only.

- **Torch stages (Demucs vocal isolation + emotion model): GPU-accelerated.**
  AMD's ROCm PyTorch for Windows (public preview, Radeon 7000/9000 incl.
  gfx1201 = 9070XT) exposes the same `torch.cuda` API as the NVIDIA build, so
  `detect_device()`, the emotion `torch.device("cuda")` path, and demucs
  `--device cuda` work unchanged — only the installer needed a new branch.
- **LLM stages (Ollama default): zero code change.** The 9070XT is NOT on
  Ollama's Windows ROCm list (only 7000-series + PRO there), so it serves
  through Ollama's default-on Vulkan backend instead
  (`https://docs.ollama.com/gpu`). Same for the llamacpp backend: the Vulkan
  `llama-server` build uses the 9070XT under the existing `--n-gpu-layers`.
- **Transcription: CPU-bound on AMD (known limitation).** Pinned
  whisper.cpp v1.7.6 ships no Vulkan Windows binary (plain x64 / BLAS /
  cublas 11.8+12.4 only — asset list verified via the release API), so the
  setup downloads the OpenBLAS CPU build wherever nvidia-smi is absent.
  The cublas exe cannot even start there (loader 0xC0000135, no
  nvcuda.dll), so there is no "CPU fallback" — BLAS is the build.
  Correct but slower; revisit when upstream publishes a Vulkan Windows asset.
- **Memory pre-flight: RAM-only on AMD, deliberately.** `free_vram_gb()`
  stays nvidia-smi-only — the only other Windows source,
  `Win32_VideoController.AdapterRAM`, is a uint32 that cannot hold a 16 GB
  card, so any AMD "fallback" would be fiction. A 0.0 VRAM reading errs
  toward warning early (safe direction for an advisory check), and the load
  is attempted regardless with the retry chain intact.

### Implementation
- `pog_engine_setup.py`: `detect_amd_gpu()` (powershell CIM
  `Win32_VideoController` name match — wmic is gone from current Win11;
  consulted only when no NVIDIA GPU exists, mixed machines keep CUDA).
  `ensure_torch()` AMD branch installs the pinned ROCm SDK + torch wheels
  (`ROCM_WINDOWS_REL`, bump when AMD publishes newer) with `--no-cache-dir`;
  any failure falls back to CPU torch, never a broken env. ROCm wheels are
  **cp312-only**, so non-3.12 Pythons (user-brought older interpreters; fresh
  PCs get 3.12.10 from the launcher) get CPU torch plus a message pointing at Python 3.12 +
  Adrenalin 26.2.2+. `torch_status()` now also reports the cuda device name
  and `torch.version.hip`, surfaced in installer logs via `_torch_summary()`.
- `pipeline_config.py`: doc-only — `free_vram_gb()`/`model_memory_fit()`
  record the NVIDIA-only behavior; `VOCAL_ISOLATION_DEVICE` comment notes
  ROCm HIP answers the `torch.cuda` API.
- `isolate_vocals.py` / `analyze_highlights_emotion.py`: comments noting no
  AMD branch is needed; the emotion `[emotion] Device:` log now names the
  GPU (guarded — a name-query failure can never sink the stage).
- `OrganizeVODAndFixSRT_Emotion.py`: the 2a mini-step tooltip no longer
  claims CUDA (CPU-bound on AMD).

### Verification
`py_compile` all six modules. Throwaway mock suite 21/21 (deleted after):
AMD CIM parsing (found/absent/missing-powershell/nonzero-exit), probe key
extraction against stub torch builds (ROCm device+hip, CPU, name-query
failure — the last caught a real `try:`-after-`;` SyntaxError in the first
probe draft), ROCm URL pins (rel + cp312 + `%2B`), AMD py3.11→CPU message,
ROCm-failure→CPU fallback, installer success/failure paths, NVIDIA path
still tries CUDA tags and never ROCm URLs, `free_vram_gb` still reads on an
NVIDIA box. Live on the RTX 3080 workstation: probe reports
`2.5.1+cu121 / cuda True / GeForce RTX 3080 / hip None`, AMD undetected.
Real 9070XT verification (ROCm torch install, Demucs/emotion on HIP,
Ollama-Vulkan latency) still needs the AMD machine.

## Fix History: pipeline moves to Python 3.12 (2026-09-08)

### Why 3.12.10, not 3.12.14
3.12.14 is the newest 3.12 patch but ships **source-only** — per its release
page, 3.12 is in security-fixes-only stage and "binary installers are no
longer provided"; **3.12.10 was the last bugfix release with Windows
installers**. The launcher bootstrap pins 3.12.10.

### What changed (priv repo only, not synced)
- `Install_PogEngine.bat`: bootstrap downloads `python-3.12.10-amd64.exe`,
  post-install path `Python312\`, fallback hint says 3.12+. Existing PCs
  that already have a Python keep it — the bootstrap only fires when
  neither `python` nor `py` exists, so current 3.11.9 installs are untouched
  until their owners reinstall Python themselves.
- `pog_engine_setup.py`: `run_all_checks()` floor warning raised 3.10 →
  3.12, warning-only (older interpreters still run everything except ROCm
  torch). AMD-branch stale comment updated for the new bootstrap.
- `AGENTS.md` version references updated (bootstrap table, conventions,
  launcher entry, tooling floor).

### What did NOT need to change
- **No CUDA-tag filtering.** The assumed gap (cu118 lacks cp312) is wrong —
  the cu118 index carries cp312 win_amd64 wheels from torch 2.2.0 through
  2.7.1 (verified live), so the existing newest-first tag walk works on 3.12
  unmodified. All other tags (cu126–cu130) ship cp312 too.
- **No dependency changes.** torch (cp312 since 2.2), demucs,
  transformers, librosa, soundfile, safetensors, Pillow, numpy, requests all
  support 3.12; the repo uses no stdlib modules removed in 3.12 (no
  asyncore/asynchat/smtpd/imp/distutils, no `utcnow()`/`time.clock`).

### Verification
`py_compile` clean. Throwaway mock suite 4/4 (deleted after): floor warns
on 3.11 and stays silent on 3.12, AMD branch attempts the ROCm install on
3.12, bootstrap bat contains the 3.12.10 URL/path and no 3.11.9 remnant.

## Fix History: launcher enforces Python 3.12 (2026-09-09)

### Why enforce in the launcher
Moving the pipeline to 3.12 left a trap: the launcher kept any existing
interpreter, so a 3.11 + NVIDIA user who swapped to AMD (or just installed
3.12 alongside) would silently keep running 3.11 on CPU torch forever — the
setup warning alone never changed which `python` ran. Enforcement belongs
in the `.bat`, which owns interpreter selection.

### What changed (priv repo only, not synced)
- `Install_PogEngine.bat`: after locating `python`/`py`, a version gate
  parses `major.minor` and installs 3.12.10 whenever the found interpreter
  is missing or below 3.12. The download/install block is now a shared
  `:INSTALL_PY312` subroutine used by both the not-found and too-old paths;
  the old install is left untouched and the handoff always uses the explicit
  new `Python312\python.exe` path (immune to stale PATH in the same console).
  After upgrading past an old interpreter, a `choice` prompt (15 s default
  No) offers to open Installed apps so the user can uninstall 3.11 —
  recommended, never automatic.
- `pog_engine_setup.py`: AMD-branch message updated (launcher auto-installs
  3.12; the old-Python branch now only triggers on direct-script runs).

### Verification
Read-only gate-parse test against this box's real 3.11.9: `NEED=1`
(deleted after). A `^&`-joined compound `set` inside the `for` block died
with `& was unexpected at this time` on the first attempt — replaced with
two plain single-`set` loops. Full-bat flow review: subroutine
`call`/`exit /b` errorlevel propagation, no fall-through past the final
`exit`, delayed-expansion quoting on spaced paths. The actual
download/install path is unverified by design (it would install Python).

### Follow-up: bare parens killed the launcher on 3.11 boxes (same day)
First real run flashed and died: `echo Downloading and installing Python
3.12.10 (your existing Python is left untouched) ...` sits inside the
version-gate `if` block, and unquoted parens in echo text close the block
early — the leftover `...` then fails with `... was unexpected at this
time` before any `pause` runs. Reworded without parens (other block echos
audited clean; the quoted `choice` string was never at risk). Verified with
a neutered launcher copy (download/handoff stubbed, 5 probes deleted
after): 3.11.9 box flows gate → install → uninstall prompt → handoff to
the closing banner. The real download/install step itself remains
unexecuted by design.

## Fix History: requests user-site repair + exact-3.12 gate (2026-09-09)

### Symptom (AMD friend's box, live log)
`C:\Python314` (admin-owned, system site not writeable): pip fell back to
`--user`, reported success, then `import requests` failed in the same
process and the installer died. Same log also showed the second problem:
the friend was on 3.14.7, outside the validated 3.12 (ROCm wheels are
cp312-only), which the old `>= 3.12` gate waved through.

### Fix
- `ensure_requests()` escalates instead of dying: plain pip + import, then
  user-site repair (`site.addsitedir(getusersitepackages())` + retry),
  then an isolated `pip --target` into `TEMP\pog_engine_vendor` (never the
  repo, so syncs/status stay clean), then a diagnostics dump
  (`ENABLE_USER_SITE`, `PYTHONNOUSERSITE`, user-site dir) with an
  admin-rerun workaround. Instant workaround for the friend: re-run the
  launcher as administrator (system site writeable, user site uninvolved).
- Launcher gate tightened `>= 3.12` → `== 3.12` (string compare, more
  robust than the old numeric `LSS`): 3.11/3.13/3.14 all trigger the
  automatic 3.12.10 install now.

### Verification
`py_compile` clean. Throwaway suite 6/6 (deleted after): importable probe,
ensure short-circuit, diagnostics keys, repair no-crash, vendor
insert+import mechanics via stubbed pip (real `requests` restored after),
plus a gate matrix bat (deleted after): 3.11.9/3.13.0/3.14.7/empty/2.7.18
all NEED=1, only 3.12.10 passes.

## Fix History: field-verified 9070XT bring-up, 7 fixes (2026-09-11)

### Post-mortem first: the snapshot shipped a real bug, not a stale extract
The friend's `ImportError: cannot import name 'VOCAL_ISOLATION_DEVICE'`
survived a wipe + reinstall from the verified-good zip — because the
verified files were verified against each other, never imported. The
`VOCAL_ISOLATION_DEVICE = ...` definition line is missing from
`pipeline_config.py` (lost in an earlier refactor; only its comment
survived, which my AMD pass then polished without noticing). The 3080 box
never fired it because no single-track VOD ever reached Demucs there.
Lesson: hash-match proves the copy, only an import proves the code — every
smoke suite from here on imports the modules it touches
(`python -c "import isolate_vocals"` catches this instantly).

### Fixes (from the field agent's AMD_FIX.md, adapted to repo conventions)
- `pipeline_config.py`: restored `VOCAL_ISOLATION_DEVICE =
  os.environ.get("VOCAL_ISOLATION_DEVICE", "auto")` after
  `VOCAL_ISOLATION_MODEL` (their exact one-liner).
- `pog_engine_setup.py` ROCm installs: `--no-deps` on the torch wheels.
  With `--upgrade --force-reinstall`, pip resolved the wheels'
  `rocm==7.2.1` dep from PyPI (0.1.0 stub) and clobbered the just-installed
  SDK — every install silently fell back to CPU while reporting green.
  Transitive gaps are backfilled by the later demucs/package installs.
- Whisper build split: cublas on NVIDIA; OpenBLAS CPU build wherever
  nvidia-smi is absent (AMD/Intel/none). The cublas exe cannot start
  without nvcuda.dll (loader 0xC0000135), so CPU "fallback" was fiction.
  Same `Release/` layout, one extractor, new `WHISPER_*_URL` constants +
  BLAS sha pin (`adde1afb…`, digest from the release API). A present-but-dead
  exe (GPU-swap leftover) is wiped and replaced.
- `find_whisper_cli()` + `_whisper_exe_runs()` probe (`--help` must exit
  0): the installer can never bless a dead binary again.
- `OrganizeVODAndFixSRT_Emotion.py`: `_cli_path()` passes whisper `-f`/`-of`
  in `\\?\` verbatim form (UUID-nested VOD paths hit 265 chars; Python opens
  them fine, whisper's narrow fopen does not). `fix_srt()` /
  `split_srt_into_chunks()` mkdir their out dir; both generated Step-1 bats
  mkdir the step folder before ffmpeg writes.

### Verification
`import isolate_vocals` → `IMPORT-OK device: cuda` (the exact import that
died). Live BLAS branch test against TEMP (mocked AMD, real 16 MB
download): sha pin verified, extract + post-probe green; probe rejects a
text file and a missing path, accepts the real cublas exe. Path suite 6/6:
verbatim prefix, fix/split mkdir on missing dirs, both step-1 bats contain
mkdir ordered before ffmpeg. Field proof (their box, 5 h VOD): Demucs 17
min on 9070XT, whisper CPU 27 min, emotion 52 s on GPU, 50 finalists + EDL.

## Fix History: launcher re-downloaded Python 3.12.10 despite it being installed (2026-09-11)

### Symptom
Box had 3.12.10 installed, yet `Install_PogEngine.bat` forced a 3.12.10 re-download.

### Root cause
Detection accepted the first `python` on PATH unconditionally (`PY_CMD=python`),
then the exact-3.12 gate rejected it when PATH pointed at another version
(this box: PATH `python` = 3.11.9, 3.12.10 present via `py -3.12` and the
per-user install path). The launcher never searched for the 3.12 it already had.

### Fix (`Install_PogEngine.bat`, priv repo only, not synced)
Detection now prefers an existing 3.12 and downloads only when none is found:
1) `python` on PATH when it is already 3.12 (`:TRY_EXE` probe, which remembers
the first non-3.12 version for the install message), 2) exact-version
`py -3.12` launcher (unaffected by which Python is newest), 3) well-known
paths (`%LocalAppData%\Programs\Python\Python312`, `C:\Python312`,
`C:\Program Files\Python312`), 4) otherwise report the found version (or
not-found) and run the unchanged `:INSTALL_PY312` + uninstall-prompt path.
New `:PARSE_VER` helper owns the dotted-version split. Echo text stays
paren-free inside blocks per the 2026-09-09 lesson.

### Verification
Neutered probe copy (install/handoff stubbed, deleted after) on the 3.11-PATH +
3.12.10 box: no install branch taken, `Using Python: Python 3.12.10`, handoff
reached; `grep` confirmed zero install calls. Real download/install path
unexecuted by design.

## Fix History: Whisper repeat-loop auto-retry in Step 2 (2026-09-12)

### Symptom
`F:\OBS VOD\2026-08-25 12-21-38`: 11 min of audible speech (00:49:17-01:00:22)
missing from the fixed SRT. Raw chunk 2 (absolute 29:50-60:00) had collapsed
into a decoder repeat loop at local ~19:27: 59 of its last 64 captions were the
identical sentence. `fix_srt()` correctly deduped them to one row, exposing the
hole. 30-min chunking (2026-08-23) bounds this failure but does not prevent it.
A second report (`2026-09-03 12-33-42`, 02:01-03:35 empty) turned out to be a
genuinely muted mic: source `0:a:1` bit-exact zero, streamer says so on tape at
03:36:10. No code defect there, but it set the empty-chunk rule below.

### Fix (branch `whisper-loop-retry`, priv repo only, not synced)
Step 2 now checks each finished chunk SRT *before* stitching (post-`fix_srt`
the evidence is gone):
- `detect_transcription_loop()`: longest run of consecutive
  `normalize_repeated_sentence_key`-identical captions; flags at
  `TRANSCRIPTION_LOOP_MIN_REPEATS=10` spanning `TRANSCRIPTION_LOOP_MIN_SPAN_SECONDS=180`.
  Whole-chunk checks would miss this incident (chunk had 247 blocks, only the
  tail looped), so the detector is run-localized and reports the onset.
- Looped spans subdivide into overlapped halves (`_resolve_retry_span`),
  each re-transcribed by a fresh Whisper process; recursion bottoms out at
  `TRANSCRIPTION_RETRY_MIN_MINUTES=5` (30->15->7.5 max). Same-window retry is
  deliberately not used: the decode is near-deterministic, only a new window
  breaks the loop.
- Empty chunk SRTs consult `transcription_chunk_max_rms()` (wave+numpy, silent
  under `TRANSCRIPTION_SILENCE_RMS=0.002`): silent means muted mic, accepted;
  empty over live audio re-enters the retry path.
- Hard caps: extra Whisper runs <= `TRANSCRIPTION_RETRY_BUDGET_FACTOR=1.0` x
  chunk count; floor/budget exhaustion accepts best-effort and records it.
  Verdicts persist in `chunk_retries.json` (fingerprinted on chunk layout +
  all five knobs, so stale verdicts never survive a config change); human
  findings go to `transcription_warnings.json` (kinds: silent-span,
  loop-subdivided, loop-accepted, empty-accepted, budget-exhausted,
  span-unmeasurable). All five knobs are env-overridable + `EDITABLE_PARAMS`
  Transcription-stage entries.
- RunAll parser untouched: retry prints reuse the "Running Whisper for chunk"
  phrasing so the 2a card stays live; `[LOOP]`/`[SILENT]`/`[RETRY]` lines go to
  the console only. Generated bats unaffected (script-only change).

### Verification
Throwaway suites (deleted after): 10 assertions on the detector (real loop
shape flags with onset/run/span, 9x and chorus-length runs pass, normalized
case/punct variants match, silence/missing-file semantics, fingerprint
mismatch) plus a stubbed-whisper end-to-end (looped 1-chunk VOD subdivides
into 2 clean spans with correct absolute offsets, warnings + retries files
written, rerun reuses with zero Whisper calls). A real multi-hour
re-transcription still needs the VOD + whisper.cpp + GPU; the 08-25 VOD folder
has since been moved off F:, so field validation on the original chunk 2 is
pending its return.

## Fix History: judge final round exceeded JUDGE_NUM_CTX (2026-09-26)

### Symptom
`F:\OBS VOD\2026-09-09 12-22-54\step6_run_20260925_235813.log`: judge round 1
completed (100 -> 50 survivors), then the final call failed deterministically:
`request (8926 tokens) exceeds the available context size (6144 tokens)`.
Retries can never fix a context-size 400. Run stopped by user.

### Root cause
`run_judge_tournament()` batched round 1 by `JUDGE_BATCH_SIZE` but sent ALL
survivors in one final `run_judge_batch()` call. With the heavy preset
(`JUDGE_NUM_CTX=6144`, `JUDGE_BATCH_SIZE=10` for the dense 27B on a 10 GB
card), 50 survivors ~= 8900 prompt tokens > 6144. The preset's trimmed ctx is
correct for VRAM; the two-round tournament shape was the bug. Small VODs
never hit it (few survivors fit one call).

### Fix
Extra `_judge_halving_round()` rounds until survivors fit one judge call
(`analyze_highlights_emotion.py`; synced to `G:\pog_dev` + public repo).
Round-1 log line unchanged; later rounds log `Judging round N ...`.
Synthetic check: pools 5-126 x batch 10/20, no call ever exceeds batch size.

### Resume
Re-run `step5_analyze_highlights\5e_Judge.bat` (uses the installed copy);
verify/emotion checkpoints are reused, nothing earlier re-runs.

## Fix History: markers quoted tail text at head time in sparse VODs (2026-09-26)

### Symptom
`F:\OBS VOD\2026-09-12 12-38-30`: `[Emotion] what the fuck just happened`
marker at 1:56:01, but the words are spoken at 01:57:26 in the fixed SRT
(85 s off). Neighbouring ranks were exact; the bad ones clustered in sparse
later splits.

### Root cause
`merge_entries_for_analysis()` built one thought block from
`01:56:01 i'm just gonna let it freeze...` through `01:57:26 what the fuck
just happened` (98 s, 27 words - sparse speech never tripped the 30-word or
2.5 s-gap closers) and stamped it with the head caption's time. Discovery
copied that head timestamp with the tail text; the <=15 s anti-hallucination
check passed (block start is a real timestamp) and verify/judge saw the same
smeared snippet, so nothing downstream could catch it. Same shape behind
`he ulted me` at 04:09:17 (true 04:11:22). Quiet singing spans (8.5 min at
20x, 5.8 min at 33x) kept the cap from closing every block at 30 s.

### Fix
New `TRANSCRIPT_MERGE_MAX_SPAN_MS = 30_000`: a block also closes when the next
caption would stretch its audio span past 30 s
(`OrganizeVODAndFixSRT_Emotion.py`; synced to public repo + `G:\pog_dev`).
The incident block now starts 01:57:12, 14 s from the true line instead of
85 s. Script-only change: no bat regeneration, no checkpoint format change.

### Resume
Re-run that VOD's `4_SplitSRT.bat`, then `5_AnalyzeHighlights.bat --stage`
discovery onward (or all of step 5); earlier checkpoints are reused, nothing
before step 4 re-runs.

## Fix History: whisper.cpp VAD smeared timestamps over silence (2026-09-29)

### Symptom
`F:\OBS VOD\2026-09-12 12-38-30`: fixed #238 starts 1:35:37 but speech starts
1:35:40, merging "just go mid" (1:35:46), "barrico challenge" (1:36:19),
"freestyling" (1:36:24) into one 50 s caption; #258 spans 1:43:56-1:46:06 for
a 2 s utterance at 1:46:02. Both already smeared in the raw chunk SRT, so
stitching/`fix_srt`/split only propagated them.

### Root cause
`_transcribe_audio_chunk()` passed whisper.cpp `--vad` unconditionally.
VAD concatenates detected speech (chunk 4: 29120000 -> 5079680 samples, 82.6%
silence excised), decodes one contiguous buffer, then stretches captions back
across the gaps: one caption per ~11 s compressed-speech span remaps onto
minutes of wall audio (267/471 chunk blocks >12 s, worst 346 s; VAD segments
themselves never exceed ~10 s). Mic RMS confirms silence where the long
captions claim speech (0.0 across 1:44-1:45); the VAD segment starts
themselves land within 1 s of true speech, only the emitted spans are fiction.

### Fix
`--vad` and its silero/threshold flags are deleted from
`_transcribe_audio_chunk()` - wall-clock decode, no VAD pre-segmentation at
all. The Step 2 bat echo drops "and VAD". `vad_removed: True` joins the retry
fingerprint so VAD-era verdicts invalidate, and `_clear_vad_era_srts()`
deletes chunk SRTs from before the removal (any stored fingerprint lacking
the marker) while reusing chunk audio; after the first post-removal run it
is a no-op. `WHISPER_VAD` stays downloaded/patched (harmless; `patch_paths()`
warns instead of failing when a constant is absent, so its future removal
needs no setup change). Script-only change: no bat regeneration, no
checkpoint format change.

### Resume
Re-run the VOD's `2_TranscribeAudio.bat` (Step 1 chunks reused; stale VAD-era
chunk SRTs auto-cleared), then steps 3-5 normally.

## Fix History: wall-clock decode hallucinations + retry-budget holes (2026-10-06)

### Symptom
`F:\OBS VOD\2026-09-20 12-47-24\step6_run_20261005_234908.log`: chunks 6-8
looped ("Thank you" chains, repeated phrases), and the retry budget (1.0x
chunk count = 8 extra runs) was spent on chunk 6's subdivision - chunks 7
and 8 then hit `budget exhausted, leaving span hole` on BOTH halves, so
~50 min of audio (02:56:20 onward) has no transcript at all in the fixed SRT
(ends at 02:56:19). Discovery found 0 candidates on parts 1-3 (hallucinated
SRT text has no real content to rank). Same run on an earlier VOD generation
(VAD-era) did not show this: VAD excised silence before decode, hiding the
silence-hallucination path the 2026-09-29 wall-clock removal exposed.

### Root causes
1. `--no-speech-thold 0.7` too permissive: wall-clock decode feeds silence
   straight to the decoder, and 0.7 lets it transcribe dead air as confident
   speech (upstream guidance for silence-heavy audio: 0.2-0.5). No timestamp
   smear anymore (fix held), but silence becomes phantom words instead.
2. `--best-of 5` wasted GPU: best-of diversifies greedy sampling only, so at
   temperature 0 + beam 5 it ran 5 identical decodes per chunk.
3. `budget-exhausted` left transcript HOLES: `continue` skipped the span
   entirely (no WAV decode, no entry in resolved), so stitched SRT jumps
   over the audio. 1.0x budget too small for a pathological VOD (chunk 6
   alone ate it via 2-level subdivision).

### Fix
- `_transcribe_audio_chunk()`: explicit `--temperature 0`, `--best-of 1`
  (was 5), `--no-speech-thold 0.5` (was 0.7). Beam 5, entropy 2.6, logprob
  -0.8, fallback ON all unchanged (loop recovery on real repetitive speech).
- `_resolve_retry_span()`: budget-exhausted spans now decode ONCE and are
  accepted with a warning (transcript flagged for review, but present) -
  no more holes. Budget raise covers the subdivision depth: pathological
  VODs need ~2x to finish all flagged chunks.
- `pipeline_config.py`: `TRANSCRIPTION_RETRY_BUDGET_FACTOR` 1.0 -> 2.0.
- Retry fingerprint gains `decode_flags` generation marker
  ("temp0-bs5-bo1-et2.6-lpt-0.8-nst0.5"); `_clear_stale_model_srts()`
  generalized to `_clear_stale_decode_srts()` (fires on model OR flag
  change). Old SRTs decoded under 0.7/bo5 auto-clear once; chunk audio reused.
  Bump the marker string on any future flag change or old SRTs silently reuse.

### Not changed
Silero VAD stays out (timestamp smear was worse than hallucinations); no
`--no-context` (kills cross-chunk continuity for marginal gain);
`--suppress-nst` kept as-is (does not cause repetition loops).

### Applied
Repository source + `G:\pog_dev` install copy (both files, byte-identical,
py_compile clean). Script-only change: no bat regeneration, no checkpoint
format change. Synthetic checks: zero-budget span decodes once and stitches
(no hole); old-flag SRTs clear once; current-flag SRTs survive.

### Resume
Re-run that VOD's `2_TranscribeAudio.bat` (Step 1 chunks reused; stale SRTs
auto-cleared), then steps 3-5 normally.
