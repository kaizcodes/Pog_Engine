---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

Feel free to skip any step that does not apply. Giving me as much info as possible is better to diagnose and fixing whatever issues you are running into.

## 1. What happened?

**What were you trying to do?** (e.g. "ran 6_RunAllSteps.bat on a 3-hour VOD")

**What went wrong?** (error message, wrong output, hang — be specific)

**Which step failed?** (1_ExtractMicAudio / 2_TranscribeAudio / 3_FixSRT / 4_SplitSRT / 5_AnalyzeHighlights / 6_RunAllSteps GUI / Install_PogEngine / ConfigurePogEngine / other)

**Did it fail consistently?** (every run / first time / only on this VOD)

## 2. Your system

- **Windows version:** (e.g. Windows 11 Pro 24H2)
- **Python version:** run `python --version` in a terminal →
- **GPU:** (e.g. NVIDIA RTX 3080 10GB / none)
- **Install folder:** (the folder where you ran Install_PogEngine.bat, e.g. `G:\pog_dev`)
- **Ollama version:** run `ollama --version` →
- **Ollama models pulled:** run `ollama list` and paste the output

## 3. Your VOD

- **Source:** local OBS recording (multi-track) or Twitch download (single track)?
- **Length:** (e.g. 7h34m)
- **File size:**
- **Audio streams:** if you know (the VOD folder's `vod_audio_info.json` says this)

## 4. Logs and files to attach

Attach these from your VOD folder (or install folder where noted). Zip them if there are many:

- [ ] `step6_run_*.log` — from the VOD folder (the RunAll GUI log; the most important file)
- [ ] `log_<stage>.txt` for the failing stage — e.g. `log_discovery.txt`, `log_verify.txt`
- [ ] `pipeline_stats.json` — from the VOD folder
- [ ] `run_info.json` — from the VOD folder
- [ ] `vod_audio_info.json` — from the VOD folder
- [ ] The exact console/bat window output if the failure was outside the GUI (copy-paste as text)

Also tell us which of these exist in the VOD folder (just the file names, no contents needed unless asked):

- [ ] `*_mic.wav` (mic audio was produced)
- [ ] `transcript_partN.txt` files (how many?)
- [ ] `checkpoint_discovery.json` / `checkpoint_audioscan.json` / `checkpoint_emotion.json` / `checkpoint_verify.json` / `checkpoint_judged.json` (which ones?)
- [ ] `top<N>_highlights.csv` (was any output produced?)

For install/setup problems, instead attach the full `pog_engine_setup.py` output (console or GUI log).

## 5. What did you already try?

- [ ] Re-ran the failing step
- [ ] Ran a debug sub-step (5a–5f) — which one, and what happened?
- [ ] Restarted Ollama
- [ ] Re-ran Install_PogEngine.bat
- [ ] Other:

## 6. Anything else?

Free text — anything odd you noticed, recent changes to your setup, etc.
