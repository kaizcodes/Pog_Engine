# Pog Engine

Pog Engine is a Windows pipeline that turns an OBS or Twitch VOD into ranked,
audio/transcript-based highlights and a DaVinci Resolve marker EDL.

By default, analysis runs locally with Whisper.cpp, Ollama, audio signal
processing, and speech-emotion scoring. Setup downloads dependencies and model
assets; the pipeline does not analyze video frames, so purely visual moments
are out of scope.

## Requirements

- Windows and Python 3.10+ (`Install_PogEngine.bat` can install Python 3.11.9)
- FFmpeg **and ffprobe** on `PATH`
- [Ollama](https://ollama.com/download/windows) with the default models:

  ```text
  qwen3:8b
  qwen3.5:9b-q4_K_M
  ```

- An NVIDIA GPU is recommended; CPU fallback is slower.

## Setup

1. Install Ollama and pull the models above.
2. Run `Install_PogEngine.bat` from the project folder. It installs/checks
   Python packages, downloads the required Whisper/emotion/separation assets,
   configures local paths, and creates `Drag MP4 on me.lnk`.
3. Optionally run `ConfigurePogEngine.bat` to change models or pipeline
   settings.

## Run

1. Copy `Drag MP4 on me.lnk` into the folder containing your VOD.
2. Drop an `.mp4` or `.mkv` onto the shortcut. Pog Engine creates a folder
   named after the VOD and generates the numbered helper scripts. It detects
   separate OBS audio tracks automatically; single-track VODs use voice
   isolation.
3. Double-click `6_RunAllSteps.bat`.

The pipeline extracts or isolates the mic audio, transcribes it with
Whisper.cpp, cleans and chunks the SRT, then:

1. discovers transcript candidates with Ollama;
2. scans the full audio for energetic moments;
3. scores speech emotion;
4. verifies and ranks candidates; and
5. exports the results.

Each analysis stage checkpoints its output, so rerunning resumes completed
work. Use `5_AnalyzeHighlights.bat` to run only the analysis stage after the
first four steps are complete.

## Outputs

The VOD folder contains the processed audio/transcript files plus:

- By default, `top50_highlights.csv` — ranked highlights and metadata
- By default, `top50_markers.edl` — import into DaVinci Resolve as timeline markers
- `run_info.json` — models, settings, and run statistics

`TOP_N` and other pipeline settings can be changed with
`ConfigurePogEngine.bat`.
