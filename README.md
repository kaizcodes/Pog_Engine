# Pog Engine

**Pog Engine** is an AI-powered local VOD processing pipeline for streamers. It automatically transcribes your recordings, detects potential viral moments using LLMs, speech emotion recognition, and audio analysis, then exports ranked highlights and DaVinci Resolve timeline markers.

Designed to eliminate hours of manually scrubbing through VODs.

## Features

- 🎤 Whisper.cpp transcription
- 🧠 Multi-stage AI highlight detection
- 😊 Speech emotion scoring
- 🔊 Audio-based highlight discovery
- ✅ AI verification & ranking
- 🎬 DaVinci Resolve EDL/CSV export
- ♻️ Resume interrupted runs with checkpoints

## Requirements

This project **only supports locally recorded OBS VODs**.

To work correctly, your OBS recording **must match the required configuration exactly**, including:

- Audio track layout
- Separate microphone track
  
See the examples below before using the pipeline.

> 📷 **Required OBS Recording Settings**
>
> <img width="1920" height="1080" alt="pog" src="https://github.com/user-attachments/assets/cd133146-dd73-4d05-af08-5431ae676620" />

## Installation:
Things to download: Save these in a single folder somewhere safe on your PC

[Whisper Large V3 for whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
>Download: ggml-large-v3.bin

[Whisper Voice Activity Detection (VAD)](https://huggingface.co/ggml-org/whisper-vad/tree/main)
>Download: ggml-silero-v6.2.0.bin

[Speech Emotion Recognition with Whisper](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3)

>You will need to download:
>- model.safetensor (RENAME TO: ```speech-emotion-recognition-with-openai-whisper-large-v3.safetensors``` )
>- config.json
>- preprocessor_config.json

Folder should look like this **SAVE IT IN A SAFEPLACE**:
<img width="798" height="228" alt="{648815AD-B343-4676-AC2A-91386056531E}" src="https://github.com/user-attachments/assets/69daba88-a443-4784-add3-20a785bb6230" />


## Setup:
1.
 
<img width="928" height="402" alt="{158D0DD9-B9C9-45C4-88EB-D38A4BDDEDDC}" src="https://github.com/user-attachments/assets/23095123-e483-4075-b655-a7499036f2ea" />

Download ZIP

2. Extract and store the files in a safe location on your system
3. Run first_run.py
4. Create shortcut from OrganizeVODAndFixSRT_py.bat (Right click -> Create shortcut)
5. Put this shortcut in your VOD folder

Using:
1. Drop your VOD.mp4 onto the shortcut
2. Then run **6_RunAllSteps.bat**
3. Watch it works

## Companion tools (RECOMMENDED):
[1 click script to import](https://github.com/kaizcodes/davinci-resolve-20-auto-scripts/tree/main/1%20Click%20Import%20%2B%20Timeline%20%2B%20SRT%20%2B%20Marker%20into%20Timeline) 
>import needed files and create a a folder with your VOD.mp4, VOD.srt (transcription), highlights_markers.edl, and automatically populate a timeline with needed items (transcription need to be imported manually)

[Highlight Marker Tracker](https://github.com/kaizcodes/davinci-resolve-20-auto-scripts/tree/main/Marker%20Tracker%20Ordered%20by%20Score)
>view your markers in score order

[Subtitle to Marker](https://github.com/kaizcodes/davinci-resolve-20-auto-scripts/tree/main/Subtitle%20to%20Marker) 
>find turn keywords in transcription into markers to find words you say a lot during hype moments like "nice!"

## Tech Stack

- Python
- Ollama
- Whisper.cpp
- FFmpeg
- PyTorch
- DaVinci Resolve
