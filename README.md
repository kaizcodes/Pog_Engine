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

## OBS setup Requirement

This project **only supports locally recorded OBS VODs FOR NOW**.

To work correctly, your OBS recording **must match the required configuration exactly**:

> 📷 **Required OBS Recording Settings**
>
> <img width="1920" height="1080" alt="pog" src="https://github.com/user-attachments/assets/cd133146-dd73-4d05-af08-5431ae676620" />

## Installating Ollama:

Install [Ollama](https://ollama.com/download/windows)

Open Command Prompt and type

```ollama run qwen3:8b ```

To download qwen3:8b

```ollama run qwen3.5:9b-q4_K_M```

To download qwen3.5:9b-q4_K_M

## Setting up Pog_Engine

Pick a safe location on your PC and create a folder name "Pog_Engine"

Inside said folder create a folder called ```models```, download and put these files in ```models```:

[Whisper Large V3 for whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
>Download: ggml-large-v3.bin

[Whisper Voice Activity Detection (VAD)](https://huggingface.co/ggml-org/whisper-vad/tree/main)
>Download: ggml-silero-v6.2.0.bin

[Speech Emotion Recognition with Whisper](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3)

>Download:
>- model.safetensor (RENAME TO: ```speech-emotion-recognition-with-openai-whisper-large-v3.safetensors``` )
>- config.json
>- preprocessor_config.json

Folder should look like this:
<img width="895" height="203" alt="{99BD0930-C0B9-42DC-ACFE-4DC138D151DD}" src="https://github.com/user-attachments/assets/5e2ec62a-8243-4de1-90a4-0958cd3b57a3" />

Return to ```Pog_Engine``` folder
Download and unzip

[Whisper cubLAS 12.4.0](https://github.com/ggml-org/whisper.cpp/releases)

Now there should be 2 folders in your ```Pog_Engine``` folder

<img width="262" height="198" alt="{4B5A1942-E102-4BBC-BD9E-0B2F492DA4DE}" src="https://github.com/user-attachments/assets/e48f8adb-061c-4480-8f91-9d6dae5538f6" />


## Pog_Engine Installation:
 
<img width="928" height="402" alt="{158D0DD9-B9C9-45C4-88EB-D38A4BdfdfDDEDDC}" src="https://github.com/udfdfser-attachments/assets/23095123-e483-407fgfg5-b655-a7499036f2ea" />

Download ZIP

1. Extract ZIP and store the files in ```Pog_Engine``` folder
2. Run installation.bat
3. Create shortcut from OrganizeVODAndFixSRT_py.bat (Right click -> Create shortcut)
4. Put this shortcut in your VOD folder

## How to use:

If you have followed step by step, at this point everything SHOULD work.

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
