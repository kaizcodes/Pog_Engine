# Pog Engine

**Pog Engine** is an AI-powered pipeline for processing **locally recorded OBS VODs**. It transcribes your recordings, detects potential viral moments using LLMs, speech emotion recognition, and audio analysis, then exports ranked highlights and DaVinci Resolve timeline markers. Designed to save hours of manually scrubbing through VODs. 

Finish stream, Run, and come back to your PC with a list of potential viral moments!

## PC Requirement

- NVIDIA GPU (8GB+ VRAM, 10GB recommended)
- CUDA 12.4 compatible
- Ollama capable of running:
  - `qwen3:8b`
  - `qwen3.5:9b-q4_K_M`

Built around an RTX 3080 (10GB VRAM). Larger models may work better on GPUs with more VRAM. Qwen performed best in my testing.

## OBS Setup Requirement

This project **only supports locally recorded OBS VODs FOR NOW**.

Your OBS recording **must match the required configuration exactly**:

> <img width="1920" height="1404" alt="pog" src="https://github.com/user-attachments/assets/78af85a5-0b44-4fd7-8151-d6033ab1d802" />


## 1. Installating Ollama:

Install [Ollama](https://ollama.com/download/windows)

Open Command Prompt and type

```ollama run qwen3:8b ```

To download qwen3:8b

```ollama run qwen3.5:9b-q4_K_M```

To download qwen3.5:9b-q4_K_M

## 2. Setting up Pog_Engine

<img width="916" height="401" alt="{B07D9456-F90D-4832-BBBF-039A72DAFAAB}" src="https://github.com/user-attachments/assets/381e916a-d8f8-4c61-9d0b-93625fa20813" />
 
Code -> Download ZIP

1. Extract ZIP and store the files in ```Pog_Engine``` folder
2. Run Install_PogEngine.bat
3. Click browse and select ```Pog_Engine``` folder

Your final result should look like this

<img width="833" height="683" alt="{D3F7F48C-B080-4423-9907-96503B98168F}" src="https://github.com/user-attachments/assets/2a339fdc-9839-4494-9cad-100632833451" />

*Side note: You can put whatever you want in gallery, I added this so I could use my fanart as wallpaper while waiting for it to finish.

## How to use:

If you have followed step by step correctly, at this point everything SHOULD work.

1. Create shortcut from OrganizeVODAndFixSRT_Emotion.bat (Right click -> Create shortcut)
2. Place this shortcut in your VOD folder
3. Drop your VOD.mp4 onto the shortcut
4. Go to the created folder
5. Open **6_RunAllSteps.bat**
6. Watch it works

## Companion tools (RECOMMENDED):
[1 click script to import](https://github.com/kaizcodes/davinci-resolve-20-auto-scripts/tree/main/1%20Click%20Import%20%2B%20Timeline%20%2B%20SRT%20%2B%20Marker%20into%20Timeline) 
>Import needed files and create a a folder with your VOD.mp4, VOD.srt (transcription), highlights_markers.edl, and automatically populate a timeline with needed items (transcription need to be imported manually)

[Highlight Marker Tracker](https://github.com/kaizcodes/davinci-resolve-20-auto-scripts/tree/main/Marker%20Tracker%20Ordered%20by%20Score)
>View your markers in score order

[Subtitle to Marker](https://github.com/kaizcodes/davinci-resolve-20-auto-scripts/tree/main/Subtitle%20to%20Marker) 
>Find turn keywords in transcription into markers to find words you say a lot during hype moments like "nice!"


## Tech Stack

- Python
- Ollama
- Whisper.cpp
- FFmpeg
- PyTorch
- DaVinci Resolve
