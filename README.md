# Pog Engine

**Pog Engine** is an AI-powered local VOD processing pipeline for streamers. It automatically transcribes your recordings, detects potential viral moments using LLMs, speech emotion recognition, and audio analysis, then exports ranked highlights and DaVinci Resolve timeline markers.

Designed to eliminate hours of manually scrubbing through VODs.

Finish stream, Click Run, and come back to your PC with a list of potential viral moments!

## PC Requirement

- NVIDIA GPU
- With more than 8GB VRAM (10GB Recommended) 
- Must be able to run CUDA 12.4 (GTX 1080 TI-> RTX 5090)
- Will not work with AMD GPU (maybe someone can write a conversion script, I don't own an AMD GPU)
- 30m-1hr of your time (on 3080)

I wrote this project to work with my RTX 3080 10GB VRAM

You'll need to be able to fit qwen3.5:9b with quantization(q4_K_M in this case)

IF you have more VRAM you can explore bigger parameter models, I specifically chose this model because it's the best model I can fit on my PC that gives me good result. and I've done a lot of research and result comparing, I found that qwen usually return the better results. Feel free to recommend if you find better result on other models.

## OBS Setup Requirement

This project **only supports locally recorded OBS VODs FOR NOW**.

To work correctly, your OBS recording **must match the required configuration exactly**:

> 📷 **Required OBS Recording Settings**
>
> <img width="1920" height="1404" alt="pog" src="https://github.com/user-attachments/assets/78af85a5-0b44-4fd7-8151-d6033ab1d802" />


## 1. Installating Ollama:

Install [Ollama](https://ollama.com/download/windows)

Open Command Prompt and type

```ollama run qwen3:8b ```

To download qwen3:8b

```ollama run qwen3.5:9b-q4_K_M```

To download qwen3.5:9b-q4_K_M

## 2. Setting up Pog_Engine

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

<img width="916" height="401" alt="{B07D9456-F90D-4832-BBBF-039A72DAFAAB}" src="https://github.com/user-attachments/assets/381e916a-d8f8-4c61-9d0b-93625fa20813" />
 
Code -> Download ZIP

1. Extract ZIP and store the files in ```Pog_Engine``` folder
2. Run Install_PogEngine.bat
3. Copy the path to ```Pog_Engine``` folder
4. Everything has to say ```OK```
5. If something is missing you need to follow each step again carefully
6. Create shortcut from OrganizeVODAndFixSRT_Emotion.bat (Right click -> Create shortcut)
7. Put this shortcut in your VOD folder

Your final result should look like this

<img width="833" height="683" alt="{D3F7F48C-B080-4423-9907-96503B98168F}" src="https://github.com/user-attachments/assets/2a339fdc-9839-4494-9cad-100632833451" />

*Side note: You can put whatever you want in gallery, I added this so I could use my fanart as wallpaper while waiting for it to finish.

## How to use:

If you have followed step by step correctly, at this point everything SHOULD work.

1. Drop your VOD.mp4 onto the shortcut
2. Go to the created folder
3. Then run **6_RunAllSteps.bat**
4. Watch it works

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
