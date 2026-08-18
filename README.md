# Pog Engine

**Pog Engine** is an AI-powered pipeline for analyzing virality in OBS or stream VODs. It transcribes your content, finds potential viral moments using **offline LLMs, speech emotion recognition, and audio analysis**, then exports ranked highlights as **DaVinci Resolve timeline markers**.

Finish your stream, run Pog Engine, and get your best moments without scrubbing through hours of footage.

**100% offline. Your data never leave your PC and will not be used to train AI.**

Built by solo content creator, for solo content creators & editors.

> **Note:** Pog Engine currently cannot detect purely visual moments like physical comedy or crazy gameplay.



## PC Requirement

- NVIDIA GPU (8GB+ VRAM, 10GB recommended)
- CUDA 12.4 compatible
- Ollama capable of running:
  - `qwen3:8b`
  - `qwen3.5:9b-q4_K_M`
  - `qwen3.6:35b-a3b` 
  >*OPTIONAL, This model is a lot smarter and slower than qwen3.5 but will require some RAM overflow if you have a 10gb card like me,


Built around an RTX 3080 (10GB VRAM). Larger models may work better on GPUs with more VRAM. Qwen performed best in my testing.

## OBS Setup Requirement FOR LOCAL RECORDED VOD

If you are using VODs downloaded from a streaming platform, you can skip this step.

Pog Engine was written to work best with a locally recorded VOD with separated audio channels.

Your OBS recording **must match the required configuration exactly**:

> <img width="1920" height="1404" alt="pog" src="https://github.com/user-attachments/assets/78af85a5-0b44-4fd7-8151-d6033ab1d802" />


## 1. Installating Ollama:

Install [Ollama](https://ollama.com/download/windows)

After finish installing Ollama, LOGIN NOT REQUIRED.

Open Command Prompt and type

Download qwen3:8b : ```ollama run qwen3:8b ```

Download qwen3.5:9b-q4_K_M : ```ollama run qwen3.5:9b-q4_K_M```

>Download qwen3.6:35b-a3b (OPTIONAL)```ollama run qwen3.6:35b-a3b```


## 2. Setting up Pog_Engine

<img width="916" height="401" alt="{B07D9456-F90D-4832-BBBF-039A72DAFAAB}" src="https://github.com/user-attachments/assets/381e916a-d8f8-4c61-9d0b-93625fa20813" />
 
Code -> Download ZIP

1. Extract ZIP and store the files in ```Pog_Engine``` folder
2. Run **Install_PogEngine.bat**
3. Click browse and select ```Pog_Engine``` folder
4. Then **Start Setup**

Your final result should say OK / Already Downloaded

<img width="845" height="703" alt="{7B6681C9-FB82-46E4-A084-B8C94ABC8A03}" src="https://github.com/user-attachments/assets/f203b77d-9957-4266-9eb5-c77d891d2565" />


*Side note: You can put whatever you want in gallery, I added this so I could use my fanart as screensaver while waiting for it to finish.

## How to use:

1. Place **Drag MP4 on me** shortcut in your **VOD folder**
3. Drop your VOD.mp4 onto the shortcut
4. Go to the created folder
5. Open **6_RunAllSteps.bat**
6. Watch it works

Files you'll need for Davinci Resolve:
VOD.mp4
VOD**fixed**.srt (must have fixed in name)
highlights.edl (your highlight markers)

You are welcomed to use any marker conversion tool to convert Davinci Resolve markers to use in other programs.

## Configure Pog Engine (ADVANCED USER ONLY)
I made the default settings for Pog Engine to work on all machines, this configurator tool is more for advanced power users who want to tweak their parameters.

Launch **ConfigurePogEngine.bat**

You can change the presets model I have written to use on my own machine and I know will work on machines with similar spec.

## Companion tools (COMING SOON): 
These are scripts that I wrote to speed up your editing process, you can buy the full pack here:

1-Click Import 
>Import needed files and create a a folder with your VOD.mp4, VOD.srt (transcription), highlights_markers.edl, and automatically populate a timeline with needed items (transcription need to be imported manually)

Marker Tracker
>View your markers in order and categories

Subtitle to Marker 
>Turn keywords in transcription into markers to find words you say a lot during hype moments like "nice!", this feature is already included in Pog Engine, this script is here just in case the AI misjudged your hype moment so you can manually find these moments yourself.

Send Markers from Timeline to Clip
>Send marker from timeline to clip so when you move the clip to another timeline to edit, the marker follows.

## Known Issues:
>none at the moment.
Please report any issues to the issues tab.

## ROADMAP:
1. Auto clips extraction, cut the middle man and get the clips immediately to use

## Tech Stack

- Python 3.11
- Ollama
- Whisper.cpp
- FFmpeg
- PyTorch
- DaVinci Resolve
