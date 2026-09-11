# Pog Engine

**Pog Engine** is an AI-powered pipeline for analyzing virality in OBS or stream VODs. It transcribes your content, finds potential viral moments using **offline LLMs, speech emotion recognition, and audio analysis**, then exports ranked highlights as **DaVinci Resolve timeline markers**.

Finish your stream, run Pog Engine, and get your best moments without scrubbing through hours of footage.

**100% offline. Your data never leave your PC and cannot be used to train AI.**

Built by solo content creator, for solo content creators & editors.

> **Note:** Pog Engine currently cannot detect purely visual moments like physical comedy or crazy gameplay.



## PC Requirement

- NVIDIA GPU (8GB+ VRAM, 10GB recommended)
- CUDA 12.4 compatible
- Ollama capable of running:
  - `qwen3.5:9b-q4_K_M`
  - More models in testing below, qwen3.5:9b should suffice on most machines.

Built around an RTX 3080 (10GB VRAM). If you have more VRAM you can try using larger and smarter models.

## OBS Setup Requirement FOR LOCAL RECORDED VOD

If you are using VODs downloaded from a streaming platform, you can skip this step.

Pog Engine was written to work best with a locally recorded VOD with separated audio channels.

Your OBS recording **must match the required configuration exactly**:

> <img width="1920" height="1404" alt="pog" src="https://github.com/user-attachments/assets/78af85a5-0b44-4fd7-8151-d6033ab1d802" />


## 1. Installating Ollama:

Install [Ollama](https://ollama.com/download/windows)

After finish installing Ollama, LOGIN NOT REQUIRED.

Open Command Prompt and type

Download qwen3.5:9b-q4_K_M : ```ollama run qwen3.5:9b-q4_K_M```

## 2. Setting up Pog_Engine
Get PogEngine.zip here
https://github.com/kaizcodes/Pog_Engine/releases/tag/Release

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
5. Open **Run_Pog_Engine.bat**
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

You can adjust whisper parameters if you know the parameters works for you specifically.

## Updating Pog Engine:

If you are on v.1, please remove everything inside of your current install of Pog Engine.

Run **Update_PogEngine.bat** in your Pog Engine folder. It checks the latest
GitHub release, and if yours is older it downloads the release ZIP and
replaces **only the code files that actually changed** (compared by hash).
Your models, VOD folders, gallery, and run histories are never touched, and
your ConfigurePogEngine tunings plus your machine's paths (whisper/model
folders) are carried into the new files automatically. Replaced originals are
kept in `update_backup_<release>_<date>/` so you can undo by copying them back.
Useful flags: `Update_PogEngine.bat --check-only` (just show versions),
`--ref v2.0.1` (update to a specific release), `--yes` (skip the confirm
prompt), `--force` (re-apply even when versions match).

## Companion tools (COMING SOON): 
These are scripts that I wrote to speed up your editing process, you can buy the full pack here:

1-Click Import 
>Import needed files and create a a folder with your VOD.mp4, VOD.srt (transcription), highlights_markers.edl, and automatically populate a timeline with needed items (transcription need to be imported manually)

Send Markers from Timeline to Clip
>Send marker from timeline to clip so when you move the clip to another timeline to edit, the marker follows.

Marker Tracker
>Instead of using Index, you can move this window around to view your markers in order and categories

Subtitle to Marker 
>Turn keywords in transcription into markers to find words you say a lot during hype moments like "nice!", this feature is already included in Pog Engine, this script is here just in case the AI misjudged your hype moment so you can manually find these moments yourself.


## Known Issues:
Please report any issues to the [issues tab](https://github.com/kaizcodes/Pog_Engine/issues).

- Step 5 is unbearably slow.
> Open Task Manager and see if your **GPU** is 100% usage, if it says CPU then you need to reinstall Ollama

## If you wish to support me
You can buy me a coffee here: https://ko-fi.com/kaizuchaneru
or check out my Twitch: https://www.twitch.tv/kaizuchaneru

The money will be put into development and maintaining this project and any spare change will be donated to local orphanages.

## TESTING OTHER MODELS:

You can just change them in the configurator if you have it installed via Ollama

Currently in testing:

qwen3.6:35b-a3b-iq4 for judging, this model is a lot smarter and better than qwen3.5:9b but the problem is, it's slow, it needs 18gb vram, i only have 10, so 8 of it has to be offloaded to RAM and CPU.


## Tech Stack

- Python 3.12
- Ollama
- Whisper.cpp
- FFmpeg
- PyTorch
- DaVinci Resolve

## Special Thanks to Gootecks the inspiration behind the project

<img width="656" height="180" alt="image" src="https://github.com/user-attachments/assets/0e03bcc3-31a0-4f13-be90-3da26305eeac" />

I met him at an event before COVID, genuinely a cool guy.
