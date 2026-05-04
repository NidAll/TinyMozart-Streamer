# TinyMozart Streamer

Local one-button piano streamer for
[`LH-Tech-AI/TinyMozart_v2_85M`](https://huggingface.co/LH-Tech-AI/TinyMozart_v2_85M).

The app loads TinyMozart locally, generates several MIDI candidates, scores/rejects weak loops, and plays the best passage in a continuous stream until stopped.

## Features

- Streamlit one-button Start/Stop UI.
- CUDA-enabled PyTorch inference when a compatible NVIDIA GPU is available.
- Candidate search for more consistent music.
- Three generation quality modes:
  - `Fast`: no multi-candidate search, quickest startup.
  - `Balanced`: 2 candidates per passage.
  - `High`: 4 candidates per passage, best default quality.
- MIDI quality scoring for note count, pitch range, density, repetition, and silence.
- Default playback through pygame MIDI, with optional SoundFont rendering if configured later.

## Setup

This project expects Python 3.12 on Windows. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file uses the PyTorch CUDA 13.0 wheel index. If you need CPU-only PyTorch or a different CUDA version, adjust the first lines of `requirements.txt`.

## Auto Install And Launch

On Windows, run:

```powershell
.\launch.bat
```

or:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

The launch script creates `venv` if needed, installs dependencies from `requirements.txt`, and starts the Streamlit UI.

## Run

```powershell
venv\Scripts\streamlit.exe run app.py
```

Open the local Streamlit URL, press **Start streaming**, and press **Stop streaming** to stop generation and playback.

The first start downloads `model.pt` from Hugging Face into the normal Hugging Face cache. The model file is not committed to this repo.

## Optional Piano Rendering

FluidSynth is not required for this repo. Without it, playback falls back to pygame MIDI.

If you later install FluidSynth and a `.sf2`/`.sf3` SoundFont, launch with:

```powershell
$env:FLUIDSYNTH_EXE="C:\path\to\fluidsynth.exe"
$env:TINYMOZART_SF2="C:\path\to\piano.sf2"
venv\Scripts\streamlit.exe run app.py
```

## Notes

TinyMozart is a small unconditional symbolic music model. The candidate scorer improves consistency, but it cannot fully remove the model's inherent quality ceiling.
