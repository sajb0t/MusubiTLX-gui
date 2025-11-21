# MusubiTLX – Qwen LoRA Training Web GUI

Simple web GUI for training Qwen-Image LoRA models using [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner).

## What this repo contains

- `webgui.py` – the Flask/Waitress web GUI.
- `MUSUBITLX_GUI.md` – full user guide for the GUI.

> Do **not** copy this README into your Musubi Tuner folder.  
> Only copy `webgui.py` and `MUSUBITLX_GUI.md` next to your existing `musubi-tuner` installation.

## Quick usage

1. Clone or install Musubi Tuner normally.
2. Copy `webgui.py` and `MUSUBITLX_GUI.md` into the Musubi Tuner project root.
3. Activate your virtualenv and start the GUI:

source venv/bin/activate
python webgui.py Open `http://127.0.0.1:5000` in your browser.
