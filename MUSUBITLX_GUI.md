## MusubiTLX – Qwen LoRA Training Web GUI

**Version 0.2**

MusubiTLX is a small web interface for training Qwen‑Image LoRA networks using **Musubi Tuner**.
It wraps the existing Musubi training scripts in a simple, browser‑based GUI.

Musubi Tuner itself is developed in the main project repository: [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner).

### Quick Start for New Users

**New to MusubiTLX?** The easiest way to get started:

1. Download or clone [MusubiTLX-gui](https://github.com/sajb0t/MusubiTLX-gui) (contains `webgui.py` and start scripts)
2. Run the start script:
   ```bash
   ./start_gui.sh  # Linux/macOS
   # or
   start_gui.bat   # Windows
   ```
3. The script will guide you through:
   - Cloning musubi-tuner automatically (if needed)
   - Creating virtual environment
   - Downloading required model files
   - Starting the web server

Everything is interactive – just answer the prompts!

### What's New in 0.2

- **Improved start scripts**: Full interactive setup with automatic PyTorch installation (CUDA version selection), dependency installation, and model downloads
- **Better live log streaming**: Added gevent WSGI server support for smoother real-time training log updates
- **All three models downloadable**: Text Encoder (`qwen_2.5_vl_7b.safetensors`) now included in automatic model download
- **System monitoring**: Added psutil dependency for accurate RAM/VRAM monitoring
- **Bug fixes**: Fixed various issues with SSE streaming and server stability

---

### 1. Features

- **Image upload + captions**
  - Upload multiple training images at once.
  - Custom file picker button with clear status feedback.
  - Drag & drop directly into the preview area.
  - Per‑image caption text fields; captions are saved as `.txt` files next to each image.
  - Individual image removal: click the "×" button on any image preview to remove it.
  - Dynamic path preview: shows the exact output path for the final LoRA file.

- **Auto-caption with ViT-GPT2 / BLIP / Qwen-VL (optional)**
  - Automatically generate captions for all uploaded images using one of three models:
    - **ViT-GPT2** – fast, lightweight (good for quick tests or low VRAM).
    - **BLIP large** – more detailed (better descriptions, slower, uses more VRAM).
      - Includes a **Detail level slider (1–5)** to control caption detail:
        - Level 1: Basic
        - Level 3: Detailed (default)
        - Level 5: Extremely Detailed
    - **Qwen-VL** – extremely detailed (best quality, slowest, highest VRAM usage).
      - Automatically uses fp8 precision to reduce VRAM usage by ~50%.
      - Requires `qwen_2.5_vl_7b.safetensors` in the project directory (same model used for training).
      - On first use, downloads processor/tokenizer from Hugging Face (~200–500 MB, cached afterwards).
  - **Caption length slider**: Control maximum token length (32–512 tokens).
  - Requires additional Python dependencies (see the Requirements section below).
  - Triggered from the GUI via the **"Auto-caption images"** button in Step 1.

- **Dataset configuration**
  - Resolution (e.g. `1024x1024`).
  - Batch size.
  - Image repeats (how many times each image is seen).
  - VRAM profile (12 GB or 16 GB) with sensible defaults for memory‑constrained GPUs.

- **Training configuration**
  - Epochs.
  - Learning rate.
    - **Info button**: Hover to see common learning rate values in both scientific and decimal notation.
  - Optimizer selection:
    - `AdamW` (default)
    - `Adafactor` (more memory friendly)
    - `AdamW8bit` (requires `bitsandbytes`)
  - LoRA rank and dims.
    - **Info buttons**: Hover to see recommendations for training real people and understanding rank vs dims.
    - Default: rank 16, dims 128 (good starting point for person training).
  - Output folder and LoRA filename.
    - Dynamic path preview updates automatically as you type.

- **Real-time system monitoring**
  - **RAM/VRAM display**: Shows current RAM and VRAM usage, updated every 3 seconds.
    - Color-coded indicators (green/yellow/red) based on usage percentage.
    - Helps monitor memory usage, especially after running other VRAM-intensive applications.
  - Located below the training buttons and above the console/log.

- **Live training log**
  - Server‑Sent Events (SSE) stream of all output from:
    - Latent cache
    - Text encoder cache
    - LoRA training
  - Output is shown line‑by‑line and auto‑scrolls.
  - **Resume log stream**: If the browser tab goes to sleep (e.g., computer sleeps) or you reload the page, you can click "Resume log stream" to reconnect and fetch the latest log content automatically.

- **Sample generation during training**
  - Enable automatic sample image generation at training checkpoints.
  - Configure sample frequency (every N epochs or every N steps).
  - Edit sample prompts directly in the GUI with support for parameters (`--w`, `--h`, `--s`, `--d`, `--f`).
  - Option to generate baseline samples before training starts for before/after comparison.
  - Default resolution for samples is 256×256 (if not specified in prompts).
  - Default seed is used if not specified in prompts.

- **Sample gallery**
  - View generated sample images as clickable thumbnails directly in the GUI.
  - Automatically refreshes every 30 seconds during training when samples are enabled.
  - Manually refresh at any time using the "Refresh sample gallery" button.
  - Samples are saved to `output/<folder>/sample/` directory.
  - Click any thumbnail to open the full-size image in a new tab.

- **Logging**
  - For each training run, a timestamped log file is created under the selected output folder, e.g.:
    - `output/art/training_log_art_20251121_221124.log`

- **RAM monitoring and protection**
  - Automatically monitors RAM usage during training.
  - If RAM usage exceeds 95%, training is automatically aborted to prevent system lockups.
  - Caption models are automatically unloaded before training starts to free VRAM.

- **Production‑style server**
  - Uses `gevent` WSGI server when available (best for live log streaming).
  - Falls back to `waitress` (32 threads) if gevent is not installed.
  - Falls back to Flask's development server only if neither is installed.

---

### 2. Requirements & Dependencies

**Note:** If you use the start scripts (`start_gui.sh` / `start_gui.bat`), they can automatically clone **Musubi Tuner** for you if it's missing. Otherwise, you need to have **Musubi Tuner** cloned first (this GUI lives in the same repository).

#### Python

- Python 3.10+ (3.12 has been tested)

#### Python packages

The GUI depends on:

- `Flask`
- `waitress` (production WSGI server)
- `gevent` (optional, recommended for better live log streaming)
- `PyYAML`
- `toml`
- `psutil` (for RAM/VRAM monitoring)

Musubi Tuner itself has additional requirements (PyTorch, safetensors, etc.) which should already be installed according to the main project’s README.

Example (inside the Musubi repo):

```bash
cd /path/to/musubi-tuner
python -m venv venv
source venv/bin/activate

# Base Musubi Tuner dependencies (simplified example)
pip install -r requirements.txt

# Extra for the web GUI
pip install flask waitress pyyaml toml psutil

# Optional: For better live log streaming during training
pip install gevent
```

If you plan to use `AdamW8bit`, you also need:

```bash
pip install bitsandbytes
```

If you plan to use **auto-captioning (ViT-GPT2 / BLIP / Qwen-VL)**, you also need:

```bash
pip install "transformers>=4.44.0" pillow
```

**Note for Qwen-VL**: The Qwen-VL model file (`qwen_2.5_vl_7b.safetensors`) must be in the project root directory. This is the same model file used for Qwen-Image training. On first use, Qwen-VL will download the processor/tokenizer from Hugging Face (~200–500 MB, cached afterwards).

---

### 3. Starting the GUI

The start scripts provide an **interactive installation and setup process** that can automatically:
- Clone musubi-tuner repository if missing
- Create virtual environment
- Download required model files
- Start the server with SSH disconnect protection

#### Option 1: Using the start script (Recommended)

**Linux/macOS:**
```bash
# If you only have the MusubiTLX-gui files (webgui.py), the script will:
# 1. Detect that musubi-tuner is missing
# 2. Offer to clone it automatically
# 3. Set up everything for you
./start_gui.sh
```

**Available flags:**
- `./start_gui.sh` – Start in background (survives SSH disconnect, recommended)
- `./start_gui.sh --fg` – Start in foreground (interactive mode, use Ctrl+C to stop)
- `./start_gui.sh --stop` – Stop the running server
- `./start_gui.sh --status` – Check if server is running

**What the start script does:**
1. **Repository check**: Detects if you're in the musubi-tuner repository. If not, offers to:
   - Automatically clone musubi-tuner from GitHub
   - Or copy `webgui.py` to an existing musubi-tuner directory
2. **Virtual environment check**: If missing, offers to create it automatically
3. **Dependency installation**: Offers to install PyTorch (with CUDA version selection), musubi-tuner dependencies, and GUI dependencies
4. **Model file check**: Detects missing models and offers automatic download:
   - DiT Model (`qwen_image_bf16.safetensors`) ~38 GB - required for training
   - VAE Model (`diffusion_pytorch_model.safetensors`) ~335 MB - required for training
   - Text Encoder (`qwen_2.5_vl_7b.safetensors`) ~16 GB - needed for Qwen-VL auto-captioning and image training.
5. **Server startup**: Starts the server in the background with nohup (survives SSH disconnect)

**Stopping the server:**
```bash
# Option 1: Use the --stop flag
./start_gui.sh --stop

# Option 2: Use the separate stop script
./stop_gui.sh

# Option 3: Use the process ID shown when starting
kill <PID>
```

**Windows:**
```cmd
cd C:\path\to\musubi-tuner
start_gui.bat
```

**Available flags:**
- `start_gui.bat` – Start in background
- `start_gui.bat --stop` – Stop the running server
- `start_gui.bat --status` – Check if server is running

**Stopping the server (Windows):**
```cmd
start_gui.bat --stop
REM or
stop_gui.bat
```

This will start the server in the background with logs written to `webgui.log`.

#### Option 2: Manual start

**Linux/macOS:**
```bash
cd /path/to/musubi-tuner
source venv/bin/activate
python webgui.py
```

**Windows:**
```cmd
cd C:\path\to\musubi-tuner
venv\Scripts\activate
python webgui.py
```

When the server starts, you'll see a message indicating which WSGI server is being used:

```text
Starting MusubiTLX with gevent...     # Best for live log streaming
# or
Starting MusubiTLX with waitress...   # Good fallback
# or
Starting MusubiTLX with Flask dev server...  # Development only
```

The GUI will be available at:

- `http://127.0.0.1:5000` (local machine)
- Or `http://<your-LAN-IP>:5000` for other devices on the same network.

**Note:** The start scripts (`start_gui.sh` / `start_gui.bat`) automatically handle SSH disconnect protection, interactive setup, and are recommended for production use. They can automatically set up everything needed for first-time users.

---

### 4. Using the GUI

#### Step 1 – Download required models (once)

**Option 1: Automatic download via start script**

If you use the start script (`./start_gui.sh` or `start_gui.bat`), it will automatically detect missing model files and offer to download them. This is the easiest way for new users.

**Option 2: Download via GUI**

At the top of the web GUI page:

- Click the buttons to download:
  - `DiT Model` (`qwen_image_bf16.safetensors`)
  - `Text Encoder` (`qwen_2.5_vl_7b.safetensors` – optional, for auto-captioning)
  - `VAE Model` (`diffusion_pytorch_model.safetensors`)

These are saved in the current working directory and used by Musubi Tuner's training scripts.

#### Step 2 – Upload training images

In **1. Upload training images**:

- Click the **Select images** button (custom file picker) and choose your images, or
- Drag & drop images directly into the preview area below.

For each image:

- A preview card appears showing the image thumbnail.
- There is a text field labeled `Caption for image N` where you can enter or edit captions.
- Click the **"×"** button on any image preview to remove it from the training set.
- The caption is saved as `<image_name>.txt` in the output folder and used as the text prompt.

**Warning for few images**: If you have fewer than 8 images and less than 300 total steps, a yellow warning will appear with recommendations:
- Use 10–20+ images for better quality
- Increase epochs to get more training steps
- Consider a lower learning rate
- Add more specific captions

##### Optional – Auto-caption images (ViT-GPT2 / BLIP / Qwen-VL)

- After selecting or dropping your images, you can click:

  - **Auto-caption images**

- **Caption model selection** (dropdown):
  - **ViT-GPT2 – fast, lightweight** (good for quick tests or low VRAM)
  - **BLIP large – more detailed** (better descriptions, slower, uses more VRAM)
    - **Detail level slider (1–5)**: Controls how detailed the BLIP captions are.
      - Level 1: Basic (6 beams, length_penalty 1.1)
      - Level 3: Detailed – default (10 beams, length_penalty 1.5)
      - Level 5: Extremely Detailed (20 beams, length_penalty 2.5)
    - Note: BLIP-large has inherent limitations and may not describe very fine details like specific lighting conditions or facial features in extreme detail, even at level 5.
  - **Qwen-VL – extremely detailed** (best quality, slowest, highest VRAM usage)
    - Automatically uses fp8 precision to reduce VRAM usage (~50% less than bfloat16).
    - Requires `qwen_2.5_vl_7b.safetensors` in the project root (automatically found).
    - On first use, downloads processor/tokenizer from Hugging Face (~200–500 MB, cached afterwards).
    - **Warning**: Requires ~7–8 GB VRAM in fp8 mode (~15 GB in bfloat16). Close other GPU applications (e.g. ComfyUI) before using.

- **Caption length slider**: Set maximum tokens for captions (32–512).
  - Default: 128 tokens (ViT-GPT2), 160 tokens (BLIP), 256 tokens (Qwen-VL)

- On first use, the selected captioning model weights will be downloaded from Hugging Face (this can take a while for ViT-GPT2/BLIP).
- If the required Python packages are not installed, the GUI will show a clear message with the exact `pip install ...` command to run.
- Generated captions are written into the same per-image text fields and saved as `.txt` files just like manual captions.

#### Step 3 – Configure training

In **2. Training settings**:

- **Trigger word**  
  A special word you can use in captions to refer to this LoRA (e.g. `tlx_man`).

- **VRAM profile**
  - `12 GB – safe / recommended (most offload, slowest)`
    - fp8, large block swap to CPU, gradient checkpointing with CPU offload.
  - `16 GB – higher VRAM use (faster, may OOM)`
    - fp8, some block swap, gradient checkpointing.
  - `24+ GB – performance (fastest)`
    - no fp8 or block swap, only gradient checkpointing (uses more VRAM but runs faster on big cards).

- **Save / Load YAML config**
  - **Save configuration as YAML**  
    Saves your current GUI settings to `configs/last_config.yaml`.
  - **Load last YAML config**  
    Reads `configs/last_config.yaml` and populates all GUI fields with the saved values (without starting training).

- **Epochs**  
  How many passes over the dataset. Combined with `Image repeats`, this controls total steps.

- **Batch size**  
  Number of images per optimization step. For 16 GB at 1024x1024, `1` is recommended.

- **Image repeats**  
  How many times each image is seen.

  Effective images per epoch:

  \[
  \text{effective\_images} = \text{num\_images} \times \text{image\_repeats}
  \]

  Steps per epoch:

  \[
  \text{steps\_per\_epoch} = \left\lceil \frac{\text{effective\_images}}{\text{batch\_size}} \right\rceil
  \]

  Total steps:

  \[
  \text{total\_steps} = \text{steps\_per\_epoch} \times \text{epochs}
  \]

- **Learning rate**  
  Typical values: `5e-5`, `1e-4`, etc.  
  Click the **ℹ️** info button next to the field to see common learning rate values in both scientific notation (e.g., `5e-5`) and decimal notation (e.g., `0.00005`).

- **Optimizer**
  - `AdamW` – standard, good default.
  - `Adafactor` – more memory‑friendly.
  - `AdamW8bit` – 8‑bit optimizer (requires `bitsandbytes`).

- **Prompt / subject description**  
  Generic description of the subject (used in combination with captions).

- **Resolution**  
  Training resolution, e.g. `1024x1024`. Higher resolutions need more VRAM.

- **Seed**  
  Random seed for reproducibility.

- **LoRA rank / dims**  
  Lower values reduce VRAM usage and model size; higher values can capture more detail.  
  - Click the **ℹ️** info buttons next to rank and dims to see recommendations:
    - Default (rank 16, dims 128) is good for training real people.
    - Rank (network_dim) controls the size of the LoRA adaptation matrix.
    - Dims (network_alpha) controls the scaling of the LoRA weights.

- **Output folder**  
  Relative to `output/`.  
  Example:
  - `art` → files go to `output/art/`.

- **LoRA filename**  
  The final trained file will be:

  ```text
  output/<output_folder>/<output_name>.safetensors
  ```

  The **Final LoRA file** path is shown dynamically below the filename field and updates as you type.

- **Advanced trainer flags & command preview (optional)**  
  At the bottom of the training settings there is an **"Show advanced trainer flags"** button:

  - When you expand it, you get:
    - A multiline textbox for **advanced trainer flags**.
    - A read‑only **"Effective training command (preview)"**.
  - By default, the textbox is pre‑filled with sensible VRAM‑related flags based on your selected VRAM profile  
    (e.g. `--fp8_base --fp8_scaled --blocks_to_swap ... --gradient_checkpointing --max_grad_norm 1.0`).
  - `--max_grad_norm 1.0` is automatically added to prevent gradient explosion and numerical instability.
  - You can edit this textbox directly if you know what you are doing; whatever you write there is appended to the
    `python qwen_image_train_network.py` command shown in the preview.
  - This makes it easy for power users to:
    - tweak or remove VRAM flags
    - add extra options (dropout, gradient clipping, etc.)
    - while always seeing exactly what command MusubiTLX will run when you click **Start training**.

- **Sample generation settings (optional)**
  - **Generate sample previews during training**: Enable this checkbox to generate sample images during training.
  - **Sample prompts**: Enter one prompt per line. Each line generates one sample image.
    - You can use your trigger word with `{{trigger}}` syntax (e.g., `A portrait of {{trigger}}`).
    - Optional parameters per prompt:
      - `--w <width>`: Image width (default: 256 if not specified)
      - `--h <height>`: Image height (default: 256 if not specified)
      - `--s <steps>`: Number of inference steps (default: 20)
      - `--d <seed>`: Random seed (default: uses training seed if not specified)
      - `--f <frames>`: Number of frames for video (keep at 1 for still images)
    - Lines starting with `#` are treated as comments and ignored.
    - Example:
      ```
      A studio portrait of {{trigger}} --w 960 --h 960 --s 30
      {{trigger}} in a fantasy landscape --w 1024 --h 768
      ```
  - **Sample every N epochs**: How often to generate samples (default: 1, generates after each epoch).
  - **Sample every N steps**: Alternative to epochs-based sampling (default: 0, disabled).
  - **Generate baseline samples before training starts**: Enable this to generate samples with the untrained model, allowing you to compare before/after results.
  - **Refresh sample gallery**: Manual button to refresh the sample gallery at any time.
  - **Auto-refresh**: The sample gallery automatically checks for new samples every 30 seconds during training when samples are enabled.

#### Step 4 – Check estimated training time and system resources

- **Estimated training time**: Just below the LoRA filename, the GUI shows a **rough time estimate** based on:
  - number of images
  - epochs
  - batch size
  - image repeats
  - VRAM profile
  - resolution

  This is only an approximation, but it helps you understand whether you are looking at seconds, minutes or hours.

- **System resources**: Below the training buttons and above the console/log, you'll see:
  - **RAM**: Current RAM usage (available/total and percentage)
  - **VRAM**: Current VRAM usage (available/total and percentage)
  - Updated every 3 seconds with color-coded indicators (green/yellow/red)
  - Useful for monitoring memory usage, especially after running other applications like ComfyUI.

#### Step 5 – Start training

- Click **Start training**.
- The spinner and message "Training in progress... Please wait." appear.
- The **Training log** area starts to fill with:
  - latent cache output
  - text encoder cache output
  - training progress (`steps: ...`, loss values, etc.)

**During training**:
- If sample generation is enabled, the sample gallery will automatically refresh every 30 seconds to show newly generated samples.
- The system monitors RAM usage and will automatically abort training if RAM exceeds 95% to prevent system lockups.

**Log stream management**:
- If the browser tab goes to sleep or you reload the page during training, click **"Resume log stream"** to reconnect and view the latest log content.
- The log stream will automatically resume if training is still active.

At the end:

- On success you'll see:

  - `TRAINING COMPLETED!`
  - Path to the final LoRA file.
  - Path to the log file.
  - If samples were enabled, the sample gallery will show all generated samples.

- On error you'll see:

  - `TRAINING ABORTED WITH ERROR (exit code X)`  
  - A note to check the log file for details.
  - If the error was due to high RAM usage (>95%), a specific message will be shown.

---

### 5. Files and output layout

Given:

- Output folder: `art`
- Output name: `art`

You will typically see something like:

- `output/art/dataset_config.toml`
- `output/art/*.png` – your training images (copied from upload)
- `output/art/*.txt` – captions for each image
- `output/art/*_qi.safetensors` – latent cache files
- `output/art/*_qi_te.safetensors` – text encoder cache files
- `output/art/art.safetensors` – **final trained LoRA**
- `output/art/training_log_art_YYYYMMDD_HHMMSS.log` – log for that training run
- `output/art/sample_prompts.txt` – sample prompts file (if samples are enabled)
- `output/art/sample/` – directory containing generated sample images (if samples are enabled)
  - Sample images are PNG files named with timestamps
  - Baseline samples (if enabled) are generated before training starts

---

### 6. Notes and tips

- For 16 GB GPUs at `1024x1024`:
  - Start with:
    - `VRAM profile = 12` (safer default, more offload)
    - `Batch size = 1`
    - `LoRA rank = 16`, `dims = 128`
    - `Image repeats` and `Epochs` adjusted to reach your desired total step count.

- If you hit CUDA OOM:
  - Close other GPU‑heavy apps (e.g. ComfyUI) – check the VRAM display in the GUI to verify VRAM is freed.
  - Lower resolution or LoRA rank/dims.
  - Use the 12 GB profile even if you have more VRAM – it's more aggressive with offloading.
  - **For Qwen-VL captioning**: 
    - Qwen-VL automatically uses fp8 precision to reduce VRAM usage (~50% less than bfloat16).
    - Still requires ~7–8 GB VRAM in fp8 mode, so close other GPU applications first.
    - If you still get OOM errors, use BLIP-large instead (uses ~3–4 GB VRAM).

- If something goes wrong:
  - Check the **Training log** in the GUI.
  - Then open the corresponding `training_log_*.log` file under `output/<your_folder>/` for full stack traces.

---

### 7. Credits

- Web GUI: **MusubiTLX**, created by **TLX**.
- Training core: **Musubi Tuner** (Qwen Image LoRA training scripts).


