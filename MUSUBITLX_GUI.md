## MusubiTLX – Qwen LoRA Training Web GUI

MusubiTLX is a small web interface for training Qwen‑Image LoRA networks using **Musubi Tuner**.
It wraps the existing Musubi training scripts in a simple, browser‑based GUI.

Musubi Tuner itself is developed in the main project repository: [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner).


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

- **Logging**
  - For each training run, a timestamped log file is created under the selected output folder, e.g.:
    - `output/art/training_log_art_20251121_221124.log`

- **Production‑style server**
  - Uses `waitress` WSGI server when available.
  - Falls back to Flask’s development server only if `waitress` is not installed.

---

### 2. Requirements & Dependencies

Assuming you already have **Musubi Tuner** cloned (this GUI lives in the same repository).

#### Python

- Python 3.10+ (3.12 has been tested)

#### Python packages

The GUI depends on:

- `Flask`
- `waitress`
- `PyYAML`
- `toml`

Musubi Tuner itself has additional requirements (PyTorch, safetensors, etc.) which should already be installed according to the main project’s README.

Example (inside the Musubi repo):

```bash
cd /path/to/musubi-tuner
python -m venv venv
source venv/bin/activate

# Base Musubi Tuner dependencies (simplified example)
pip install -r requirements.txt

# Extra for the web GUI
pip install flask waitress pyyaml toml
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

From the Musubi Tuner project root:

```bash
cd /path/to/musubi-tuner
source venv/bin/activate
python webgui.py
```

If `waitress` is installed, you’ll see a message like:

```text
Starting MusubiTLX with waitress...
```

The GUI will be available at:

- `http://127.0.0.1:5000` (local machine)
- Or `http://<your-LAN-IP>:5000` for other devices on the same network.

---

### 4. Using the GUI

#### Step 1 – Download required models (once)

At the top of the page:

- Click the buttons to download:
  - `DiT Model` (`qwen_image_bf16.safetensors`)
  - `Text Encoder` (`qwen_2.5_vl_7b.safetensors`)
  - `VAE Model` (`diffusion_pytorch_model.safetensors`)

These are saved in the current working directory and used by Musubi Tuner’s training scripts.

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
  At the bottom of the training settings there is an **“Show advanced trainer flags”** button:

  - When you expand it, you get:
    - A multiline textbox for **advanced trainer flags**.
    - A read‑only **“Effective training command (preview)”**.
  - By default, the textbox is pre‑filled with sensible VRAM‑related flags based on your selected VRAM profile  
    (e.g. `--fp8_base --fp8_scaled --blocks_to_swap ... --gradient_checkpointing`).
  - You can edit this textbox directly if you know what you are doing; whatever you write there is appended to the
    `python qwen_image_train_network.py` command shown in the preview.
  - This makes it easy for power users to:
    - tweak or remove VRAM flags
    - add extra options (dropout, gradient clipping, etc.)
    - while always seeing exactly what command MusubiTLX will run when you click **Start training**.

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
- The spinner and message “Training in progress... Please wait.” appear.
- The **Training log** area starts to fill with:
  - latent cache output
  - text encoder cache output
  - training progress (`steps: ...`, loss values, etc.)

At the end:

- On success you’ll see:

  - `TRAINING COMPLETED!`
  - Path to the final LoRA file.
  - Path to the log file.

- On error you’ll see:

  - `TRAINING ABORTED WITH ERROR (exit code X)`  
  - A note to check the log file for details.

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

### 7. Additional Features

- **Caption model unloading**: Before training starts, caption models (ViT-GPT2, BLIP, Qwen-VL) are automatically unloaded from VRAM to free up memory for training.
- **Training compatibility**: Training settings are optimized for compatibility with various inference settings in ComfyUI and other inference tools.
- **Warning system**: 
  - Warning appears if you have too few images (< 8) with recommendations to improve training results.
  - Info buttons provide helpful tooltips for complex settings (learning rate, LoRA rank/dims).
- **Individual image management**: Remove individual images from the preview without clearing all images.

### 8. Credits

- Web GUI: **MusubiTLX**, created by **TLX**.
- Training core: **Musubi Tuner** (Qwen Image LoRA training scripts).


