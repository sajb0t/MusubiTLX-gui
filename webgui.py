from flask import Flask, render_template_string, request, send_file, Response, stream_with_context, jsonify, make_response
import subprocess, os, yaml, urllib.request, urllib.parse, shlex, mimetypes, sys, platform
from werkzeug.utils import secure_filename
import toml
import threading
import queue
import time
import signal
import gzip
import io
from datetime import datetime

app = Flask(__name__)

# Add gzip compression to reduce response size (helps with network timeouts)
@app.after_request
def compress_response(response):
    # Compress HTML and JSON responses that are large enough to benefit
    if (response.status_code == 200 and 
        response.content_type and 
        ('text/html' in response.content_type or 'application/json' in response.content_type) and
        len(response.get_data()) > 500):
        
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' in accept_encoding.lower():
            # Compress the response
            gzip_buffer = io.BytesIO()
            with gzip.GzipFile(mode='wb', fileobj=gzip_buffer, compresslevel=6) as gzip_file:
                gzip_file.write(response.get_data())
            
            response.set_data(gzip_buffer.getvalue())
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Content-Length'] = len(response.get_data())
            response.headers['Vary'] = 'Accept-Encoding'
    
    return response
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
OUTPUT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "output"))
current_log_path = None
latest_readme_output_dir = None

# Global queue for streaming output
output_queue = queue.Queue()

# Globals for optional auto-captioning
caption_vit_model = None
caption_vit_feature_extractor = None
caption_vit_tokenizer = None
caption_blip_model = None
caption_blip_processor = None
caption_qwen_vl_model = None
caption_qwen_vl_processor = None
caption_qwen_vl_use_fp8 = False  # Track if we're using fp8 for Qwen-VL


def _normalize_caption_text(text: str) -> str:
    """
    Light cleanup of caption phrasing to make results a bit nicer for training:
    - Strip leading filler like "there is ...", "there are ...", "this is ..."
    - Trim whitespace and normalize first letter to uppercase
    """
    if not isinstance(text, str):
        return text
    t = text.strip()
    lower = t.lower()
    prefixes = ["there is ", "there are ", "this is ", "there's "]
    for p in prefixes:
        if lower.startswith(p):
            t = t[len(p) :].lstrip()
            break
    if t:
        t = t[0].upper() + t[1:]
    return t

# Track active training-related subprocesses so we can cancel them from the UI
active_processes = []
active_processes_lock = threading.Lock()

def register_process(proc):
    with active_processes_lock:
        active_processes.append(proc)

def unregister_process(proc):
    with active_processes_lock:
        if proc in active_processes:
            active_processes.remove(proc)

def cancel_active_processes():
    with active_processes_lock:
        procs = list(active_processes)
    for p in procs:
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass

def is_training_active():
    with active_processes_lock:
        for proc in active_processes:
            if proc.poll() is None:
                return True
    return False

def _require_caption_deps():
    try:
        import transformers  # type: ignore  # noqa: F401
        import PIL  # type: ignore  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Auto-caption dependencies are missing.\n"
            "Please install them in your Musubi Tuner virtual environment:\n"
            "  pip install \"transformers>=4.44.0\" pillow\n"
            "After installing, restart Musubi Tuner and try auto-caption again."
        )

def load_vit_gpt2_caption_model():
    """
    Lazy-load a lightweight image captioning model (ViT-GPT2).
    """
    global caption_vit_model, caption_vit_feature_extractor, caption_vit_tokenizer
    if (
        caption_vit_model is not None
        and caption_vit_feature_extractor is not None
        and caption_vit_tokenizer is not None
    ):
        return caption_vit_model, caption_vit_feature_extractor, caption_vit_tokenizer

    _require_caption_deps()

    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer  # type: ignore
    import torch

    model_id = os.environ.get("CAPTION_MODEL_ID_VIT", "nlpconnect/vit-gpt2-image-captioning")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    caption_vit_model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
    caption_vit_feature_extractor = ViTImageProcessor.from_pretrained(model_id)
    caption_vit_tokenizer = AutoTokenizer.from_pretrained(model_id)

    return caption_vit_model, caption_vit_feature_extractor, caption_vit_tokenizer

def load_blip_caption_model():
    """
    Lazy-load a more descriptive BLIP image captioning model.
    """
    global caption_blip_model, caption_blip_processor
    if caption_blip_model is not None and caption_blip_processor is not None:
        return caption_blip_model, caption_blip_processor

    _require_caption_deps()

    from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
    import torch

    model_id = os.environ.get("CAPTION_MODEL_ID_BLIP", "Salesforce/blip-image-captioning-large")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    caption_blip_model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    caption_blip_processor = BlipProcessor.from_pretrained(model_id)

    return caption_blip_model, caption_blip_processor

def load_qwen_vl_caption_model():
    """
    Lazy-load Qwen2.5-VL model for highly detailed captions.
    Automatically finds the model in the project directory, or downloads it if missing.
    """
    global caption_qwen_vl_model, caption_qwen_vl_processor
    if caption_qwen_vl_model is not None and caption_qwen_vl_processor is not None:
        return caption_qwen_vl_model, caption_qwen_vl_processor

    _require_caption_deps()

    import torch
    from transformers import AutoProcessor
    from musubi_tuner.qwen_image.qwen_image_utils import load_qwen2_5_vl
    from PIL import Image

    # Try to find the model automatically in common locations
    # IMPORTANT: Check same dir as webgui.py FIRST (most common location)
    webgui_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = "qwen_2.5_vl_7b.safetensors"
    
    # Build list of possible paths to check
    possible_paths = []
    
    # 1. User override via environment variable
    env_path = os.environ.get("CAPTION_MODEL_PATH_QWEN_VL", "")
    if env_path:
        possible_paths.append(env_path)
    
    # 2. Same directory as webgui.py (MOST COMMON - check this first!)
    webgui_path = os.path.join(webgui_dir, model_filename)
    possible_paths.append(webgui_path)
    
    # 3. Current working directory
    cwd_path = os.path.join(os.getcwd(), model_filename)
    possible_paths.append(cwd_path)
    
    # 4. Relative path from current directory
    possible_paths.append(model_filename)
    
    # Check all possible paths
    model_path = None
    for path in possible_paths:
        if not path:
            continue
        # Try both absolute and relative path
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            model_path = abs_path
            break
        if os.path.exists(path) and os.path.isfile(path):
            model_path = os.path.abspath(path)
            break
    
    # CRITICAL: Only download if model_path is STILL None after checking ALL paths
    if not model_path:
        # Use the same URL as in DOWNLOADS dictionary
        qwen_vl_url = DOWNLOADS.get("Text Encoder", "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors?download=true")
        
        # Try downloading to project root (same directory as webgui.py)
        download_path = os.path.join(webgui_dir, model_filename)
        
        # ABSOLUTE FINAL CHECK: Make absolutely sure file doesn't exist before downloading
        # Check both absolute and relative paths one more time
        if os.path.exists(download_path) and os.path.isfile(download_path):
            model_path = os.path.abspath(download_path)
        elif os.path.exists(os.path.abspath(download_path)) and os.path.isfile(os.path.abspath(download_path)):
            model_path = os.path.abspath(download_path)
        elif not os.path.exists(download_path) and not os.path.exists(os.path.abspath(download_path)):
            try:
                print(f"Qwen-VL model not found. Downloading from Hugging Face...")
                print(f"This is a large file (~14GB) and may take several minutes.")
                download_with_progress(qwen_vl_url, download_path)
                print(f"Download completed: {download_path}")
                if os.path.exists(download_path) and os.path.isfile(download_path):
                    model_path = download_path
            except Exception as e:
                # If download fails, try current working directory as fallback
                download_path = model_filename
                if os.path.exists(download_path) and os.path.isfile(download_path):
                    model_path = os.path.abspath(download_path)
                elif not os.path.exists(download_path):
                    try:
                        download_with_progress(qwen_vl_url, download_path)
                        print(f"Download completed: {download_path}")
                        if os.path.exists(download_path) and os.path.isfile(download_path):
                            model_path = os.path.abspath(download_path)
                    except Exception as e2:
                        raise RuntimeError(
                            f"Qwen-VL model not found and automatic download failed.\n"
                            f"Download error: {e2}\n"
                            f"Please download 'qwen_2.5_vl_7b.safetensors' manually from:\n"
                            f"{qwen_vl_url}\n"
                            f"Or set CAPTION_MODEL_PATH_QWEN_VL environment variable to point to the model file."
                        )
        
        # Verify the file exists
        if not model_path or not (os.path.exists(model_path) and os.path.isfile(model_path)):
            raise RuntimeError(
                "Qwen-VL model not found.\n"
                "Please ensure 'qwen_2.5_vl_7b.safetensors' is in the project directory, or set CAPTION_MODEL_PATH_QWEN_VL environment variable."
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load processor (uses Hugging Face defaults)
    IMAGE_FACTOR = 28
    DEFAULT_MAX_SIZE = 1280
    min_pixels = 256 * IMAGE_FACTOR * IMAGE_FACTOR
    max_pixels = DEFAULT_MAX_SIZE * IMAGE_FACTOR * IMAGE_FACTOR
    caption_qwen_vl_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    
    # Load model - use fp8 to reduce VRAM usage (much lower memory footprint)
    # fp8 reduces VRAM usage by ~50% compared to bfloat16
    global caption_qwen_vl_use_fp8
    try:
        # Try fp8 first for lower VRAM usage (recommended if you have VRAM constraints)
        dtype = torch.float8_e4m3fn
        _, caption_qwen_vl_model = load_qwen2_5_vl(model_path, dtype=dtype, device=device, disable_mmap=True)
        caption_qwen_vl_use_fp8 = True
    except (RuntimeError, AttributeError, TypeError) as e:
        # Fall back to bfloat16 if fp8 not available
        dtype = torch.bfloat16
        _, caption_qwen_vl_model = load_qwen2_5_vl(model_path, dtype=dtype, device=device, disable_mmap=True)
        caption_qwen_vl_use_fp8 = False
    caption_qwen_vl_model.eval()

    return caption_qwen_vl_model, caption_qwen_vl_processor

def get_ram_percent():
    """
    Get current RAM usage percentage.
    Returns None if psutil is not available.
    """
    try:
        import psutil
        ram = psutil.virtual_memory()
        return ram.percent
    except ImportError:
        return None

def unload_caption_models():
    """
    Unload caption models from memory to free VRAM before training.
    This helps prevent out-of-memory errors when training with many images.
    """
    global caption_vit_model, caption_vit_feature_extractor, caption_vit_tokenizer
    global caption_blip_model, caption_blip_processor
    global caption_qwen_vl_model, caption_qwen_vl_processor
    
    try:
        import torch
    except ImportError:
        return
    
    # Unload ViT-GPT2 model
    if caption_vit_model is not None:
        try:
            if hasattr(caption_vit_model, 'to'):
                caption_vit_model.to('cpu')
            del caption_vit_model
        except Exception:
            pass
        caption_vit_model = None
    
    if caption_vit_feature_extractor is not None:
        caption_vit_feature_extractor = None
    
    if caption_vit_tokenizer is not None:
        caption_vit_tokenizer = None
    
    # Unload BLIP model
    if caption_blip_model is not None:
        try:
            if hasattr(caption_blip_model, 'to'):
                caption_blip_model.to('cpu')
            del caption_blip_model
        except Exception:
            pass
        caption_blip_model = None
    
    if caption_blip_processor is not None:
        caption_blip_processor = None
    
    # Unload Qwen-VL model
    if caption_qwen_vl_model is not None:
        try:
            if hasattr(caption_qwen_vl_model, 'to'):
                caption_qwen_vl_model.to('cpu')
            del caption_qwen_vl_model
        except Exception:
            pass
        caption_qwen_vl_model = None
    
    if caption_qwen_vl_processor is not None:
        caption_qwen_vl_processor = None
    
    # Clear CUDA cache to free up VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

DOWNLOADS = {
    "DiT Model": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors?download=true",
    "Text Encoder": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors?download=true",
    "VAE Model": "https://huggingface.co/Qwen/Qwen-Image/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true",
}

def download_with_progress(url, fname):
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            sys.stdout.write("\rDownloading %s %d%%" % (fname, percent))
            sys.stdout.flush()
        else:
            sys.stdout.write("\rDownloading %s read %d" % (fname, readsofar))
            sys.stdout.flush()
    urllib.request.urlretrieve(url, fname, reporthook)
    print("\nDownload finished")

HTML = r'''
<!DOCTYPE html>
<html>
<head>
  <title>MusubiTLX – Qwen LoRA Training Panel</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    html { scroll-behavior: smooth; }
    body { font-family: 'Inter', Arial, sans-serif; background: #222; color: #eee; margin:0;}
    .container { background: #2b2f33; box-shadow: 0 0 18px rgba(0,0,0,0.75); padding: 2.2em; max-width: 1100px; margin: 2.2em auto; border-radius: 22px;}
    .section { background: #23272E; border-radius:16px; margin-bottom: 24px; padding: 1.25em 1.7em;}
    .section h3 { margin-top:0; color: #86c6fe; font-size: 1.28em; font-weight:600; letter-spacing:0.02em; cursor: pointer; user-select: none; display: flex; align-items: center; justify-content: space-between; gap: 12px;}
    .section h3:hover { color: #00ddcb; }
    .section-toggle { font-size: 0.9em; color: #86c6fe; transition: transform 0.3s ease; display: inline-block; }
    .section.collapsed .section-toggle { transform: rotate(-90deg); }
    .section-content { overflow: hidden; transition: max-height 0.3s ease, opacity 0.2s ease, padding 0.3s ease; padding-top: 0.5em; }
    .section.collapsed .section-content { max-height: 0; opacity: 0; padding-top: 0; padding-bottom: 0; margin: 0; }
    .section:not(.collapsed) .section-content { max-height: 10000px; opacity: 1; }
    input, select, textarea { width: 96%; padding: 8px; margin: 6px 0 16px; border-radius: 6px; border: 1px solid #666; background: #1e2023; color: #eee; font-size: 1.09rem;}
    input[type="file"] { display: none; }
    .file-input-wrapper { position: relative; width: 96%; }
    .file-input-label { display: inline-block; padding: 10px 28px; border-radius: 8px; background: #0098FF; color: #fff; border: none; font-size: 1.07em; cursor: pointer; font-weight: 500; margin-bottom: 16px; }
    .file-input-label:hover { background: #00ddcb; }
    .file-input-clear-btn { display: inline-block; padding: 8px 20px; border-radius: 6px; background: #666; color: #fff; border: none; font-size: 0.95em; cursor: pointer; font-weight: 500; margin-left: 8px; margin-bottom: 16px; }
    .file-input-clear-btn:hover { background: #888; }
    .file-input-text { display: inline-block; margin-left: 12px; color: #aaa; font-size: 0.95em; }
    button { padding: 10px 28px; border-radius: 8px; background: #0098FF; color: #fff; border: none; font-size: 1.07em; cursor:pointer; font-weight:500;}
    button:hover { background: #00ddcb; }
    label { display: block; margin-top: 10px; font-weight: 600;}
    .output { background: #121315; color: #b1f59f; box-shadow:0 0 8px #1a2e1a; padding: 1.3em; border-radius: 10px; margin-top: 1.3em; white-space: pre-wrap;}
    .yaml-preview { background: #23272E; color: #ffeec1; padding: 1em; border-radius: 10px; font-size:1.01em; margin-bottom:20px;}
    #spinner { display:none; margin-top:30px; text-align:center;}
    .downloads button { background: #058D34;}
    .downloads button:hover { background: #00C85A; }
    .imglist { margin:12px 0 33px 0; display:flex; flex-wrap:wrap; gap:18px; min-height:120px; align-items:flex-start;}
    .imglist-empty { border: 2px dashed #444; border-radius: 12px; padding: 16px; justify-content:center; align-items:center; }
    .drop-hint { color:#aaa; font-size:0.95em; text-align:center; }
    .imglist.dragover { border-color: #00ddcb; background: rgba(0,221,203,0.04); }
    .gallery { display:flex; flex-wrap:wrap; gap:20px; margin-top:10px;}
    .imgbox { background:#151618; border-radius:13px; box-shadow:0 0 8px #444 inset; padding:13px; text-align:center; width:190px; position:relative;}
    .imgbox img { max-width:170px; max-height:170px; border-radius:7px; box-shadow:0 0 7px #262f2a; cursor: pointer; transition: transform 0.2s ease; }
    .imgbox img:hover { transform: scale(1.05); }
    .image-zoom-modal { display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.9); cursor: pointer; }
    .image-zoom-modal.active { display: flex; align-items: center; justify-content: center; }
    .image-zoom-content { max-width: 90%; max-height: 90%; position: relative; }
    .image-zoom-content img { max-width: 100%; max-height: 90vh; border-radius: 8px; box-shadow: 0 0 30px rgba(0, 152, 255, 0.5); }
    .image-zoom-close { position: absolute; top: -40px; right: 0; color: #fff; font-size: 32px; font-weight: bold; cursor: pointer; text-shadow: 0 0 10px rgba(0, 0, 0, 0.8); }
    .image-zoom-close:hover { color: #00ddcb; }
    .imgbox input[type=text] { width:97%; margin-top:10px; background:#222; color:#eee; border-radius:6px; border: 1px solid #666; }
    .caption-textarea { width:97%; margin-top:10px; background:#222; color:#eee; border-radius:6px; border:1px solid #666; font-size:0.9rem; line-height:1.25; min-height:3.2em; resize:vertical; }
    .imgbox-remove { position:absolute; top:5px; right:5px; background:#ff4444; color:#fff; border:none; border-radius:50%; width:24px; height:24px; cursor:pointer; font-size:16px; font-weight:bold; line-height:1; padding:0; display:flex; align-items:center; justify-content:center; opacity:0.8; }
    .imgbox-remove:hover { opacity:1; background:#ff6666; }
    .advanced-textarea { width:96%; background:#1e2023; color:#eee; border-radius:6px; border:1px solid #666; font-size:0.9rem; line-height:1.3; min-height:6em; }
    .title { 
      font-size: 2.4em; 
      background: linear-gradient(135deg, #0098FF 0%, #00ddcb 50%, #86c6fe 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: 0.5px; 
      margin-bottom: 0.5em; 
      text-align: left;
      font-weight: 700;
      text-shadow: 0 0 30px rgba(0, 152, 255, 0.3);
      position: relative;
      padding-bottom: 12px;
    }
    .title::after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 0;
      width: 100%;
      height: 2px;
      background: linear-gradient(90deg, #0098FF 0%, #00ddcb 100%);
      border-radius: 2px;
      opacity: 0.6;
    }
    .nav-links {
      display: flex;
      gap: 20px;
      margin-top: 16px;
      margin-bottom: 24px;
      flex-wrap: wrap;
      padding: 12px 20px;
      border-bottom: 1px solid rgba(0, 152, 255, 0.2);
      position: sticky;
      top: 0;
      background: #2b2f33;
      z-index: 100;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    .nav-links a {
      color: #86c6fe;
      text-decoration: none;
      font-size: 0.95em;
      font-weight: 500;
      padding: 6px 12px;
      border-radius: 6px;
      transition: all 0.2s ease;
      position: relative;
    }
    .nav-links a:hover {
      color: #00ddcb;
      background: rgba(0, 152, 255, 0.1);
      transform: translateY(-1px);
    }
    .nav-links a.active {
      color: #00ddcb;
      background: rgba(0, 152, 255, 0.15);
    }
    .nav-links a::after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 50%;
      transform: translateX(-50%);
      width: 0;
      height: 2px;
      background: linear-gradient(90deg, #0098FF 0%, #00ddcb 100%);
      transition: width 0.3s ease;
    }
    .nav-links a:hover::after,
    .nav-links a.active::after {
      width: 80%;
    }
    .model-list { margin-bottom: 15px; }
    .model-list form { display: inline; }
    .model-list button { margin-right: 8px; }
    .footer { margin-top: 24px; font-size: 0.9em; color:#999; text-align:right; }
    .footer a { color:#7abfff; text-decoration:none; margin-left:12px; }
    .footer a:hover { text-decoration:underline; }
    .status-box { margin-top:8px; font-size:0.9em; color:#b3daff; word-wrap: break-word; overflow-wrap: break-word; max-width: 100%; }
    .validation-error { background: #3d1a1a; border: 2px solid #ff4444; border-radius: 8px; padding: 16px; margin: 16px 0; color: #ffaaaa; }
    .validation-error h4 { margin: 0 0 12px 0; color: #ff6666; font-size: 1.1em; }
    .validation-error ul { margin: 8px 0 0 20px; padding: 0; }
    .validation-error li { margin: 6px 0; }
    .field-error { border: 2px solid #ff4444 !important; background: #2a1a1a !important; }
    .progress-bar-container { width: 100%; background: #1a1d24; border-radius: 8px; height: 24px; margin-top: 8px; overflow: hidden; border: 1px solid #444; position: relative; }
    .progress-bar-fill { height: 100%; background: linear-gradient(90deg, #0098FF, #00ddcb); transition: width 0.3s ease; display: flex; align-items: center; justify-content: center; color: #fff; font-size: 0.75em; font-weight: 600; }
    .progress-bar-text { width: 100%; text-align: center; color: #b3daff; font-size: 0.85em; margin-top: 4px; word-wrap: break-word; overflow-wrap: break-word; max-width: 100%; }
    .cmd-preview { background:#111317; border-radius:8px; padding:8px 10px; margin-top:6px; font-size:0.9em; color:#d8f4ff; max-height:140px; overflow-y:auto; white-space:pre-wrap; }
    .loss-chart-container { background:#1a1d24; border-radius:8px; padding:16px; margin-top:16px; border:1px solid #444; }
    .loss-chart-container h4 { margin:0 0 12px 0; color:#86c6fe; font-size:1em; font-weight:600; }
    #loss-chart-canvas { max-height:300px; }
    .info-icon-wrapper { display: inline-flex; align-items: center; gap: 6px; }
    .info-icon { display: inline-block; width: 18px; height: 18px; border-radius: 50%; background: #0098FF; color: #fff; text-align: center; line-height: 18px; font-size: 12px; font-weight: bold; cursor: help; position: relative; flex-shrink: 0; }
    .info-icon:hover { background: #00ddcb; }
    .info-tooltip { visibility: hidden; opacity: 0; position: absolute; bottom: 125%; left: 50%; transform: translateX(-50%); background-color: #1a1d24; color: #eee; text-align: left; padding: 12px 14px; border-radius: 8px; border: 1px solid #444; box-shadow: 0 4px 12px rgba(0,0,0,0.6); z-index: 1000; width: 380px; font-size: 0.9em; font-weight: normal; line-height: 1.5; white-space: normal; transition: opacity 0.3s, visibility 0.3s; pointer-events: none; }
    .info-icon:hover .info-tooltip { visibility: visible; opacity: 1; }
    .info-tooltip::after { content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: #444 transparent transparent transparent; }
    .info-tooltip strong { color: #86c6fe; display: block; margin-bottom: 4px; }
    .info-tooltip ul { margin: 6px 0; padding-left: 20px; }
    .info-tooltip li { margin: 4px 0; }
    .unload-icon { display: inline-block; width: 16px; height: 16px; border-radius: 3px; background: transparent; color: #888; text-align: center; line-height: 16px; font-size: 13px; cursor: pointer; position: relative; flex-shrink: 0; margin-left: 6px; opacity: 0.6; transition: opacity 0.2s, color 0.2s; }
    .unload-icon:hover { opacity: 1; color: #ff6b6b; background: rgba(255, 107, 107, 0.1); }
    .unload-icon:disabled { opacity: 0.3; cursor: not-allowed; }
    .sample-section { background:#1b1f24; border:1px solid #32363f; border-radius:14px; padding:16px 18px; margin-top:18px; }
    .sample-toggle-row { display:flex; align-items:center; gap:12px; }
    .sample-toggle-row input[type="checkbox"] { width:auto; transform:scale(1.2); }
    .sample-gallery { display:flex; flex-wrap:wrap; gap:16px; margin-top:12px; }
    .sample-card { width:190px; background:#151618; border-radius:12px; padding:10px; box-shadow:0 0 8px rgba(0,0,0,0.4); transition:transform 0.2s ease; }
    .sample-card:hover { transform:translateY(-3px); box-shadow:0 6px 16px rgba(0,0,0,0.45); }
    .sample-card a { color:inherit; text-decoration:none; display:block; }
    .sample-card img, .sample-card video { width:100%; border-radius:8px; object-fit:cover; max-height:150px; background:#0f1012; cursor: pointer; }
    .sample-card-meta { font-size:0.85em; color:#bbb; margin-top:6px; word-break:break-word; }
    .sample-gallery-empty { border:1px dashed #444; border-radius:10px; padding:14px; color:#777; font-size:0.95em; display:flex; justify-content:center; align-items:center; min-height:80px; width:100%; }
    .form-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 12px; }
    .field-group { display: flex; flex-direction: column; }
    .field-group-half { grid-column: span 1; }
    .field-group-wide { grid-column: span 2; }
    .field-group button { width: auto; align-self: flex-start; }
    .button-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 20px; }
    .sample-section { grid-column: span 2; }
    #sample-options { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
    #sample-options > .field-group-half { grid-column: span 1; width: 100%; }
    @media (max-width: 768px) {
      .form-grid { grid-template-columns: 1fr; }
      .field-group-half, .field-group-wide, .sample-section { grid-column: span 1; }
      #sample-options { grid-template-columns: 1fr; }
      #sample-options > .field-group-half { grid-column: span 1; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="title">MusubiTLX – Qwen LoRA Training</div>
    <div class="nav-links">
      <a href="#model-files">Model Files</a>
      <a href="#upload-images">Upload Images</a>
      <a href="#training-settings">Training Settings</a>
      <a href="#preview">Preview & Captions</a>
      <a href="#system-resources">System Resources</a>
      <a href="#training-output">Training Output</a>
    </div>
    <div class="section model-list" id="model-files">
      <h3 onclick="window.toggleSection && window.toggleSection('model-files')">
        <span>Model files (download from Hugging Face)</span>
        <span class="section-toggle">▼</span>
      </h3>
      <div class="section-content">
      {% for label, url in downloads.items() %}
        <form method="POST">
          <input type="hidden" name="dl_label" value="{{label}}">
          <input type="hidden" name="dl_url" value="{{url}}">
          <button type="submit" name="action" value="download">{{label}}</button>
        </form>
      {% endfor %}
      {% if download_msg %}<div style="margin-top:10px;">{{download_msg}}</div>{% endif %}
      </div>
    </div>
    <form id="training-form" method="POST" enctype="multipart/form-data">
      <div class="section" id="upload-images">
        <h3 onclick="window.toggleSection && window.toggleSection('upload-images')">
          <span>1. Upload training images</span>
          <span class="section-toggle">▼</span>
        </h3>
        <div class="section-content">
        <label>Select images (multiple allowed)</label>
        <div class="file-input-wrapper">
          <label for="images" class="file-input-label">Choose images</label>
          <button type="button" class="file-input-clear-btn" id="clear-images-btn" onclick="window.clearSelectedImages && window.clearSelectedImages()" style="display:none;">Clear all</button>
          <span class="file-input-text" id="file-input-status">No files selected</span>
          <input type="file" id="images" name="images" multiple accept="image/*" onchange="window.handleFileInputChange && window.handleFileInputChange(event)">
        </div>
        <div class="imglist imglist-empty" id="preview" style="scroll-margin-top: 100px;">
          <div class="drop-hint">
            Drag &amp; drop images here or use the file picker above.
          </div>
        </div>
      <div id="autocap-status" class="status-box" style="display:none; margin-top:12px;"></div>
      <div id="autocap-progress-container" style="display:none; margin-top:12px;">
        <div class="progress-bar-container">
          <div id="autocap-progress-bar" class="progress-bar-fill" style="width: 0%;"></div>
        </div>
      </div>
      <button type="button" id="autocap-btn" onclick="window.autoCaptionCaptionModel && window.autoCaptionCaptionModel()" style="margin-top:16px;">
        Auto-caption images
      </button>
      <label class="info-icon-wrapper" style="margin-top:16px;">
        Caption model (auto-caption)
        <span class="info-icon">
          ?
          <span class="info-tooltip">
            <strong>Captioning Dependencies:</strong>
            Captioning dependencies must be installed manually in your Musubi Tuner virtual environment before this feature works.
            <br><br>
            See the GUI README for the exact <code>pip install</code> command.
            <br><br>
            <strong>Note:</strong> The first time you use a caption model, it will be automatically downloaded from Hugging Face. Qwen-VL is a large file (~14GB) and may take 10-15 minutes to download.
          </span>
        </span>
      </label>
      <select id="caption_model" name="caption_model">
        <option value="vit-gpt2" selected>ViT-GPT2 – fast, lightweight</option>
        <option value="blip-large">BLIP large – more detailed (slower, more VRAM)</option>
        <option value="qwen-vl">Qwen-VL – extremely detailed (slowest, most VRAM)</option>
      </select>
      <label style="margin-top:8px;">
        Caption length
      </label>
      <input type="range" id="caption_max_length" name="caption_max_length" min="32" max="192" step="8" value="128" oninput="window.updateCaptionLengthLabel && window.updateCaptionLengthLabel()">
      <div style="font-size:0.85em; color:#aaa; margin-top:2px;">
        Target length: <span id="caption_length_label">128</span> tokens (approximate, longer = more detail)
      </div>
      <label style="margin-top:12px;">
        Detail level (BLIP only)
      </label>
      <input type="range" id="caption_detail_level" name="caption_detail_level" min="1" max="5" step="1" value="3" oninput="window.updateDetailLevelLabel && window.updateDetailLevelLabel()">
      <div style="font-size:0.85em; color:#aaa; margin-top:2px;">
        Level: <span id="detail_level_label">3</span> / 5 (<span id="detail_level_desc">Detailed</span>)
      </div>
      <div style="font-size:0.8em; color:#888; margin-top:4px; font-style:italic;">
        Note: BLIP-large has inherent limitations and may not describe very fine details like specific lighting conditions or facial features in extreme detail, even at level 5.
      </div>
        <input type="hidden" name="uploaded_count" id="uploaded_count">
      </div>
      </div>
      <div class="section" id="training-settings">
        <h3 onclick="window.toggleSection && window.toggleSection('training-settings')">
          <span>2. Training settings</span>
          <span class="section-toggle">▼</span>
        </h3>
        <div class="section-content">
        <div class="form-grid">
          <div class="field-group field-group-wide">
            <label class="info-icon-wrapper">
              Training preset
              <span class="info-icon">
                ?
                <span class="info-tooltip">
                  <strong>Training Presets:</strong>
                  Choose a preset to automatically configure training parameters, or select "Custom" to set everything manually.
                  <br><br>
                  <strong>Presets:</strong>
                  <ul>
                    <li><strong>Custom:</strong> Manual control over all settings</li>
                    <li><strong>Fast Training:</strong> Quick tests, rapid prototyping (2 epochs, LR: 5e-5, rank 16, alpha 16, ~16 total repeats/image)</li>
                    <li><strong>Balanced:</strong> General purpose, recommended default (3 epochs, LR: 4e-5, rank 32, alpha 32, ~18 total repeats/image)</li>
                    <li><strong>High Quality:</strong> Detailed textures, production quality (4 epochs, LR: 3e-5, rank 64, alpha 32, ~20 total repeats/image)</li>
                    <li><strong>Person/Character:</strong> Training specific people/characters (3 epochs, LR: 5e-5, rank 32, alpha 32, 50-60 total repeats for likeness)</li>
                    <li><strong>Style/Concept:</strong> Artistic styles, abstract concepts (4 epochs, LR: 2e-5, rank 16, alpha 8, ~20 total repeats/image)</li>
                    <li><strong>Few Images (&lt;30):</strong> Small datasets, prevents overfitting (5 epochs, LR: 2e-5, rank 16, alpha 16, 50+ total repeats minimum)</li>
                    <li><strong>Many Images (100+):</strong> Large datasets, comprehensive training (3 epochs, LR: 5e-5, rank 64, alpha 32, ~6 total repeats/image)</li>
                  </ul>
                  <strong>Note:</strong> Epochs, repeats, and learning rate are automatically adjusted based on the number of uploaded images to optimize training.
                  <strong>Note:</strong> Adjust presets based on your specific dataset and needs.
                </span>
              </span>
            </label>
            <select id="training-preset" name="training_preset" onchange="window.applyTrainingPreset && window.applyTrainingPreset()">
              <option value="custom" selected>Custom (manual settings)</option>
              <option value="fast">Fast Training</option>
              <option value="balanced">Balanced</option>
              <option value="high-quality">High Quality</option>
              <option value="person">Person/Character</option>
              <option value="style">Style</option>
              <option value="few-images">Few Images (&lt;30 images)</option>
              <option value="many-images">Many Images (100+ images)</option>
            </select>
            <small style="color: #888; display: block; margin-top: -8px; margin-bottom: 10px;">
              Select a preset to auto-fill training parameters, or choose "Custom" to set everything manually.
            </small>
          </div>
          <div class="field-group">
            <label>Trigger word (label used in captions)</label>
            <input name="trigger" type="text" value="{{trigger}}">
          </div>
          <div class="field-group field-group-half">
            <label>Your GPU VRAM (select your graphics card's VRAM)</label>
            <select name="vram_profile" onchange="window.updateEstimate && window.updateEstimate(); window.syncAdvancedFlagsFromGui && window.syncAdvancedFlagsFromGui(false);">
              <option value="12" {% if vram_profile == '12' %}selected{% endif %}>I have 12GB VRAM – uses ~12GB VRAM, offloads most to RAM (64GB+ RAM recommended, slowest but safest)</option>
              <option value="16" {% if vram_profile == '16' %}selected{% endif %}>I have 16GB VRAM – uses ~16-18GB VRAM, offloads some to RAM (64GB+ RAM recommended, balanced)</option>
              <option value="24" {% if vram_profile == '24' %}selected{% endif %}>I have 24GB+ VRAM – uses ~24GB VRAM, offloads minimal to RAM (64GB+ RAM recommended, fastest)</option>
            </select>
            <small style="color: #888; display: block; margin-top: -12px; margin-bottom: 10px;">
              Settings are automatically adjusted based on your GPU VRAM. The Qwen-Image model is ~38GB total, so model blocks are offloaded to RAM (64GB+ RAM recommended). 
              <strong>16GB VRAM GPUs get optimized settings (~16-18GB VRAM usage) for better performance than 12GB GPUs.</strong>
            </small>
          </div>
          <div class="field-group field-group-half">
            <label>Epochs (how many passes over the dataset)</label>
            <input name="epochs" type="number" value="{{epochs}}" oninput="window.updateEstimate && window.updateEstimate()">
          </div>
          <div class="field-group field-group-half">
            <label>Batch size</label>
            <input name="batch_size" type="number" value="{{batch_size}}" oninput="window.updateEstimate && window.updateEstimate()">
          </div>
          <div class="field-group field-group-half">
            <label>Image repeats (how many times each image is seen)</label>
            <input name="image_repeats" type="number" value="{{image_repeats}}" oninput="window.updateEstimate && window.updateEstimate()">
          </div>
          <div class="field-group field-group-half">
            <label class="info-icon-wrapper">
              Learning rate
              <span class="info-icon">
                ?
                <span class="info-tooltip">
                  <strong>Learning Rate:</strong>
                  Controls how quickly the model learns. Higher values = faster learning but may be unstable. Lower values = more stable but slower.
                  <br><br>
                  <strong>Common values:</strong>
                  <ul>
                    <li><strong>1e-4</strong> = 0.0001 (high, fast learning, may overfit)</li>
                    <li><strong>5e-5</strong> = 0.00005 (default, balanced)</li>
                    <li><strong>1e-5</strong> = 0.00001 (low, stable, slower)</li>
                    <li><strong>5e-6</strong> = 0.000005 (very low, very stable, very slow)</li>
                    <li><strong>1e-6</strong> = 0.000001 (extremely low, minimal changes)</li>
                  </ul>
                  <strong>Recommendation:</strong> Start with <strong>5e-5</strong> (0.00005). Use lower values (1e-5) for fine-tuning existing LoRAs or if you notice overfitting.
                </span>
              </span>
            </label>
            <input name="lr" type="text" value="{{lr}}">
          </div>
          <div class="field-group field-group-half">
            <label>Optimizer</label>
            <select name="optimizer_type">
              <option value="AdamW" {% if optimizer_type == 'AdamW' %}selected{% endif %}>AdamW (standard)</option>
              <option value="Adafactor" {% if optimizer_type == 'Adafactor' %}selected{% endif %}>Adafactor (more memory friendly)</option>
              <option value="AdamW8bit" {% if optimizer_type == 'AdamW8bit' %}selected{% endif %}>AdamW 8‑bit (requires bitsandbytes)</option>
            </select>
          </div>
          <div class="field-group">
            <label>Prompt / subject description</label>
            <input name="prompt" type="text" value="{{prompt}}">
          </div>
          <div class="field-group field-group-half">
            <label>Resolution (e.g. 1024x1024)</label>
            <input name="resolution" type="text" value="{{resolution}}" oninput="window.updateEstimate && window.updateEstimate()">
          </div>
          <div class="field-group field-group-half">
            <label>Seed (same seed = repeatable result)</label>
            <input name="seed" type="number" value="{{seed}}">
          </div>
          <div class="field-group field-group-half">
            <label class="info-icon-wrapper">
              LoRA rank (lower = less VRAM)
              <span class="info-icon">
                ?
                <span class="info-tooltip">
                  <strong>LoRA Rank (capacity):</strong>
                  Controls how much detail the model can learn. Higher rank = more detail but more VRAM usage.
                  <br><br>
                  <strong>Recommendations:</strong>
                  <ul>
                    <li><strong>16:</strong> Low capacity, okay for simple styles</li>
                    <li><strong>32:</strong> Good for people and faces</li>
                    <li><strong>64:</strong> Very good for detailed faces</li>
                    <li><strong>128:</strong> Maximum (may overfit with few images)</li>
                  </ul>
                  For training on a real person: use <strong>32-64</strong>.
                </span>
              </span>
            </label>
            <input name="rank" type="number" value="{{rank}}">
          </div>
          <div class="field-group field-group-half">
            <label class="info-icon-wrapper">
              LoRA dims (lower = less VRAM)
              <span class="info-icon">
                ?
                <span class="info-tooltip">
                  <strong>LoRA Dims (alpha):</strong>
                  Controls the strength of the LoRA adaptation. Should be 1x-2x the rank value.
                  <br><br>
                  <strong>Recommendations:</strong>
                  <ul>
                    <li>If rank = 16 → dims = 16-32</li>
                    <li>If rank = 32 → dims = 32-64</li>
                    <li>If rank = 64 → dims = 64-128</li>
                    <li>If rank = 128 → dims = 128-256</li>
                  </ul>
                  Default: dims = rank × 2 (e.g., rank 32 → dims 64)
                </span>
              </span>
            </label>
            <input name="dims" type="number" value="{{dims}}">
          </div>
          <div class="field-group field-group-wide">
            <label>Output folder (under <code>output/</code>)</label>
            <input name="output_dir" type="text" value="{{user_output_dir}}" placeholder="empty = use 'output'" oninput="window.updateFinalPath && window.updateFinalPath()">
            <small style="color: #888; display: block; margin-top: -8px; margin-bottom: 6px;">If you enter <code>art</code> files will be saved in <code>output/art/</code></small>
          </div>
          <div class="field-group field-group-wide">
            <label>LoRA filename (.safetensors)</label>
            <input name="output_name" type="text" value="{{output_name}}" placeholder="lora" required oninput="window.updateFinalPath && window.updateFinalPath()">
            <small style="color: #888; display: block; margin-top: -8px; margin-bottom: 4px;">Final LoRA file: <code id="final-path-preview">output/{% if user_output_dir %}{{user_output_dir}}/{% endif %}{{output_name}}.safetensors</code></small>
          </div>
          <div class="field-group field-group-wide">
            <button type="button" id="advanced-toggle">
              Show advanced trainer flags
            </button>
            <div id="advanced-flags-panel" style="display:none; margin-top:10px;">
              <label>Advanced trainer flags (optional)</label>
              <textarea name="advanced_flags" class="advanced-textarea" rows="5" placeholder="Example: --network_dropout 0.05 --max_grad_norm 0.5">{{advanced_flags}}</textarea>
              <small style="color:#888; display:block; margin-top:4px;">
                These are the extra flags that will be passed to the trainer (including VRAM-related flags). You can edit them directly if you know what you are doing.
              </small>
              <label style="margin-top:10px;">Effective training command (preview)</label>
              <pre id="cmd-preview" class="cmd-preview"></pre>
              <small style="color:#777; display:block; margin-top:4px;">
                Read-only preview of the <code>python qwen_image_train_network.py</code> command that will be used when you click "Start training".
              </small>
            </div>
          </div>
          <div class="sample-section">
          <div class="sample-toggle-row">
            <label class="info-icon-wrapper" style="margin:0; gap:10px;">
              <input type="checkbox" id="samples_enabled" name="samples_enabled" {% if samples_enabled %}checked{% endif %} onchange="window.toggleSampleOptions && window.toggleSampleOptions(); window.updateCommandPreview && window.updateCommandPreview();">
              <span style="font-weight:600;">Generate sample previews during training</span>
              <span class="info-icon">
                ?
                <span class="info-tooltip">
                  <strong>Sample previews:</strong>
                  Runs Qwen inference at checkpoints and saves PNG files under <code>output/.../sample</code>.
                  <ul>
                    <li>Costs extra VRAM/time (especially with FP8 offload).</li>
                    <li>Only enable if you need automatic previews.</li>
                  </ul>
                </span>
              </span>
            </label>
          </div>
          <small style="color:#8ea2be; display:block; margin-top:6px; margin-bottom:12px;">
            When enabled, the prompts below are saved to <code>sample_prompts.txt</code> inside your output folder and passed to the trainer automatically.
          </small>
          <div id="sample-options" style="margin-top:10px; {% if not samples_enabled %}display:none;{% endif %}">
            <label>Sample prompts (one prompt per line)</label>
            <textarea id="sample_prompt_text" name="sample_prompt_text" class="advanced-textarea" rows="6" placeholder="# Example prompts (one per line):&#10;A studio portrait of {{trigger}} dressed as a cyberpunk hero --w 960 --h 960 --s 30&#10;{{trigger}} in a fantasy landscape, detailed --w 1024 --h 768 --s 25">{{sample_prompt_text}}</textarea>
            <small style="color:#888; display:block; margin-top:4px; margin-bottom:8px;">
              Write prompts with optional parameters. Each line = one sample image.
              <br>Available parameters: <code>--w</code> (width), <code>--h</code> (height), <code>--s</code> (steps, default: 20), <code>--d</code> (seed), <code>--f</code> (frames, keep 1 for still images).
              <br>Lines starting with <code>#</code> are comments. Use <code>{{trigger}}</code> to insert your trigger word.
            </small>
            <div class="field-group field-group-half" style="margin-top:10px;">
              <label>Sample every N epochs (0 = disable)</label>
              <input name="sample_every_epochs" type="number" min="0" value="{% if sample_every_epochs is defined %}{{sample_every_epochs}}{% else %}1{% endif %}" placeholder="1" oninput="window.updateCommandPreview && window.updateCommandPreview()">
              <small style="color:#888; display:block; margin-top:2px;">Default: 1 (sample after each epoch)</small>
            </div>
            <div class="field-group field-group-half" style="margin-top:10px;">
              <label>Sample every N steps (0 = disable)</label>
              <input name="sample_every_steps" type="number" min="0" value="{% if sample_every_steps is defined %}{{sample_every_steps}}{% else %}0{% endif %}" placeholder="0" oninput="window.updateCommandPreview && window.updateCommandPreview()">
              <small style="color:#888; display:block; margin-top:2px;">Default: 0 (use epochs instead)</small>
            </div>
            <div class="sample-toggle-row" style="margin-top:10px;">
              <label class="info-icon-wrapper" style="margin:0; gap:10px;">
                <input type="checkbox" id="sample_at_first" name="sample_at_first" {% if sample_at_first %}checked{% endif %} onchange="window.updateCommandPreview && window.updateCommandPreview()">
                <span style="font-weight:600;">Generate baseline samples before training starts</span>
                <span class="info-icon">
                  ?
                  <span class="info-tooltip">
                    <strong>Baseline Samples:</strong>
                    Generate sample images before training begins (with the untrained model) so you can compare before/after results.
                    <br><br>
                    This helps you see how much the model improves during training and verify that sample generation works correctly.
                  </span>
                </span>
              </label>
            </div>
            <div style="margin-top:18px;">
              <button type="button" onclick="window.refreshSampleGallery && window.refreshSampleGallery()" style="background:#7a5cff;">Refresh sample gallery</button>
              <small style="color:#888; display:block; margin-top:4px;">
                Uses the current output folder (shown above). Samples save to <code>{{output_dir if output_dir else 'output'}}/sample/</code>.
              </small>
            </div>
            <div id="sample-gallery-status" class="status-box" style="display:none; margin-top:10px;"></div>
            <div id="sample-gallery" class="sample-gallery sample-gallery-empty" style="margin-top:12px;">
              <span>No sample previews found yet.</span>
            </div>
          </div>
        </div>
        </div>
        <div style="color:#aaa; font-size:0.9em; margin-bottom:6px; margin-top:4px;">
          <span id="time-estimate">Estimated training time: add images and adjust epochs/batch size.</span>
        </div>
        <div id="few-images-warning" style="display:none; background: #332211; border-left: 4px solid #ffaa44; padding: 10px; margin-top: 10px; border-radius: 4px; color: #ffdd88;">
          <strong>⚠️ Warning: Few training images detected</strong>
          <br>
          Training with very few images may not produce good results for person training. 
          <strong>Recommendations:</strong>
          <ul style="margin: 8px 0 0 20px; padding: 0;">
            <li>Add more images (10-20 ideal for person training)</li>
            <li>Increase epochs to 5-10 for more training steps</li>
            <li>Use lower learning rate (1e-5) to reduce overfitting</li>
            <li>Make captions more specific about the person's unique features</li>
          </ul>
        </div>
        <div class="button-row">
          <button type="button" id="start-btn" onclick="window.startTraining && window.startTraining()">Start training</button>
          <button type="button" id="cancel-btn" onclick="window.cancelTraining && window.cancelTraining()">Cancel training</button>
          <button type="submit" id="saveyaml-btn" name="action" value="saveyaml">Save configuration as YAML</button>
          <button type="submit" id="loadyaml-btn" name="action" value="loadyaml">Load last YAML config</button>
        </div>
        </div>
      </div>
    </form>
    {% if uploaded_gallery %}
      <div class="section" id="preview-captions">
        <h3 onclick="window.toggleSection && window.toggleSection('preview-captions')">
          <span>3. Preview images & captions</span>
          <span class="section-toggle">▼</span>
        </h3>
        <div class="section-content">
        <div class="gallery">
        {% for fname, cap in uploaded_gallery %}
          <div class="imgbox">
            <img src="{{url_for('uploaded_image', filename=fname)}}">
            <div style="word-break:break-word; font-size:1.05em; margin-top:8px;">{{cap}}</div>
          </div>
        {% endfor %}
        </div>
        </div>
      </div>
    {% endif %}
    <div id="spinner">
      <span style="font-size:2em;">🔄 Training in progress... Please wait.<br></span>
      <div style="margin-top:10px;">
        <svg width="50" height="50" viewBox="0 0 50 50">
          <circle cx="25" cy="25" r="20" fill="none" stroke="#0077FF" stroke-width="6"
            stroke-dasharray="31.4 31.4" transform="rotate(-90 25 25)">
            <animateTransform attributeName="transform" type="rotate"
              from="0 25 25" to="360 25 25" dur="1s" repeatCount="indefinite"/>
          </circle>
        </svg>
      </div>
    </div>
    <div class="section" style="margin-top: 20px;" id="system-resources">
      <h3 onclick="window.toggleSection && window.toggleSection('system-resources')" style="margin-top:0; color: #86c6fe; font-size: 1.28em; font-weight:600; letter-spacing:0.02em;">
        <span>System Resources</span>
        <span class="section-toggle">▼</span>
      </h3>
      <div class="section-content">
      <div id="system-resources-content" style="background: #1a1d24; border-radius: 8px; padding: 12px 16px; border: 1px solid #444;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
          <div style="flex: 1; min-width: 200px; display: flex; align-items: center;">
            <strong style="color: #86c6fe; font-size: 0.95em;">RAM:</strong>
            <span id="ram-info" style="color: #ddd; font-size: 0.9em; margin-left: 8px;">Loading...</span>
            <span class="unload-icon" id="force-unload-ram-icon" onclick="window.forceUnloadRAM && window.forceUnloadRAM()" title="Click to unload RAM">
              🗑️
              <span class="info-tooltip" style="width: 320px;">
                <strong>Force Unload RAM</strong>
                Aggressively unloads caption models, runs multiple garbage collection passes, clears PyTorch pinned memory, and attempts to clear system-level cached RAM (e.g., from ComfyUI or other processes).
                <br><br>
                <strong>Note:</strong> System cache clearing requires root permissions. If training is active, you'll be asked to confirm before unloading.
              </span>
            </span>
          </div>
          <div style="flex: 1; min-width: 200px; display: flex; align-items: center;">
            <strong style="color: #86c6fe; font-size: 0.95em;">VRAM:</strong>
            <span id="vram-info" style="color: #ddd; font-size: 0.9em; margin-left: 8px;">Loading...</span>
            <span class="unload-icon" id="force-unload-vram-icon" onclick="window.forceUnloadVRAM && window.forceUnloadVRAM()" title="Click to unload VRAM">
              🗑️
              <span class="info-tooltip" style="width: 320px;">
                <strong>Force Unload VRAM</strong>
                Unloads caption models from GPU memory and clears CUDA cache to free up VRAM. This is useful if VRAM isn't freed after canceling training.
                <br><br>
                <strong>Note:</strong> If training is active, you'll be asked to confirm before unloading.
              </span>
            </span>
          </div>
        </div>
        <small id="unload-status" style="color: #8ea2be; display: none; margin-top: 8px;"></small>
      </div>
    </div>
    </div>
    <div class="section" id="training-output" style="margin-top: 20px;">
      <h3 onclick="window.toggleSection && window.toggleSection('training-output')" style="margin-top:0; color: #86c6fe; font-size: 1.28em; font-weight:600; letter-spacing:0.02em;">
        <span>Training Output</span>
        <span class="section-toggle">▼</span>
      </h3>
      <div class="section-content">
    <div id="output-container" style="display:none;">
      <div class="output">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
          <strong>Training log:</strong>
          <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
            <button type="button" onclick="window.resumeLogStream && window.resumeLogStream()">Resume log stream</button>
            <small id="log-resume-status" style="color:#8ea2be; display:none;"></small>
          </div>
        </div>
        <pre id="output-text" style="max-height: 500px; overflow-y: auto; margin-top:10px;"></pre>
        <div id="loss-chart-container" class="loss-chart-container" style="display:none;">
          <h4>Training Loss Over Time</h4>
          <canvas id="loss-chart-canvas"></canvas>
        </div>
        <div id="readme-link-container" style="display:none; margin-top:12px; padding-top:12px; border-top:1px solid #444;">
          <a id="readme-link" href="#" target="_blank" style="display:inline-block; padding:8px 16px; background:#7a5cff; color:#fff; text-decoration:none; border-radius:6px; font-weight:500;">
            📄 View LoRA README.md
          </a>
          <small style="display:block; color:#888; margin-top:6px;">Ready to publish on CivitAI or other platforms</small>
        </div>
        </div>
      </div>
    </div>
    <div class="footer">
      MusubiTLX Web GUI created by TLX
      <a href="/musubitlx_gui_readme" target="_blank">GUI README</a>
      <span id="server-info" style="margin-left: 12px; color: #666; font-size: 0.85em;"></span>
    </div>
    <!-- Image zoom modal -->
    <div id="image-zoom-modal" class="image-zoom-modal" onclick="window.closeImageZoom && window.closeImageZoom()">
      <div class="image-zoom-content" onclick="event.stopPropagation()">
        <span class="image-zoom-close" onclick="window.closeImageZoom && window.closeImageZoom()">&times;</span>
        <img id="zoomed-image" src="" alt="Zoomed image">
      </div>
    </div>
  </div>
  <script>
    // Section collapse/expand functionality
    function toggleSection(sectionId) {
      const section = document.getElementById(sectionId);
      if (!section) return;
      
      section.classList.toggle('collapsed');
      
      // Save state to localStorage
      try {
        const collapsedSections = JSON.parse(localStorage.getItem('musubitlx_collapsed_sections') || '[]');
        const index = collapsedSections.indexOf(sectionId);
        
        if (section.classList.contains('collapsed')) {
          // Add to collapsed list if not already there
          if (index === -1) {
            collapsedSections.push(sectionId);
          }
        } else {
          // Remove from collapsed list
          if (index !== -1) {
            collapsedSections.splice(index, 1);
          }
        }
        
        localStorage.setItem('musubitlx_collapsed_sections', JSON.stringify(collapsedSections));
      } catch (err) {
        console.warn('Failed to save section state:', err);
      }
    }
    
    function restoreSectionStates() {
      try {
        const collapsedSections = JSON.parse(localStorage.getItem('musubitlx_collapsed_sections') || '[]');
        collapsedSections.forEach(sectionId => {
          const section = document.getElementById(sectionId);
          if (section) {
            section.classList.add('collapsed');
          }
        });
      } catch (err) {
        console.warn('Failed to restore section states:', err);
      }
    }
    
    // Make functions available globally immediately
    window.toggleSection = toggleSection;
    window.toggleAdvancedFlags = function() {
      const panel = document.getElementById('advanced-flags-panel');
      const toggle = document.getElementById('advanced-toggle');
      if (!panel || !toggle) return;
      const visible = panel.style.display === "block";
      panel.style.display = visible ? "none" : "block";
      toggle.textContent = visible ? "Show advanced trainer flags" : "Hide advanced trainer flags";
      if (!visible) {
        if (typeof updateCommandPreview === 'function') {
          updateCommandPreview();
        }
      }
    };
    
    window.toggleSampleOptions = function() {
      const checkbox = document.getElementById('samples_enabled');
      const options = document.getElementById('sample-options');
      if (!checkbox || !options) return;
      options.style.display = checkbox.checked ? "block" : "none";
    };
    
    // Force unload functions - make available globally
    window.forceUnloadVRAM = async function() {
      const spinner = document.getElementById("spinner");
      if (spinner && spinner.style.display === "block") {
        const confirmed = window.confirm("⚠️ Training is currently active!\n\nAre you sure you want to force unload VRAM? This may cause training to fail.\n\nClick OK to proceed anyway, or Cancel to abort.");
        if (!confirmed) return;
      }
      const icon = document.getElementById("force-unload-vram-icon");
      if (icon) {
        icon.style.pointerEvents = "none";
        icon.style.opacity = "0.3";
      }
      const statusEl = document.getElementById('unload-status');
      if (statusEl) {
        statusEl.style.display = "inline";
        statusEl.style.color = "#8ea2be";
        statusEl.textContent = "Unloading VRAM...";
      }
      try {
        const resp = await fetch('/force_unload_vram', { method: 'POST' });
        const data = await resp.json();
        if (statusEl) {
          if (data.status === "ok") {
            statusEl.textContent = "✅ " + (data.message || "VRAM unloaded successfully");
            statusEl.style.color = "#8ea2be";
            if (typeof updateSystemResources === 'function') {
              updateSystemResources();
            }
          } else {
            statusEl.textContent = "❌ " + (data.message || "Failed to unload VRAM");
            statusEl.style.color = "#ff8080";
          }
          setTimeout(() => {
            if (statusEl) {
              statusEl.style.display = "none";
              statusEl.textContent = "";
            }
          }, 5000);
        }
      } catch (err) {
        if (statusEl) {
          statusEl.textContent = "❌ Error: " + err.message;
          statusEl.style.color = "#ff8080";
        }
      } finally {
        if (icon) {
          icon.style.pointerEvents = "auto";
          icon.style.opacity = "0.6";
        }
      }
    };
    
    window.forceUnloadRAM = async function() {
      const spinner = document.getElementById("spinner");
      if (spinner && spinner.style.display === "block") {
        const confirmed = window.confirm("⚠️ Training is currently active!\n\nAre you sure you want to force unload RAM? This may cause training to fail.\n\nClick OK to proceed anyway, or Cancel to abort.");
        if (!confirmed) return;
      }
      const icon = document.getElementById("force-unload-ram-icon");
      if (icon) {
        icon.style.pointerEvents = "none";
        icon.style.opacity = "0.3";
      }
      const statusEl = document.getElementById('unload-status');
      if (statusEl) {
        statusEl.style.display = "inline";
        statusEl.style.color = "#8ea2be";
        statusEl.textContent = "Unloading RAM...";
      }
      try {
        const resp = await fetch('/force_unload_ram', { method: 'POST' });
        const data = await resp.json();
        if (statusEl) {
          if (data.status === "ok") {
            statusEl.textContent = "✅ " + (data.message || "RAM unloaded successfully");
            statusEl.style.color = "#8ea2be";
            if (typeof updateSystemResources === 'function') {
              updateSystemResources();
            }
          } else {
            statusEl.textContent = "❌ " + (data.message || "Failed to unload RAM");
            statusEl.style.color = "#ff8080";
          }
          setTimeout(() => {
            if (statusEl) {
              statusEl.style.display = "none";
              statusEl.textContent = "";
            }
          }, 5000);
        }
      } catch (err) {
        if (statusEl) {
          statusEl.textContent = "❌ Error: " + err.message;
          statusEl.style.color = "#ff8080";
        }
      } finally {
        if (icon) {
          icon.style.pointerEvents = "auto";
          icon.style.opacity = "0.6";
        }
      }
    };
    
    // Update system resources (RAM/VRAM) display
    async function updateSystemResources() {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
        const response = await fetch('/system_resources', { signal: controller.signal });
        clearTimeout(timeoutId);
        if (!response.ok) {
          throw new Error('HTTP ' + response.status);
        }
        const data = await response.json();
        
        const ramInfo = document.getElementById('ram-info');
        const vramInfo = document.getElementById('vram-info');
        
        if (ramInfo && data.ram) {
          if (data.ram.total_gb !== null) {
            const available = data.ram.available_gb !== null ? data.ram.available_gb : 0;
            const used = data.ram.used_gb !== null ? data.ram.used_gb : 0;
            const total = data.ram.total_gb;
            const percent = data.ram.percent !== null ? data.ram.percent : 0;
            const color = percent > 90 ? '#ff6666' : percent > 75 ? '#ffaa44' : '#88ff88';
            ramInfo.innerHTML = '<span style="color: ' + color + ';">' + available.toFixed(1) + ' GB free</span> / ' + total.toFixed(1) + ' GB total (' + percent.toFixed(1) + '% used)';
          } else {
            ramInfo.textContent = 'psutil not available';
            ramInfo.style.color = '#888';
          }
        }
        
        if (vramInfo && data.vram) {
          if (data.vram.total_gb !== null) {
            const available = data.vram.available_gb !== null ? data.vram.available_gb : 0;
            const used = data.vram.used_gb !== null ? data.vram.used_gb : 0;
            const total = data.vram.total_gb;
            const percent = data.vram.percent !== null ? data.vram.percent : 0;
            const color = percent > 90 ? '#ff6666' : percent > 75 ? '#ffaa44' : '#88ff88';
            vramInfo.innerHTML = '<span style="color: ' + color + ';">' + available.toFixed(1) + ' GB free</span> / ' + total.toFixed(1) + ' GB total (' + percent.toFixed(1) + '% used)';
          } else {
            vramInfo.textContent = 'NVIDIA GPU not detected';
            vramInfo.style.color = '#888';
          }
        }
      } catch (err) {
        // Silently fail for AbortError (timeout), log others
        if (err.name !== 'AbortError') {
        console.error('Failed to fetch system resources:', err);
        }
        const ramInfo = document.getElementById('ram-info');
        const vramInfo = document.getElementById('vram-info');
        if (ramInfo && ramInfo.textContent === 'Loading...') {
          ramInfo.textContent = 'Error loading';
          ramInfo.style.color = '#ff6666';
        }
        if (vramInfo && vramInfo.textContent === 'Loading...') {
          vramInfo.textContent = 'Error loading';
          vramInfo.style.color = '#ff6666';
        }
      }
    }

    function ensureOutputVisible() {
      const container = document.getElementById('output-container');
      if (container && container.style.display !== "block") {
        container.style.display = "block";
      }
    }
    
    function initLossChart() {
      const chartContainer = document.getElementById('loss-chart-container');
      const canvas = document.getElementById('loss-chart-canvas');
      if (!chartContainer || !canvas) return;
      
      // Destroy existing chart if it exists
      if (lossChart) {
        lossChart.destroy();
        lossChart = null;
      }
      
      const ctx = canvas.getContext('2d');
      lossChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [{
            label: 'Average Loss',
            data: [],
            borderColor: '#00ddcb',
            backgroundColor: 'rgba(0, 221, 203, 0.1)',
            borderWidth: 2,
            pointRadius: 2,
            pointHoverRadius: 4,
            fill: true,
            tension: 0.4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              labels: {
                color: '#eee'
              }
            },
            tooltip: {
              mode: 'index',
              intersect: false,
              backgroundColor: 'rgba(0, 0, 0, 0.8)',
              titleColor: '#86c6fe',
              bodyColor: '#eee',
              borderColor: '#00ddcb',
              borderWidth: 1
            }
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'Training Step',
                color: '#86c6fe'
              },
              ticks: {
                color: '#aaa'
              },
              grid: {
                color: 'rgba(255, 255, 255, 0.1)'
              }
            },
            y: {
              title: {
                display: true,
                text: 'Loss',
                color: '#86c6fe'
              },
              ticks: {
                color: '#aaa'
              },
              grid: {
                color: 'rgba(255, 255, 255, 0.1)'
              },
              beginAtZero: false
            }
          },
          interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
          }
        }
      });
      
      chartContainer.style.display = 'block';
    }
    
    function resetLossChart() {
      lossData = {
        steps: [],
        losses: []
      };
      currentStep = 0;
      
      if (lossChart) {
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.update('none');
      }
      
      const chartContainer = document.getElementById('loss-chart-container');
      if (chartContainer) {
        chartContainer.style.display = 'none';
      }
    }
    
    function parseAndUpdateLoss(data) {
      if (!data || typeof data !== 'string') return;
      
      // Parse loss from progress bar format: avr_loss=0.124
      // Also handle formats like: loss=0.123, avr_loss: 0.124, etc.
      const lossPatterns = [
        /avr_loss[=:]\s*([\d.]+)/i,
        /loss[=:]\s*([\d.]+)/i
      ];
      
      let lossValue = null;
      for (const pattern of lossPatterns) {
        const match = data.match(pattern);
        if (match && match[1]) {
          lossValue = parseFloat(match[1]);
          if (!isNaN(lossValue) && isFinite(lossValue)) {
            break;
          }
        }
      }
      
      if (lossValue === null || isNaN(lossValue) || !isFinite(lossValue)) {
        return;
      }
      
      // Extract step number if available (e.g., "steps: 5%|▌ | 3/60")
      const stepMatch = data.match(/steps:.*?\|.*?(\d+)\/(\d+)/);
      if (stepMatch && stepMatch[1]) {
        const stepNum = parseInt(stepMatch[1]);
        if (!isNaN(stepNum)) {
          currentStep = stepNum;
        }
      } else {
        // Increment step if not found in data
        currentStep++;
      }
      
      // Initialize chart if not already done
      if (!lossChart) {
        initLossChart();
      }
      
      // Add data point
      lossData.steps.push(currentStep);
      lossData.losses.push(lossValue);
      
      // Update chart
      if (lossChart) {
        lossChart.data.labels.push('Step ' + currentStep);
        lossChart.data.datasets[0].data.push(lossValue);
        
        // Limit to last 500 points for performance
        const maxPoints = 500;
        if (lossChart.data.labels.length > maxPoints) {
          lossChart.data.labels.shift();
          lossChart.data.datasets[0].data.shift();
        }
        
        // Update chart with animation
        lossChart.update('none'); // 'none' for no animation (faster updates)
      }
    }

    function setLogStatus(message, isError = false) {
      const statusEl = document.getElementById('log-resume-status');
      if (!statusEl) return;
      if (!message) {
        statusEl.style.display = "none";
        statusEl.textContent = "";
        return;
      }
      statusEl.style.display = "inline";
      statusEl.style.color = isError ? "#ff8080" : "#8ea2be";
      statusEl.textContent = message;
    }

    async function loadCurrentLog(showStatus = false) {
      try {
        const resp = await fetch('/current_log');
        if (!resp.ok) {
          throw new Error('HTTP ' + resp.status);
        }
        const data = await resp.json();
        if (typeof data.log_content === "string") {
          ensureOutputVisible();
          const outputText = document.getElementById("output-text");
          outputText.textContent = data.log_content;
          outputText.scrollTop = outputText.scrollHeight;
          
          // Show README link if training is finished and README exists
          if (data.log_content.includes("✅ TRAINING COMPLETED!") || data.log_content.includes("LoRA README generated")) {
            refreshReadmeLink();
          }
        }
        if (showStatus) {
          if (data.training_active) {
            setLogStatus("Training is running. Live stream available.", false);
          } else if (data.log_content) {
            setLogStatus("Training finished. Showing latest log.", false);
          } else {
            setLogStatus("No log data available yet.", true);
          }
        }
        return data;
      } catch (err) {
        console.error('Failed to load log file:', err);
        setLogStatus('Failed to load log: ' + err.message, true);
        return null;
      }
    }

    async function resumeLogStream() {
      const data = await loadCurrentLog(true);
      if (data && data.training_active) {
        setLogStatus("Live stream reconnected...", false);
        openLogStream();
      }
    }
    
    function hideReadmeLink() {
      const container = document.getElementById("readme-link-container");
      if (container) {
        container.style.display = "none";
      }
    }

    async function refreshReadmeLink() {
      const container = document.getElementById("readme-link-container");
      const link = document.getElementById("readme-link");
      if (!container || !link) return;
      try {
        const resp = await fetch('/latest_readme_link');
        if (!resp.ok) {
          throw new Error('HTTP ' + resp.status);
        }
        const data = await resp.json();
        if (data.available && data.output_dir) {
          link.href = '/lora_readme?output_dir=' + encodeURIComponent(data.output_dir);
          container.style.display = "block";
        } else {
          container.style.display = "none";
        }
      } catch (err) {
        console.error('Failed to refresh README link:', err);
      }
    }

    function showSpinner() { 
      document.getElementById("spinner").style.display = "block";
      const outputContainer = document.getElementById("output-container");
      outputContainer.style.display = "block";
      // Reset border color to green when starting a new training
      outputContainer.style.border = "3px solid #00ff00";
      outputContainer.style.borderRadius = "10px";
      document.getElementById("output-text").textContent = "";
      hideReadmeLink();
      
      // Reset loss chart for new training
      resetLossChart();
      const startBtn = document.getElementById("start-btn");
      const cancelBtn = document.getElementById("cancel-btn");
      const saveYamlBtn = document.getElementById("saveyaml-btn");
      const loadYamlBtn = document.getElementById("loadyaml-btn");
      const unloadVramIcon = document.getElementById("force-unload-vram-icon");
      const unloadRamIcon = document.getElementById("force-unload-ram-icon");
      if (startBtn) startBtn.disabled = true;
      if (cancelBtn) cancelBtn.disabled = false;
      if (saveYamlBtn) saveYamlBtn.disabled = true;
      if (loadYamlBtn) loadYamlBtn.disabled = true;
      if (unloadVramIcon) unloadVramIcon.style.pointerEvents = "none";
      if (unloadRamIcon) unloadRamIcon.style.pointerEvents = "none";
    }
    
    function enableAllButtons() {
      const startBtn = document.getElementById("start-btn");
      const cancelBtn = document.getElementById("cancel-btn");
      const saveYamlBtn = document.getElementById("saveyaml-btn");
      const loadYamlBtn = document.getElementById("loadyaml-btn");
      const unloadVramIcon = document.getElementById("force-unload-vram-icon");
      const unloadRamIcon = document.getElementById("force-unload-ram-icon");
      if (startBtn) startBtn.disabled = false;
      if (cancelBtn) cancelBtn.disabled = true;
      if (saveYamlBtn) saveYamlBtn.disabled = false;
      if (loadYamlBtn) loadYamlBtn.disabled = false;
      if (unloadVramIcon) unloadVramIcon.style.pointerEvents = "auto";
      if (unloadRamIcon) unloadRamIcon.style.pointerEvents = "auto";
    }
    
    function setUnloadStatus(message, isError = false) {
      const statusEl = document.getElementById('unload-status');
      if (!statusEl) return;
      if (!message) {
        statusEl.style.display = "none";
        statusEl.textContent = "";
        return;
      }
      statusEl.style.display = "inline";
      statusEl.style.color = isError ? "#ff8080" : "#8ea2be";
      statusEl.textContent = message;
      // Clear status after 5 seconds
      setTimeout(() => setUnloadStatus(""), 5000);
    }
    
    // Functions are already defined globally above
    async function forceUnloadVRAM() {
      if (window.forceUnloadVRAM) {
        return window.forceUnloadVRAM();
      }
    }
    
    async function forceUnloadRAM() {
      if (window.forceUnloadRAM) {
        return window.forceUnloadRAM();
      }
    }
    
    function updateEstimate() {
        const uploaded = parseInt(document.getElementById('uploaded_count').value || "0");
        const epochsInput = document.querySelector('input[name="epochs"]');
        const batchInput = document.querySelector('input[name="batch_size"]');
        const repeatsInput = document.querySelector('input[name="image_repeats"]');
        const vramSelect = document.querySelector('select[name="vram_profile"]');
        const resInput = document.querySelector('input[name="resolution"]');
        const estElem = document.getElementById('time-estimate');
        const warningElem = document.getElementById('few-images-warning');
        if (!epochsInput || !batchInput || !vramSelect || !resInput || !estElem) return;
        
        const epochs = parseInt(epochsInput.value || "0");
        const batchSize = parseInt(batchInput.value || "1");
        const repeats = parseInt(repeatsInput ? (repeatsInput.value || "1") : "1");
        const vramProfile = vramSelect.value || "16";
        const resText = resInput.value || "1024x1024";
        
        if (!uploaded || !epochs || !batchSize) {
            estElem.textContent = "Estimated training time: add images and set epochs/batch size.";
            if (warningElem) warningElem.style.display = "none";
            return;
        }
        
        const effectiveImages = uploaded * (repeats > 0 ? repeats : 1);
        let stepsPerEpoch = Math.ceil(effectiveImages / batchSize);
        let totalSteps = stepsPerEpoch * epochs;
        
        // Show warning if too few images or too few steps for person training
        if (warningElem) {
            if (uploaded < 8 && totalSteps < 300) {
                warningElem.style.display = "block";
            } else {
                warningElem.style.display = "none";
            }
        }
        
        // Parse resolution to scale estimate by pixel count
        let w = 1024, h = 1024;
        const m = resText.match(/(\d+)\s*x\s*(\d+)/i);
        if (m) {
            w = parseInt(m[1] || "1024");
            h = parseInt(m[2] || "1024");
        }
        const resFactor = (w * h) / (1024 * 1024); // 1.0 for 1024x1024
        
        // Rough base seconds per step depending on VRAM profile and resolution
        let baseSec = 4.0; // default for 16 GB
        if (vramProfile === "12") baseSec = 6.0; // slower for heavy offload
        if (vramProfile === "24") baseSec = 3.0; // faster for high-VRAM cards
        baseSec *= resFactor;
        
        // Add constant overhead for model loading + caching
        const overheadSec = 60;
        const totalSec = overheadSec + totalSteps * baseSec;
        let text;
        if (totalSec < 60) {
            text = 'Estimated training time: ~' + totalSec.toFixed(0) + ' seconds (' + totalSteps + ' steps).';
        } else if (totalSec < 3600) {
            const min = Math.round(totalSec / 60);
            text = 'Estimated training time: ~' + min + ' minutes (' + totalSteps + ' steps).';
        } else {
            const hours = (totalSec / 3600).toFixed(1);
            text = 'Estimated training time: ~' + hours + ' hours (' + totalSteps + ' steps).';
        }
        estElem.textContent = text;
    }
    // Store currently selected files globally to preserve them when adding more
    var storedFiles = [];
    
    function setupNavigation() {
      const navLinks = document.querySelectorAll('.nav-links a');
      const sections = {
        'model-files': document.getElementById('model-files'),
        'upload-images': document.getElementById('upload-images'),
        'training-settings': document.getElementById('training-settings'),
        'preview': document.getElementById('preview'),
        'system-resources': document.getElementById('system-resources'),
        'training-output': document.getElementById('training-output')
      };
      
      // Smooth scroll on click
      navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
          const href = this.getAttribute('href');
          if (href.startsWith('#')) {
            const targetId = href.substring(1);
            const targetElement = sections[targetId] || document.getElementById(targetId);
            if (targetElement) {
              e.preventDefault();
              const navHeight = document.querySelector('.nav-links').offsetHeight;
              const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - navHeight - 20;
              window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
              });
            }
          }
        });
      });
      
      // Update active state based on scroll position
      // Use a simpler approach: find the section closest to the top of the viewport
      function updateActiveNav() {
        const navHeight = document.querySelector('.nav-links').offsetHeight;
        const scrollOffset = window.pageYOffset + navHeight + 50; // Offset for sticky nav
        
        let activeId = null;
        let minDistance = Infinity;
        
        // Check each main section
        const mainSectionIds = ['model-files', 'upload-images', 'training-settings', 'system-resources', 'training-output'];
        
        mainSectionIds.forEach(sectionId => {
          const section = sections[sectionId];
          if (section) {
            const rect = section.getBoundingClientRect();
            const sectionTop = window.pageYOffset + rect.top;
            const distance = Math.abs(sectionTop - scrollOffset);
            
            // If section is visible and closer to the top than previous best
            if (rect.top < window.innerHeight && rect.bottom > 0 && distance < minDistance) {
              minDistance = distance;
              activeId = sectionId;
            }
          }
        });
        
        // Special handling for preview - only if upload-images is not active
        if (activeId !== 'upload-images') {
          const previewElement = sections['preview'];
          if (previewElement) {
            const rect = previewElement.getBoundingClientRect();
            if (rect.top < window.innerHeight && rect.bottom > 0) {
              const previewTop = window.pageYOffset + rect.top;
              const previewDistance = Math.abs(previewTop - scrollOffset);
              // Only use preview if it's closer and upload-images is not the active one
              if (previewDistance < minDistance) {
                activeId = 'preview';
              }
            }
          }
        }
        
        // Update active state
        navLinks.forEach(link => {
          const href = link.getAttribute('href');
          if (href === '#' + activeId) {
            link.classList.add('active');
          } else {
            link.classList.remove('active');
          }
        });
      }
      
      // Update on scroll
      let scrollTimeout;
      window.addEventListener('scroll', function() {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(updateActiveNav, 50);
      }, { passive: true });
      
      // Initial update
      updateActiveNav();
    }
    
    function saveDraftToStorage() {
      try {
        const draft = {};
        
        // Save all form fields
        const fieldsToSave = [
          'epochs', 'batch_size', 'image_repeats', 'lr', 'resolution', 'seed',
          'rank', 'dims', 'output_dir', 'output_name', 'trigger', 'vram_profile',
          'caption_model', 'caption_max_length', 'caption_detail_level',
          'training_preset', 'sample_every_epochs', 'sample_every_steps',
          'sample_prompt_text', 'advanced_flags'
        ];
        
        fieldsToSave.forEach(fieldName => {
          const input = document.querySelector(`input[name="${fieldName}"], select[name="${fieldName}"], textarea[name="${fieldName}"]`);
          if (input) {
            if (input.type === 'checkbox') {
              draft[fieldName] = input.checked;
            } else {
              draft[fieldName] = input.value;
            }
          }
        });
        
        // Save samples_enabled checkbox separately
        const samplesCheckbox = document.getElementById('samples_enabled');
        if (samplesCheckbox) {
          draft.samples_enabled = samplesCheckbox.checked;
        }
        
        // Save to localStorage
        localStorage.setItem('musubitlx_training_draft', JSON.stringify(draft));
        localStorage.setItem('musubitlx_draft_timestamp', new Date().toISOString());
      } catch (err) {
        console.warn('Failed to save draft to localStorage:', err);
      }
    }
    
    function loadDraftFromStorage() {
      try {
        const draftJson = localStorage.getItem('musubitlx_training_draft');
        if (!draftJson) return;
        
        const draft = JSON.parse(draftJson);
        const timestamp = localStorage.getItem('musubitlx_draft_timestamp');
        
        // Check if draft is recent (within last 30 days)
        if (timestamp) {
          const draftDate = new Date(timestamp);
          const now = new Date();
          const daysDiff = (now - draftDate) / (1000 * 60 * 60 * 24);
          if (daysDiff > 30) {
            // Draft is too old, don't load it
            localStorage.removeItem('musubitlx_training_draft');
            localStorage.removeItem('musubitlx_draft_timestamp');
            return;
          }
        }
        
        // Load all saved fields
        Object.keys(draft).forEach(fieldName => {
          const input = document.querySelector(`input[name="${fieldName}"], select[name="${fieldName}"], textarea[name="${fieldName}"]`);
          if (input) {
            if (input.type === 'checkbox') {
              input.checked = draft[fieldName] === true;
            } else {
              input.value = draft[fieldName] || '';
            }
            // Trigger input event to update dependent fields
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
          }
        });
        
        // Load samples_enabled checkbox separately
        if (draft.samples_enabled !== undefined) {
          const samplesCheckbox = document.getElementById('samples_enabled');
          if (samplesCheckbox) {
            samplesCheckbox.checked = draft.samples_enabled === true;
            samplesCheckbox.dispatchEvent(new Event('change', { bubbles: true }));
          }
        }
        
        // Update dependent UI elements
        setTimeout(function() {
          if (typeof updateEstimate === 'function') {
            updateEstimate();
          }
          if (typeof updateCommandPreview === 'function') {
            updateCommandPreview();
          }
          if (typeof updatePresetDropdown === 'function') {
            updatePresetDropdown();
          }
          if (typeof updateCaptionLengthLabel === 'function') {
            updateCaptionLengthLabel();
          }
          if (typeof updateDetailLevelLabel === 'function') {
            updateDetailLevelLabel();
          }
        }, 100);
      } catch (err) {
        console.warn('Failed to load draft from localStorage:', err);
      }
    }
    
    function setupAutoSave() {
      // Save draft when any form field changes
      const form = document.getElementById('training-form');
      if (!form) return;
      
      // Debounce save to avoid too frequent saves
      let saveTimeout;
      function debouncedSave() {
        clearTimeout(saveTimeout);
        saveTimeout = setTimeout(saveDraftToStorage, 500); // Save 500ms after last change
      }
      
      // Listen to all input, change, and select events
      form.addEventListener('input', debouncedSave);
      form.addEventListener('change', debouncedSave);
      
      // Also listen to checkbox changes
      const samplesCheckbox = document.getElementById('samples_enabled');
      if (samplesCheckbox) {
        samplesCheckbox.addEventListener('change', debouncedSave);
      }
      
      // Save on page unload as well
      window.addEventListener('beforeunload', function() {
        saveDraftToStorage();
      });
    }
    
    function updateStartButtonState() {
        const startBtn = document.getElementById('start-btn');
        if (!startBtn) return;
        
        const uploadedCount = document.getElementById('uploaded_count');
        const imageCount = uploadedCount ? parseInt(uploadedCount.value || "0") : 0;
        
        if (imageCount === 0) {
            startBtn.disabled = true;
            startBtn.style.opacity = "0.5";
            startBtn.style.cursor = "not-allowed";
            startBtn.title = "Please select at least one image before starting training";
        } else {
            startBtn.disabled = false;
            startBtn.style.opacity = "1";
            startBtn.style.cursor = "pointer";
            startBtn.title = "";
        }
    }
    
    function handleFileInputChange(event) {
        const fileInput = event.target;
        if (!fileInput) return;
        
        // Get new files that were just selected (browser has already updated fileInput.files)
        const newFiles = Array.from(fileInput.files || []);
        
        // Get existing files from our stored list
        const existingFiles = storedFiles.slice();
        
        // Combine existing and new files (avoid duplicates)
        const allFiles = existingFiles.slice();
        const existingKeys = new Set(existingFiles.map(f => f.name + f.size + f.lastModified));
        
        for (let i = 0; i < newFiles.length; i++) {
            const newFile = newFiles[i];
            const fileKey = newFile.name + newFile.size + newFile.lastModified;
            if (!existingKeys.has(fileKey)) {
                allFiles.push(newFile);
                existingKeys.add(fileKey);
            }
        }
        
        // Update stored files
        storedFiles = allFiles.slice();
        
        // Update file input with combined files
        const dt = new DataTransfer();
        for (let i = 0; i < allFiles.length; i++) {
            dt.items.add(allFiles[i]);
        }
        fileInput.files = dt.files;
        
        // Call previewImages with the combined files
        previewImages({ target: fileInput, files: allFiles });
    }
    
    function previewImages(event) {
        var preview = document.getElementById('preview');
        const fileInput = document.getElementById('images');
        if (!fileInput) return;
        
        // Get files from event.files (if provided) or from fileInput.files
        // When called from handleFileInputChange, event.files contains the combined list
        var allFiles = event.files ? event.files : (event.target && event.target.files ? Array.from(event.target.files) : []);
        
        // If we got files from event, update storedFiles to match
        if (event.files) {
            storedFiles = Array.from(allFiles);
        } else {
            // Otherwise, use storedFiles (for drag & drop or other cases)
            allFiles = storedFiles.slice();
        }
        
        // Clear preview and rebuild with all files
        preview.innerHTML = '';
        preview.classList.remove('imglist-empty');
        
        const totalFiles = allFiles.length;
        document.getElementById('uploaded_count').value = totalFiles;
        
        // Update start button state based on image count
        updateStartButtonState();
        
        // Update preset values if a preset is active
        setTimeout(function() {
          if (typeof updatePresetValuesForImageCount === 'function') {
            updatePresetValuesForImageCount();
          }
        }, 50);
        
        // Update file input status text and clear button visibility
        const statusEl = document.getElementById('file-input-status');
        const clearBtn = document.getElementById('clear-images-btn');
        if (statusEl) {
            if (totalFiles === 0) {
                statusEl.textContent = "No files selected";
                if (clearBtn) clearBtn.style.display = "none";
                preview.classList.add('imglist-empty');
                preview.innerHTML = '<div class="drop-hint">Drag &amp; drop images here or use the file picker above.</div>';
            } else if (totalFiles === 1) {
                statusEl.textContent = "1 file selected";
                if (clearBtn) clearBtn.style.display = "inline-block";
            } else {
                statusEl.textContent = totalFiles + " files selected";
                if (clearBtn) clearBtn.style.display = "inline-block";
            }
        }
        
        // Render all files
        for (let i = 0; i < allFiles.length; i++) {
            let f = allFiles[i];
            let reader = new FileReader();

            // Create container and caption in the correct order immediately
            var div = document.createElement('div');
            div.className = "imgbox";
            div.setAttribute('data-file-index', i);
            div.innerHTML = '<button type="button" class="imgbox-remove" onclick="window.removeImage && window.removeImage(' + i + ')" title="Remove this image">×</button>' +
                '<img><br>' +
                '<textarea class="caption-textarea" name="caption_' + i + '" rows="2" placeholder="Caption for image ' + (i + 1) + '"></textarea>';
            let imgEl = div.querySelector('img');
            preview.appendChild(div);

            // When the file is loaded, just set the image source
            reader.onload = function(e) {
                if (imgEl) {
                    imgEl.src = e.target.result;
                    // Add click handler for zoom
                    imgEl.addEventListener('click', function() {
                        openImageZoom(e.target.result);
                    });
                }
            }
            reader.readAsDataURL(f);
        }
        updateEstimate();
    }

    function openImageZoom(imageSrc) {
        const modal = document.getElementById('image-zoom-modal');
        const img = document.getElementById('zoomed-image');
        if (modal && img) {
            img.src = imageSrc;
            modal.classList.add('active');
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
            
            // Close on Escape key
            const escapeHandler = function(e) {
                if (e.key === 'Escape') {
                    closeImageZoom();
                    document.removeEventListener('keydown', escapeHandler);
                }
            };
            document.addEventListener('keydown', escapeHandler);
        }
    }
    
    function closeImageZoom() {
        const modal = document.getElementById('image-zoom-modal');
        if (modal) {
            modal.classList.remove('active');
            document.body.style.overflow = ''; // Restore scrolling
        }
    }

    function removeImage(indexToRemove) {
        const fileInput = document.getElementById('images');
        if (!fileInput || storedFiles.length === 0) return;
        
        // Remove file from stored files
        storedFiles.splice(indexToRemove, 1);
        
        // Create a new FileList without the removed file
        const dt = new DataTransfer();
        for (let i = 0; i < storedFiles.length; i++) {
            dt.items.add(storedFiles[i]);
        }
        
        // Update file input with new FileList
        fileInput.files = dt.files;
        
        // Re-render preview with updated files
        previewImages({ target: fileInput, files: storedFiles.slice() });
        
        // Update start button state
        updateStartButtonState();
        
        // Note: updatePresetValuesForImageCount will be called from previewImages
    }

    function clearSelectedImages() {
        const fileInput = document.getElementById('images');
        const preview = document.getElementById('preview');
        const statusEl = document.getElementById('file-input-status');
        const clearBtn = document.getElementById('clear-images-btn');
        const uploadedCount = document.getElementById('uploaded_count');
        
        // Clear stored files
        storedFiles = [];
        
        if (fileInput) {
            // Clear the file input by creating a new one (old value can't be cleared directly)
            fileInput.value = '';
        }
        
        if (preview) {
            preview.innerHTML = '<div class="drop-hint">Drag &amp; drop images here or use the file picker above.</div>';
            preview.classList.remove('imglist-empty');
            preview.classList.add('imglist-empty');
        }
        
        if (statusEl) {
            statusEl.textContent = "No files selected";
        }
        
        if (clearBtn) {
            clearBtn.style.display = "none";
        }
        
        if (uploadedCount) {
            uploadedCount.value = "0";
        }
        
        // Update start button state
        updateStartButtonState();
        
        // Update preset values if a preset is active
        setTimeout(function() {
          if (typeof updatePresetValuesForImageCount === 'function') {
            updatePresetValuesForImageCount();
          }
        }, 50);
        
        updateEstimate();
    }

    let autoCapTimer = null;
    let autoCapStart = null;

    let advFlagsDirty = false;

    function updateCaptionLengthLabel() {
      const slider = document.getElementById('caption_max_length');
      const label = document.getElementById('caption_length_label');
      if (!slider || !label) return;
      label.textContent = slider.value;
    }

    function updateDetailLevelLabel() {
      const slider = document.getElementById('caption_detail_level');
      const label = document.getElementById('detail_level_label');
      const desc = document.getElementById('detail_level_desc');
      if (!slider || !label || !desc) return;
      const level = parseInt(slider.value);
      label.textContent = level;
      const descriptions = {
        1: 'Basic',
        2: 'Standard',
        3: 'Detailed',
        4: 'Very Detailed',
        5: 'Extremely Detailed'
      };
      desc.textContent = descriptions[level] || 'Standard';
    }

    function computeFinalOutputDir() {
      const baseOutput = "output";
      const outputDirInput = document.querySelector('input[name="output_dir"]');
      let userOutputDir = outputDirInput ? (outputDirInput.value || "") : "";
      userOutputDir = userOutputDir.trim().replace(/\\/g, "/");
      userOutputDir = userOutputDir.replace(/^\/+/, "").replace(/\s+$/, "");
      if (!userOutputDir) {
        return baseOutput;
      }
      return (baseOutput + '/' + userOutputDir).replace(/\/+/g, "/");
    }

    function markSampleGalleryPending(message) {
      const gallery = document.getElementById('sample-gallery');
      if (!gallery) return;
      gallery.innerHTML = '<span>' + (message || "Output folder changed. Click refresh to load samples.") + '</span>';
      gallery.classList.add('sample-gallery-empty');
    }

    function updateFinalPath() {
      const outputDirInput = document.querySelector('input[name="output_dir"]');
      const outputNameInput = document.querySelector('input[name="output_name"]');
      const finalPathPreview = document.getElementById('final-path-preview');
      
      if (!outputDirInput || !outputNameInput || !finalPathPreview) return;
      
      const outputName = (outputNameInput.value || "lora").trim();
      const finalOutDir = computeFinalOutputDir();
      const finalPath = (finalOutDir + '/' + outputName + '.safetensors').replace(/\/+/g, "/");
      
      finalPathPreview.textContent = finalPath;
      markSampleGalleryPending("Output folder changed. Click refresh to load samples.");
    }

    function setSampleGalleryStatus(message, isError) {
      const statusEl = document.getElementById('sample-gallery-status');
      if (!statusEl) return;
      if (!message) {
        statusEl.style.display = "none";
        statusEl.textContent = "";
        return;
      }
      statusEl.style.display = "block";
      statusEl.style.color = isError ? "#ff8080" : "#b3daff";
      statusEl.textContent = message;
    }

    function renderSampleGallery(files) {
      const gallery = document.getElementById('sample-gallery');
      if (!gallery) return;
      gallery.innerHTML = "";
      if (!files || files.length === 0) {
        gallery.classList.add('sample-gallery-empty');
        gallery.innerHTML = "<span>No sample previews found for the current output folder.</span>";
        return;
      }
      gallery.classList.remove('sample-gallery-empty');
      files.forEach(file => {
        const card = document.createElement('div');
        card.className = "sample-card";
        const media =
          file.kind === "video"
            ? '<video src="' + file.url + '" muted loop playsinline></video>'
            : '<img src="' + file.url + '" alt="' + file.name + ' thumbnail" class="sample-image-zoom">';
        const timestamp = file.mtime_display || "";
        card.innerHTML = 
          '<a href="' + file.url + '" target="_blank" rel="noopener">' +
            media +
            '<div class="sample-card-meta">' + file.name + '</div>' +
            '<div class="sample-card-meta" style="color:#888;">' + timestamp + '</div>' +
          '</a>';
        gallery.appendChild(card);
        
        // Add click handler for zoom on images (not videos)
        if (file.kind !== "video") {
          const img = card.querySelector('img.sample-image-zoom');
          if (img) {
            img.addEventListener('click', function(e) {
              e.preventDefault();
              e.stopPropagation();
              openImageZoom(file.url);
            });
          }
        }
      });
    }

    async function refreshSampleGallery(auto=false) {
      const finalOutDir = computeFinalOutputDir();
      const gallery = document.getElementById('sample-gallery');
      if (!gallery) return;
      if (!finalOutDir) {
        renderSampleGallery([]);
        setSampleGalleryStatus("No output folder selected.", true);
        return;
      }
      setSampleGalleryStatus("Loading sample previews...", false);
      try {
        const resp = await fetch('/list_samples?output_dir=' + encodeURIComponent(finalOutDir));
        if (!resp.ok) {
          throw new Error('Server responded with ' + resp.status);
        }
        const data = await resp.json();
        renderSampleGallery(data.files || []);
        if (data.files && data.files.length) {
          setSampleGalleryStatus('Showing ' + data.files.length + ' sample file' + (data.files.length > 1 ? 's' : '') + '.', false);
        } else if (auto) {
          setSampleGalleryStatus("No samples found yet for this run.", false);
        } else {
          setSampleGalleryStatus("No sample previews found. Run training with samples enabled, then refresh.", false);
        }
      } catch (err) {
        console.error("Failed to load sample gallery:", err);
        setSampleGalleryStatus('Failed to load samples: ' + err.message, true);
        renderSampleGallery([]);
      }
    }

    function buildDefaultExtraFlags(vramProfile) {
      // Map user's GPU VRAM to actual training settings
      // 12GB VRAM GPU: Use blocks_to_swap 45 → ~12GB VRAM usage (safest, most offload to RAM)
      // Note: gradient_checkpointing_cpu_offload is removed to avoid CUDA alignment issues with fp8_scaled
      if (vramProfile === "12") {
        return "--fp8_base --fp8_scaled --blocks_to_swap 45 --gradient_checkpointing --max_grad_norm 1.0";
      }
      // 16GB VRAM GPU: Use blocks_to_swap 30 → ~16-18GB VRAM usage (balanced, faster than 12GB profile)
      // blocks_to_swap 16 would need 24GB VRAM which doesn't fit on 16GB cards
      // blocks_to_swap 30 is a middle ground that should give ~16-18GB VRAM usage
      if (vramProfile === "16") {
        return "--fp8_base --fp8_scaled --blocks_to_swap 30 --gradient_checkpointing --max_grad_norm 1.0";
      }
      // 24GB+ VRAM GPU: Can use blocks_to_swap 16 → ~24GB VRAM usage (fastest, minimal offload)
      if (vramProfile === "24") {
        return "--fp8_base --fp8_scaled --blocks_to_swap 16 --gradient_checkpointing --max_grad_norm 1.0";
      }
      return "";
    }

    function syncAdvancedFlagsFromGui(initial) {
      const advFlagsInput = document.querySelector('textarea[name="advanced_flags"]');
      const samplesCheckbox = document.getElementById('samples_enabled');
      const samplePromptTextarea = document.getElementById('sample_prompt_text');
      const sampleEveryEpochsInput = document.querySelector('input[name="sample_every_epochs"]');
      const sampleEveryStepsInput = document.querySelector('input[name="sample_every_steps"]');
      const sampleAtFirstCheckbox = document.getElementById('sample_at_first');
      const vramSelect = document.querySelector('select[name="vram_profile"]');
      if (!advFlagsInput || !vramSelect) return;

      // Do not override user edits once they have started typing (unless this is initial fill and field is empty)
      if (!initial && advFlagsDirty) {
        return;
      }

      const vramProfile = vramSelect.value || "12";
      const defaults = buildDefaultExtraFlags(vramProfile);

      if (initial) {
        // Only fill if empty so existing YAML-loaded value is preserved
        if (!advFlagsInput.value || advFlagsInput.value.trim() === "") {
          advFlagsInput.value = defaults;
        }
      } else {
        advFlagsInput.value = defaults;
      }
      updateCommandPreview();
    }

    function updateCommandPreview() {
      const preview = document.getElementById('cmd-preview');
      if (!preview) return;

      const vramSelect = document.querySelector('select[name="vram_profile"]');
      const epochsInput = document.querySelector('input[name="epochs"]');
      const batchInput = document.querySelector('input[name="batch_size"]');
      const lrInput = document.querySelector('input[name="lr"]');
      const rankInput = document.querySelector('input[name="rank"]');
      const dimsInput = document.querySelector('input[name="dims"]');
      const seedInput = document.querySelector('input[name="seed"]');
      const resInput = document.querySelector('input[name="resolution"]');
      const outDirInput = document.querySelector('input[name="output_dir"]');
      const outNameInput = document.querySelector('input[name="output_name"]');
      const optSelect = document.querySelector('select[name="optimizer_type"]');
      const advFlagsInput = document.querySelector('textarea[name="advanced_flags"]');
      const samplesCheckbox = document.getElementById('samples_enabled');
      const samplePromptTextarea = document.getElementById('sample_prompt_text');
      const sampleEveryEpochsInput = document.querySelector('input[name="sample_every_epochs"]');
      const sampleEveryStepsInput = document.querySelector('input[name="sample_every_steps"]');
      const sampleAtFirstCheckbox = document.getElementById('sample_at_first');

      const vramProfile = vramSelect ? (vramSelect.value || "12") : "12";
      const epochs = epochsInput ? (epochsInput.value || "6") : "6";
      const batchSize = batchInput ? (batchInput.value || "1") : "1";
      const lr = lrInput ? (lrInput.value || "5e-5") : "5e-5";
      const rank = rankInput ? (rankInput.value || "16") : "16";
      const dims = dimsInput ? (dimsInput.value || "128") : "128";
      const seed = seedInput ? (seedInput.value || "42") : "42";
      const resolution = resInput ? (resInput.value || "1024x1024") : "1024x1024";
      const outputName = outNameInput ? (outNameInput.value || "lora") : "lora";
      const optimizer = optSelect ? (optSelect.value || "AdamW") : "AdamW";
      const advFlagsText = advFlagsInput ? (advFlagsInput.value || "") : "";
      const samplesEnabled = samplesCheckbox ? samplesCheckbox.checked : false;
      const samplePromptText = samplePromptTextarea ? (samplePromptTextarea.value || "") : "";
      const sampleEveryEpochs = sampleEveryEpochsInput ? parseInt(sampleEveryEpochsInput.value || "0") : 0;
      const sampleEverySteps = sampleEveryStepsInput ? parseInt(sampleEveryStepsInput.value || "0") : 0;
      const sampleAtFirst = sampleAtFirstCheckbox ? sampleAtFirstCheckbox.checked : false;

      const finalOutDir = computeFinalOutputDir();
      const datasetConfig = finalOutDir + "/dataset_config.toml";

      const ditModel = "qwen_image_bf16.safetensors";
      const vaeModel = "diffusion_pytorch_model.safetensors";
      const textEncoder = "qwen_2.5_vl_7b.safetensors";

      let cmdParts = [
        "python", "src/musubi_tuner/qwen_image_train_network.py",
        "--dit", ditModel,
        "--vae", vaeModel,
        "--text_encoder", textEncoder,
        "--dataset_config", datasetConfig,
        "--max_train_epochs", String(epochs),
        "--save_every_n_epochs", "1",
        "--learning_rate", String(lr),
        "--network_dim", String(rank),
        "--network_alpha", String(dims),
        "--seed", String(seed),
        "--output_dir", finalOutDir,
        "--output_name", outputName,
        "--network_module", "networks.lora_qwen_image",
        "--optimizer_type", optimizer,
        "--mixed_precision", "bf16",
        "--sdpa",
        "--timestep_sampling", "qwen_shift",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers"
      ];

      if (advFlagsText.trim() !== "") {
        const extra = advFlagsText.trim().split(/\s+/);
        cmdParts = cmdParts.concat(extra);
      }

      if (samplesEnabled && samplePromptText.trim() !== "") {
        const samplePromptPath = finalOutDir + "/sample_prompts.txt";
        cmdParts.push("--sample_prompts", samplePromptPath);
        if (!isNaN(sampleEveryEpochs) && sampleEveryEpochs > 0) {
          cmdParts.push("--sample_every_n_epochs", String(sampleEveryEpochs));
        }
        if (!isNaN(sampleEverySteps) && sampleEverySteps > 0) {
          cmdParts.push("--sample_every_n_steps", String(sampleEverySteps));
        }
        if (sampleAtFirst) {
          cmdParts.push("--sample_at_first");
        }
      }

      const cmdString = cmdParts.map(p => (/\s/.test(p) ? '"' + p + '"' : p)).join(" ");
      if (preview) {
        preview.textContent = cmdString;
      }
    }

    async function autoCaptionCaptionModel() {
      const fileInput = document.getElementById('images');
      const files = fileInput && fileInput.files ? fileInput.files : null;
      if (!files || !files.length) {
        alert("Please select or drop images first before auto-captioning.");
        return;
      }

      const btn = document.getElementById('autocap-btn');
      const statusEl = document.getElementById('autocap-status');
      const progressContainer = document.getElementById('autocap-progress-container');
      const progressBar = document.getElementById('autocap-progress-bar');
      const modelSelect = document.getElementById('caption_model');
      const selectedModelValue = modelSelect ? (modelSelect.value || 'vit-gpt2') : 'vit-gpt2';
      const runningModelName = selectedModelValue === 'qwen-vl' ? 'Qwen-VL' : 
                               selectedModelValue === 'blip-large' ? 'BLIP large' : 'ViT-GPT2';
      
      // Check if model is already loaded and if it's cached
      let modelLoaded = false;
      let modelCached = false;
      try {
        const statusResp = await fetch('/caption_model_status?model=' + encodeURIComponent(selectedModelValue));
        if (statusResp.ok) {
          const statusData = await statusResp.json();
          modelLoaded = statusData.model_loaded || false;
          modelCached = statusData.model_cached || false;
        }
      } catch (err) {
        // Silently fail - assume model not loaded
      }
      
      if (btn) {
        btn.disabled = true;
        btn.textContent = "Auto-captioning with " + runningModelName + "...";
      }

      const totalImages = files.length;
      let currentImage = 0;
      
      // Estimated time per image (in seconds) based on model type
      const timePerImage = {
        'vit-gpt2': 2,
        'blip-large': 5,
        'qwen-vl': 10
      };
      const estimatedTimePerImage = timePerImage[selectedModelValue] || 3;
      
      // Show progress bar
      if (progressContainer) {
        progressContainer.style.display = "block";
      }
      if (progressBar) {
        progressBar.style.width = "0%";
      }

        if (statusEl) {
        autoCapStart = Date.now();
        statusEl.style.display = "block";
        
        // Different messages based on whether model is loaded and cached
        let initialStatusText;
        if (!modelLoaded) {
          if (modelCached) {
            // Model is cached on disk, just loading into memory
            initialStatusText = "🔄 Loading " + runningModelName + " model from cache into memory...";
          } else {
            // Model not cached, will download
            if (selectedModelValue === 'qwen-vl') {
              initialStatusText = "📥 " + runningModelName + " model not found. Downloading from Hugging Face (~14GB, this may take 10-15 minutes)...";
            } else {
              initialStatusText = "📥 " + runningModelName + " model not found. Downloading from Hugging Face (first run only, this may take several minutes)...";
            }
          }
        } else {
          initialStatusText = "🔄 Loading " + runningModelName + " model into memory...";
        }
        statusEl.textContent = initialStatusText;

        if (autoCapTimer) {
          clearInterval(autoCapTimer);
        }
        autoCapTimer = setInterval(() => {
          if (!autoCapStart) return;
          const elapsedSec = Math.round((Date.now() - autoCapStart) / 1000);
          
          // Estimate progress based on elapsed time
          // For first run: downloading can take 1-15 minutes depending on model size
          // ViT-GPT2: ~500MB, BLIP: ~1.5GB, Qwen-VL: ~14GB
          const downloadTime = {
            'vit-gpt2': 60,      // ~1 minute for ~500MB
            'blip-large': 120,   // ~2 minutes for ~1.5GB
            'qwen-vl': 600       // ~10 minutes for ~14GB (may take 10-15 minutes depending on connection)
          };
          // If model is loaded, it's quick. If cached but not loaded, loading from cache is faster than downloading.
          const modelLoadTime = modelLoaded ? 5 : (modelCached ? 10 : (downloadTime[selectedModelValue] || 60));
          const estimatedTotalTime = modelLoadTime + (totalImages * estimatedTimePerImage);
          
          if (elapsedSec < modelLoadTime) {
            // Still loading/downloading model
            const progressPercent = Math.min(10, (elapsedSec / modelLoadTime) * 10);
            if (progressBar) progressBar.style.width = progressPercent + "%";
            
            if (!modelLoaded) {
              if (modelCached) {
                // Model is cached, just loading from cache into memory
                statusEl.textContent = "🔄 Loading " + runningModelName + " model from cache into memory...";
              } else {
                // Model is being downloaded
                if (selectedModelValue === 'qwen-vl') {
                  // Qwen-VL is a large file (~14GB), estimate download time
                  const qwenVlDownloadTime = 600; // ~10 minutes for ~14GB at reasonable speeds
                  const progressPercent = Math.min(10, (elapsedSec / qwenVlDownloadTime) * 10);
                  if (progressBar) progressBar.style.width = progressPercent + "%";
                  
                  statusEl.textContent = "📥 Downloading " + runningModelName + " model (~14GB) from Hugging Face... This may take 10-15 minutes depending on your connection speed.";
                } else {
                  statusEl.textContent = "📥 Downloading " + runningModelName + " model (" + (selectedModelValue === 'vit-gpt2' ? '~500MB' : '~1.5GB') + ") from Hugging Face... This only happens on first run.";
                }
              }
            } else {
              // Model already loaded in memory
              statusEl.textContent = "🔄 Loading " + runningModelName + " model into memory...";
            }
          } else {
            // Processing images - show clear progress
            const processingTime = elapsedSec - modelLoadTime;
            currentImage = Math.min(totalImages - 1, Math.floor(processingTime / estimatedTimePerImage));
            const progress = Math.min(95, 10 + ((currentImage + 1) / totalImages) * 85);
            
            if (progressBar) progressBar.style.width = progress + "%";
            statusEl.textContent = "✨ Writing caption for image " + (currentImage + 1) + " of " + totalImages + " with " + runningModelName + "...";
          }
        }, 500); // Update every 500ms for smoother progress
      }

      try {
        const fd = new FormData();
        for (let i = 0; i < files.length; i++) {
          fd.append('images', files[i], files[i].name);
        }
        const modelSelect2 = document.getElementById('caption_model');
        if (modelSelect2) {
          fd.append('caption_model', modelSelect2.value || 'vit-gpt2');
        }
        const lengthSlider = document.getElementById('caption_max_length');
        if (lengthSlider) {
          fd.append('caption_max_length', lengthSlider.value || '128');
        }
        const detailLevelSlider = document.getElementById('caption_detail_level');
        if (detailLevelSlider) {
          fd.append('caption_detail_level', detailLevelSlider.value || '3');
        }

        const resp = await fetch('/autocaption', {
          method: 'POST',
          body: fd
        });

        let data = null;
        try {
          data = await resp.json();
        } catch (e) {
          throw new Error("Server returned a non-JSON response.");
        }

        if (!resp.ok) {
          const msg = data && data.error ? data.error : resp.statusText;
          if (statusEl) statusEl.textContent = "Autocaption error: " + msg;
          return;
        }

        if (data.missing_dependencies) {
          const msg = data.error || "Captioning dependencies are missing. Please install them in your environment.";
          if (statusEl) statusEl.textContent = msg;
          return;
        }

        if (!data.captions || !Array.isArray(data.captions)) {
          if (statusEl) statusEl.textContent = "Autocaption response was invalid.";
          return;
        }

        data.captions.forEach((cap, idx) => {
          const field = document.querySelector('[name="caption_' + idx + '"]');
          if (field && (!field.value || field.value.trim() === "")) {
            field.value = cap;
          }
        });

        if (statusEl) {
          const count = data.captions.length;
          statusEl.textContent = '✅ Auto-captioning finished successfully for ' + count + ' image(s).';
        }
        if (progressBar) {
          progressBar.style.width = "100%";
        }
      } catch (err) {
        console.error(err);
        if (statusEl) statusEl.textContent = "❌ Autocaption request failed: " + err.message;
        if (progressBar) {
          progressBar.style.width = "0%";
        }
      } finally {
        if (autoCapTimer) {
          clearInterval(autoCapTimer);
          autoCapTimer = null;
        }
        autoCapStart = null;
        if (btn) {
          btn.disabled = false;
          btn.textContent = "Auto-caption images";
        }
        // Hide progress bar after 3 seconds
        setTimeout(() => {
          const progressContainer = document.getElementById('autocap-progress-container');
          if (progressContainer) {
            progressContainer.style.display = "none";
          }
        }, 3000);
      }
    }

    function setupDropzone() {
        const dropArea = document.getElementById('preview');
        const fileInput = document.getElementById('images');
        if (!dropArea || !fileInput) return;
        
        ['dragenter','dragover'].forEach(type => {
          dropArea.addEventListener(type, function(e) {
            e.preventDefault();
            e.stopPropagation();
            dropArea.classList.add('dragover');
          });
        });
        ['dragleave','drop'].forEach(type => {
          dropArea.addEventListener(type, function(e) {
            e.preventDefault();
            e.stopPropagation();
            dropArea.classList.remove('dragover');
          });
        });
        dropArea.addEventListener('drop', function(e) {
          const dtFiles = e.dataTransfer && e.dataTransfer.files ? Array.from(e.dataTransfer.files) : [];
          const imageFiles = dtFiles.filter(f => f.type.startsWith('image/'));
          if (!imageFiles.length) return;
          
          // Get existing files from storedFiles
          const existingFiles = storedFiles.slice();
          const existingNames = new Set(existingFiles.map(f => f.name + f.size + f.lastModified));
          
          // Add only new files (avoid duplicates)
          const allFiles = existingFiles.slice();
          imageFiles.forEach(f => {
            const fileKey = f.name + f.size + f.lastModified;
            if (!existingNames.has(fileKey)) {
              allFiles.push(f);
              existingNames.add(fileKey);
            }
          });
          
          // Update stored files
          storedFiles = allFiles.slice();
          
          // Update file input
          const dt = new DataTransfer();
          allFiles.forEach(f => dt.items.add(f));
          fileInput.files = dt.files;
          
          previewImages({ target: fileInput, files: allFiles });
          
          // Update start button state (previewImages will also call it, but this ensures it happens)
          updateStartButtonState();
        });
    }
    
    // Live output updates via Server-Sent Events
    let eventSource = null;
    let sseBuffer = "";
    let lastSseFlush = 0;
    const SSE_FLUSH_INTERVAL_MS = 1000; // flush to UI every 1 second for responsive updates
    
    // Loss chart tracking
    let lossChart = null;
    let lossData = {
      steps: [],
      losses: []
    };
    let currentStep = 0;
    
    // Auto-refresh sample gallery during training
    let sampleGalleryPollInterval = null;
    const SAMPLE_GALLERY_POLL_INTERVAL_MS = 30000; // Check for new samples every 30 seconds
    let lastSampleFileCount = 0;

    function startSampleGalleryPolling() {
      // Clear any existing interval
      if (sampleGalleryPollInterval) {
        clearInterval(sampleGalleryPollInterval);
      }
      lastSampleFileCount = 0;
      
      // Start polling for new samples
      sampleGalleryPollInterval = setInterval(async () => {
        const finalOutDir = computeFinalOutputDir();
        if (!finalOutDir) return;
        
        try {
          const resp = await fetch('/list_samples?output_dir=' + encodeURIComponent(finalOutDir));
          if (!resp.ok) return;
          const data = await resp.json();
          const files = data.files || [];
          
          // If we found new files, refresh the gallery silently
          if (files.length > lastSampleFileCount) {
            const newCount = files.length - lastSampleFileCount;
            lastSampleFileCount = files.length;
            renderSampleGallery(files);
            // Update status briefly to show new samples were found
            setSampleGalleryStatus(files.length + ' sample file' + (files.length > 1 ? 's' : '') + ' (' + newCount + ' new)', false);
          }
        } catch (err) {
          // Silent fail - don't spam errors during polling
        }
      }, SAMPLE_GALLERY_POLL_INTERVAL_MS);
    }

    function stopSampleGalleryPolling() {
      if (sampleGalleryPollInterval) {
        clearInterval(sampleGalleryPollInterval);
        sampleGalleryPollInterval = null;
      }
    }

    function openLogStream() {
      ensureOutputVisible();
      
      // Close any existing eventSource before opening a new one
      // Store reference to avoid race conditions
      const oldEventSource = eventSource;
      if (oldEventSource) {
        try {
          oldEventSource.close();
        } catch (e) {
          // Ignore errors when closing old connection
        }
        eventSource = null;
      }
      
      // Clear buffer when starting a new stream
      sseBuffer = "";
      lastSseFlush = 0;
      
      // Create new EventSource connection
      eventSource = new EventSource('/stream');
      
      eventSource.onmessage = function(event) {
        // Check if eventSource was closed/cancelled while processing
        if (!eventSource || eventSource !== this) {
          return;
        }
        
        const outputText = document.getElementById("output-text");
        if (!outputText) return;
        
        const data = event.data;
        
        if (!data || data.trim() === "") {
          return;
        }
        
        // Handle SSE reconnect signal (server closes connection periodically)
        if (data.includes("[SSE_RECONNECT]")) {
          // Silently reconnect without showing anything to user
          setTimeout(function() {
            if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
              openLogStream();
            }
          }, 500);
          return;
        }
        
        // Handle training errors
        if (data.includes("TRAINING ABORTED WITH ERROR") || data.includes("torch.OutOfMemoryError")) {
          document.getElementById("spinner").style.display = "none";
          stopSampleGalleryPolling();
          const currentEventSource = eventSource;
          if (currentEventSource) {
            try {
              currentEventSource.close();
            } catch (e) {
              // Ignore errors
            }
            eventSource = null;
          }
          enableAllButtons();
          const outputContainer = document.getElementById("output-container");
          if (outputContainer) {
          outputContainer.style.border = "3px solid #ff5555";
          outputContainer.style.borderRadius = "10px";
          }
          
          // Final update of loss chart
          if (lossChart) {
            lossChart.update();
          }
          
          refreshSampleGallery(true);
          return;
        }
        
        // Handle training completion
        if (data.includes("[TRAINING_FINISHED]")) {
          document.getElementById("spinner").style.display = "none";
          stopSampleGalleryPolling();
          const currentEventSource = eventSource;
          if (currentEventSource) {
            try {
              currentEventSource.close();
            } catch (e) {
              // Ignore errors
            }
            eventSource = null;
          }
          enableAllButtons();
          const outputContainer = document.getElementById("output-container");
          if (outputContainer) {
          outputContainer.style.border = "3px solid #00ff00";
          outputContainer.style.borderRadius = "10px";
          }
          
          // Final update of loss chart
          if (lossChart) {
            lossChart.update();
          }
          
          // Show README link if available
          refreshReadmeLink();
          
          refreshSampleGallery(true);
          return;
        }
        
        // Parse loss values from data and update chart
        parseAndUpdateLoss(data);
        
        // Buffer and display regular log data
        const processedData = data + "\n";
        sseBuffer += processedData;
        const now = Date.now();
        
        // Flush buffer if interval elapsed or important message received
        if (!lastSseFlush || 
            (now - lastSseFlush) >= SSE_FLUSH_INTERVAL_MS || 
            data.includes("TRAINING ABORTED WITH ERROR") || 
            data.includes("✅ TRAINING COMPLETED!") || 
            data.includes("LoRA README generated")) {
          
          if (outputText) {
          outputText.textContent += sseBuffer;
          outputText.scrollTop = outputText.scrollHeight;
          }
          
          // Check if README was generated and show link
          if (sseBuffer.includes("LoRA README generated")) {
            refreshReadmeLink();
          }
          
          sseBuffer = "";
          lastSseFlush = now;
        }
      };
      
      eventSource.onerror = function(event) {
        // Auto-reconnect on error (common with Waitress/gevent)
        const currentEventSource = eventSource;
        if (currentEventSource && currentEventSource.readyState === EventSource.CLOSED) {
          const spinner = document.getElementById("spinner");
          // Only reconnect if training is still active
          if (spinner && spinner.style.display === "block") {
            // Training is active, try to reconnect after a delay
            setTimeout(function() {
              // Double-check training is still active before reconnecting
              const spinnerCheck = document.getElementById("spinner");
              if (spinnerCheck && spinnerCheck.style.display === "block" && !eventSource) {
                openLogStream();
              }
            }, 1000);
          }
        }
      };
    }

    function validateTrainingForm() {
      const errors = [];
      
      // Check for images
      const uploadedCount = document.getElementById('uploaded_count');
      const imageCount = uploadedCount ? parseInt(uploadedCount.value || "0") : 0;
      if (imageCount === 0) {
        errors.push("No training images selected. Please upload at least one image.");
      }
      
      // Check required fields
      const epochsInput = document.querySelector('input[name="epochs"]');
      const epochs = epochsInput ? parseInt(epochsInput.value || "0") : 0;
      if (!epochs || epochs <= 0) {
        errors.push("Epochs must be greater than 0.");
        if (epochsInput) epochsInput.classList.add('field-error');
      } else if (epochsInput) {
        epochsInput.classList.remove('field-error');
      }
      
      const lrInput = document.querySelector('input[name="lr"]');
      const lr = lrInput ? (lrInput.value || "").trim() : "";
      if (!lr) {
        errors.push("Learning rate is required.");
        if (lrInput) lrInput.classList.add('field-error');
      } else {
        // Validate learning rate format (should be like "1e-4" or "0.0001")
        const lrMatch = lr.match(/^[\d.]+(e-?\d+)?$/i);
        if (!lrMatch) {
          errors.push("Learning rate format is invalid. Use format like '1e-4' or '0.0001'.");
          if (lrInput) lrInput.classList.add('field-error');
        } else if (lrInput) {
          lrInput.classList.remove('field-error');
        }
      }
      
      const rankInput = document.querySelector('input[name="rank"]');
      const rank = rankInput ? parseInt(rankInput.value || "0") : 0;
      if (!rank || rank <= 0) {
        errors.push("LoRA rank must be greater than 0.");
        if (rankInput) rankInput.classList.add('field-error');
      } else if (rankInput) {
        rankInput.classList.remove('field-error');
      }
      
      const dimsInput = document.querySelector('input[name="dims"]');
      const dims = dimsInput ? parseInt(dimsInput.value || "0") : 0;
      if (!dims || dims <= 0) {
        errors.push("LoRA dims (alpha) must be greater than 0.");
        if (dimsInput) dimsInput.classList.add('field-error');
      } else if (dims < rank) {
        errors.push("LoRA dims (alpha) should be greater than or equal to LoRA rank. Recommended: dims = rank * 2.");
        if (dimsInput) dimsInput.classList.add('field-error');
      } else if (dimsInput) {
        dimsInput.classList.remove('field-error');
      }
      
      const resolutionInput = document.querySelector('input[name="resolution"]');
      const resolution = resolutionInput ? (resolutionInput.value || "").trim() : "";
      if (!resolution) {
        errors.push("Resolution is required (e.g., '1024x1024').");
        if (resolutionInput) resolutionInput.classList.add('field-error');
      } else {
        // Validate resolution format
        const resMatch = resolution.match(/^\d+\s*x\s*\d+$/i);
        if (!resMatch) {
          errors.push("Resolution format is invalid. Use format like '1024x1024'.");
          if (resolutionInput) resolutionInput.classList.add('field-error');
        } else if (resolutionInput) {
          resolutionInput.classList.remove('field-error');
        }
      }
      
      const batchSizeInput = document.querySelector('input[name="batch_size"]');
      const batchSize = batchSizeInput ? parseInt(batchSizeInput.value || "0") : 0;
      if (!batchSize || batchSize <= 0) {
        errors.push("Batch size must be greater than 0.");
        if (batchSizeInput) batchSizeInput.classList.add('field-error');
      } else if (batchSizeInput) {
        batchSizeInput.classList.remove('field-error');
      }
      
      return errors;
    }
    
    function showValidationErrors(errors) {
      // Remove any existing validation error box
      const existingError = document.getElementById('validation-error-box');
      if (existingError) {
        existingError.remove();
      }
      
      // Create error box
      const errorBox = document.createElement('div');
      errorBox.id = 'validation-error-box';
      errorBox.className = 'validation-error';
      errorBox.style.display = 'block';
      errorBox.style.zIndex = '1000';
      errorBox.innerHTML = '<h4>⚠️ Cannot start training - Please fix the following issues:</h4><ul>' +
        errors.map(err => '<li>' + err + '</li>').join('') +
        '</ul>';
      
      // Try to insert before the start button's parent container
      const startBtn = document.getElementById('start-btn');
      if (startBtn) {
        // Find the button row or parent container
        let container = startBtn.parentNode;
        // If it's a div with class button-row, use that, otherwise use parent
        if (container && container.classList && container.classList.contains('button-row')) {
          container.insertBefore(errorBox, startBtn);
        } else {
          // Insert before the button row or start button
          if (container) {
            container.insertBefore(errorBox, startBtn);
          } else {
            // Fallback: insert at the end of training-settings section
            const trainingSettings = document.getElementById('training-settings');
            if (trainingSettings) {
              trainingSettings.appendChild(errorBox);
            }
          }
        }
        
        // Scroll to error box after a short delay to ensure it's rendered
        setTimeout(function() {
          errorBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
      } else {
        // Fallback: insert at the end of training-settings section
        const trainingSettings = document.getElementById('training-settings');
        if (trainingSettings) {
          trainingSettings.appendChild(errorBox);
          setTimeout(function() {
            errorBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }, 100);
        }
      }
      
      // Also show an alert for immediate feedback
      alert('⚠️ Cannot start training!\n\nPlease fix the following issues:\n\n' + errors.join('\n'));
    }
    
    function startTraining() {
      // Validate form first
      const errors = validateTrainingForm();
      if (errors.length > 0) {
        showValidationErrors(errors);
        return;
      }
      
      // Remove any existing validation error box
      const existingError = document.getElementById('validation-error-box');
      if (existingError) {
        existingError.remove();
      }
      
      // Show confirmation dialog
      const confirmStart = window.confirm(
        "Are you sure you want to start training?\n\n" +
        "This will begin the training process with your current settings.\n" +
        "Training can take a long time depending on your configuration.\n\n" +
        "Click OK to start training, or Cancel to review your settings."
      );
      
      if (!confirmStart) {
        return;
      }
      
      showSpinner();
      
      // Ensure any existing eventSource is closed before opening a new one
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      
      // Open log stream to receive training updates
      openLogStream();
      
      // Start auto-refreshing sample gallery during training
      const samplesCheckbox = document.getElementById('samples_enabled');
      if (samplesCheckbox && samplesCheckbox.checked) {
        startSampleGalleryPolling();
      }
      
      // Submit form via AJAX to prevent page reload
      var form = document.getElementById('training-form');
      var formData = new FormData(form);
      formData.append('action', 'train'); // Manually add action since we are not using submit button
      
      fetch('/', {
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (!response.ok) {
            document.getElementById("output-text").textContent += "❌ Server error when starting training: " + response.statusText + "\n";
            stopSampleGalleryPolling();
            if (eventSource) {
              eventSource.close();
              eventSource = null;
            }
        }
      })
      .catch(error => {
            document.getElementById("output-text").textContent += "❌ Network error when starting training: " + error + "\n";
            stopSampleGalleryPolling();
            if (eventSource) {
              eventSource.close();
              eventSource = null;
            }
      });
    }

    async function cancelTraining() {
      const confirmCancel = window.confirm("Are you sure you want to cancel the current training run?");
      if (!confirmCancel) return;
      stopSampleGalleryPolling();
      
      // Immediately show cancel message in console
      const outputText = document.getElementById("output-text");
      const outputContainer = document.getElementById("output-container");
      if (outputContainer) {
        outputContainer.style.display = "block";
      }
      if (outputText) {
        outputText.textContent += "\n" + "=".repeat(60) + "\n";
        outputText.textContent += "⏹ TRAINING CANCEL REQUEST RECEIVED. Attempting to stop processes...\n";
        outputText.textContent += "Unloading models from VRAM/RAM...\n";
        outputText.textContent += "=".repeat(60) + "\n";
        outputText.scrollTop = outputText.scrollHeight;
      }
      
      const statusEl = document.getElementById('autocap-status');
      try {
        const resp = await fetch('/cancel_training', { method: 'POST' });
        if (!resp.ok) {
          const text = await resp.text();
          if (statusEl) statusEl.textContent = "Cancel request failed: " + text;
          if (outputText) {
            outputText.textContent += "❌ Cancel request failed: " + text + "\n";
            outputText.scrollTop = outputText.scrollHeight;
          }
          return;
        }
        
        // Read response message
        const result = await resp.json();
        const successMessage = result.message || "Training canceled. Processes stopped.";
        
        if (statusEl) statusEl.textContent = successMessage;
        
        // Show success message in console
        if (outputText) {
          outputText.textContent += "✅ " + successMessage + "\n";
          outputText.textContent += "=".repeat(60) + "\n";
          outputText.scrollTop = outputText.scrollHeight;
        }
        
        // Update UI to show training is finished
        document.getElementById("spinner").style.display = "none";
        stopSampleGalleryPolling();
        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }
        enableAllButtons();
        if (outputContainer) {
          outputContainer.style.border = "3px solid #ff5555";
          outputContainer.style.borderRadius = "10px";
        }
      } catch (err) {
        if (statusEl) statusEl.textContent = "Cancel request error: " + err;
        if (outputText) {
          outputText.textContent += "❌ Cancel request error: " + err + "\n";
          outputText.scrollTop = outputText.scrollHeight;
        }
      }
    }
    
    // Make ALL functions available globally for inline handlers - MUST be after all function definitions
    window.openImageZoom = openImageZoom;
    window.closeImageZoom = closeImageZoom;
    window.handleFileInputChange = handleFileInputChange;
    window.clearSelectedImages = clearSelectedImages;
    window.previewImages = previewImages;
    window.removeImage = removeImage;
    window.autoCaptionCaptionModel = autoCaptionCaptionModel;
    window.updateCaptionLengthLabel = updateCaptionLengthLabel;
    window.updateDetailLevelLabel = updateDetailLevelLabel;
    window.updateFinalPath = updateFinalPath;
    window.updateEstimate = updateEstimate;
    window.updateCommandPreview = updateCommandPreview;
    window.syncAdvancedFlagsFromGui = syncAdvancedFlagsFromGui;
    window.refreshSampleGallery = refreshSampleGallery;
    window.startTraining = startTraining;
    window.cancelTraining = cancelTraining;
    window.resumeLogStream = resumeLogStream;
    window.setSampleGalleryStatus = setSampleGalleryStatus;
    window.renderSampleGallery = renderSampleGallery;
    window.computeFinalOutputDir = computeFinalOutputDir;
    window.markSampleGalleryPending = markSampleGalleryPending;
    window.applyTrainingPreset = applyTrainingPreset;
    window.updatePresetDropdown = updatePresetDropdown;
    window.updatePresetValuesForImageCount = updatePresetValuesForImageCount;
    
    // Preset configurations
    // Updated according to best practices for LoRA training
    const presetConfigs = {
      'fast': {
        epochs: 2,
        batch_size: 1,
        image_repeats: 8,
        lr: '5e-5',
        rank: 16,
        dims: 16
      },
      'balanced': {
        epochs: 3,
        batch_size: 1,
        image_repeats: 6,
        lr: '4e-5',
        rank: 32,
        dims: 32
      },
      'high-quality': {
        epochs: 4,
        batch_size: 1,
        image_repeats: 5,
        lr: '3e-5',
        rank: 64,
        dims: 32
      },
      'person': {
        epochs: 3,
        batch_size: 1,
        image_repeats: 7,
        lr: '5e-5',
        rank: 32,
        dims: 32
      },
      'style': {
        epochs: 4,
        batch_size: 1,
        image_repeats: 5,
        lr: '2e-5',
        rank: 16,
        dims: 8
      },
      'few-images': {
        epochs: 5,
        batch_size: 1,
        image_repeats: 10,
        lr: '2e-5',
        rank: 16,
        dims: 16
      },
      'many-images': {
        epochs: 3,
        batch_size: 1,
        image_repeats: 2,
        lr: '5e-5',
        rank: 64,
        dims: 32
      }
    };
    
    // Check if current form values match a preset
    // Note: We match on fixed values (lr, rank, dims, batch_size) and ignore epochs/repeats
    // since those are dynamically adjusted based on image count
    function detectCurrentPreset() {
      const batchSizeInput = document.querySelector('input[name="batch_size"]');
      const lrInput = document.querySelector('input[name="lr"]');
      const rankInput = document.querySelector('input[name="rank"]');
      const dimsInput = document.querySelector('input[name="dims"]');
      
      if (!batchSizeInput || !lrInput || !rankInput || !dimsInput) {
        return 'custom';
      }
      
      const currentValues = {
        batch_size: parseInt(batchSizeInput.value) || 0,
        lr: (lrInput.value || '').trim(),
        rank: parseInt(rankInput.value) || 0,
        dims: parseInt(dimsInput.value) || 0
      };
      
      // Check each preset to see if fixed values match
      // We ignore epochs and image_repeats since they're dynamically adjusted
      for (const [presetName, presetConfig] of Object.entries(presetConfigs)) {
        if (currentValues.batch_size === presetConfig.batch_size &&
            currentValues.lr === presetConfig.lr &&
            currentValues.rank === presetConfig.rank &&
            currentValues.dims === presetConfig.dims) {
          return presetName;
        }
      }
      
      return 'custom';
    }
    
    // Calculate dynamic image_repeats and epochs based on number of images
    // Also adjusts learning rate for small/large datasets
    function calculateDynamicRepeatsAndEpochs(preset, baseEpochs, baseRepeats, baseLr, imageCount) {
      let adjustedEpochs = baseEpochs;
      let adjustedRepeats = baseRepeats;
      let adjustedLr = baseLr;
      
      if (imageCount <= 0) {
        // No images, use base values
        return { epochs: adjustedEpochs, repeats: adjustedRepeats, lr: adjustedLr };
      }
      
      // Convert LR string to number for calculations
      let lrValue = parseFloat(baseLr);
      if (isNaN(lrValue)) {
        // Try to parse scientific notation
        const match = baseLr.match(/([\d.]+)e-?(\d+)/i);
        if (match) {
          lrValue = parseFloat(match[1]) * Math.pow(10, -parseInt(match[2]));
        } else {
          lrValue = 5e-5; // Default fallback
        }
      }
      
      // Adjust based on image count
      if (imageCount < 20) {
        // Very few images: high repeats, more epochs, lower LR to prevent overfitting
        adjustedEpochs = Math.round(baseEpochs * 1.5);
        adjustedRepeats = Math.round(baseRepeats * 2.0);
        lrValue = lrValue * 0.5;
      } else if (imageCount < 50) {
        // Few images: moderate-high repeats, moderate epochs, slightly lower LR
        adjustedEpochs = Math.round(baseEpochs * 1.2);
        adjustedRepeats = Math.round(baseRepeats * 1.3);
        lrValue = lrValue * 0.7;
      } else if (imageCount < 100) {
        // Medium images: use base preset values
        adjustedEpochs = baseEpochs;
        adjustedRepeats = baseRepeats;
        // Keep base learning rate
      } else if (imageCount < 300) {
        // Many images: lower repeats, fewer epochs
        adjustedEpochs = Math.round(baseEpochs * 0.8);
        adjustedRepeats = Math.round(baseRepeats * 0.5);
        // Keep base learning rate
      } else {
        // Very many images (300+): low repeats, fewer epochs, slightly higher LR
        adjustedEpochs = Math.round(baseEpochs * 0.6);
        adjustedRepeats = Math.round(baseRepeats * 0.3);
        lrValue = lrValue * 1.2;
      }
      
      // Special handling for "few-images" and "many-images" presets
      if (preset === 'few-images') {
        // This preset is specifically for few images, ensure minimum 50 total repeats
        const totalRepeats = imageCount * adjustedRepeats * adjustedEpochs;
        if (totalRepeats < 50) {
          // Adjust repeats to reach minimum 50 total repeats
          adjustedRepeats = Math.max(10, Math.ceil(50 / (imageCount * adjustedEpochs)));
        }
      } else if (preset === 'many-images') {
        // This preset is specifically for many images, keep low repeats
        adjustedRepeats = Math.max(2, Math.min(4, adjustedRepeats));
        adjustedEpochs = Math.max(2, Math.min(5, adjustedEpochs));
      }
      
      // Critical check for Character/Person preset
      // Characters need ~40-60 total lookups per image to learn likeness
      if (preset === 'person') {
        const totalRepeats = imageCount * adjustedRepeats * adjustedEpochs;
        const targetTotalRepeats = 50; // Target 50 total repeats minimum
        if (totalRepeats < 40) {
          // Adjust repeats to reach target total repeats
          adjustedRepeats = Math.max(1, Math.ceil(targetTotalRepeats / (imageCount * adjustedEpochs)));
        }
        // Ensure we're in the 50-60 range for optimal likeness learning
        const finalTotalRepeats = imageCount * adjustedRepeats * adjustedEpochs;
        if (finalTotalRepeats > 60) {
          // Slightly reduce if we're over 60
          adjustedRepeats = Math.max(1, Math.floor(60 / (imageCount * adjustedEpochs)));
        }
      }
      
      // Convert LR back to string format (e.g., "5e-5" not "1.0e-5")
      // Use the same format as the original presets
      let lrString = baseLr; // Default to original format
      const baseLrValue = parseFloat(baseLr);
      if (isNaN(baseLrValue)) {
        // Try to parse scientific notation
        const match = baseLr.match(/([\d.]+)e-?(\d+)/i);
        if (match) {
          const baseLrParsed = parseFloat(match[1]) * Math.pow(10, -parseInt(match[2]));
          if (Math.abs(lrValue - baseLrParsed) > 0.0000001) {
            // LR was adjusted, convert to scientific notation in compact format
            const exponent = Math.floor(Math.log10(lrValue));
            const mantissa = lrValue / Math.pow(10, exponent);
            // Use integer mantissa if possible (e.g., "5e-5" instead of "1.0e-5")
            if (Math.abs(mantissa - Math.round(mantissa)) < 0.01) {
              lrString = Math.round(mantissa) + 'e-' + Math.abs(exponent);
            } else {
              // Round to 1 decimal place and remove trailing zero
              const rounded = parseFloat(mantissa.toFixed(1));
              lrString = rounded + 'e-' + Math.abs(exponent);
            }
          }
        }
      } else if (Math.abs(lrValue - baseLrValue) > 0.0000001) {
        // LR was adjusted, convert to scientific notation in compact format
        const exponent = Math.floor(Math.log10(lrValue));
        const mantissa = lrValue / Math.pow(10, exponent);
        // Use integer mantissa if possible (e.g., "5e-5" instead of "1.0e-5")
        if (Math.abs(mantissa - Math.round(mantissa)) < 0.01) {
          lrString = Math.round(mantissa) + 'e-' + Math.abs(exponent);
        } else {
          // Round to 1 decimal place and remove trailing zero
          const rounded = parseFloat(mantissa.toFixed(1));
          lrString = rounded + 'e-' + Math.abs(exponent);
        }
      }
      
      return { epochs: adjustedEpochs, repeats: adjustedRepeats, lr: lrString };
    }
    
    // Training preset function - applies preset values to form fields
    function applyTrainingPreset() {
      const presetSelect = document.getElementById('training-preset');
      if (!presetSelect) return;
      const preset = presetSelect.value;
      
      // If custom, don't change anything, but update dropdown to show detected preset
      if (preset === 'custom') {
        const detectedPreset = detectCurrentPreset();
        if (detectedPreset !== 'custom' && presetSelect.value === 'custom') {
          // Don't change if user explicitly selected custom
          return;
        }
        return;
      }
      
      // Get number of uploaded images
      const uploadedCountEl = document.getElementById('uploaded_count');
      const imageCount = uploadedCountEl ? parseInt(uploadedCountEl.value || "0") : 0;
      
      // Get form fields
      const epochsInput = document.querySelector('input[name="epochs"]');
      const batchSizeInput = document.querySelector('input[name="batch_size"]');
      const imageRepeatsInput = document.querySelector('input[name="image_repeats"]');
      const lrInput = document.querySelector('input[name="lr"]');
      const rankInput = document.querySelector('input[name="rank"]');
      const dimsInput = document.querySelector('input[name="dims"]');
      
      const presetConfig = presetConfigs[preset];
      if (!presetConfig) return;
      
      // Calculate dynamic repeats, epochs, and learning rate based on image count
      const dynamic = calculateDynamicRepeatsAndEpochs(
        preset,
        presetConfig.epochs,
        presetConfig.image_repeats,
        presetConfig.lr,
        imageCount
      );
      
      // Temporarily disable preset detection to avoid feedback loop
      let presetDetectionDisabled = true;
      
      // Apply preset values with dynamic adjustments
      if (epochsInput) epochsInput.value = dynamic.epochs;
      if (batchSizeInput) batchSizeInput.value = presetConfig.batch_size;
      if (imageRepeatsInput) imageRepeatsInput.value = dynamic.repeats;
      if (lrInput) lrInput.value = dynamic.lr; // Use dynamically adjusted LR
      if (rankInput) rankInput.value = presetConfig.rank;
      if (dimsInput) dimsInput.value = presetConfig.dims;
      
      // Ensure dropdown shows the selected preset
      presetSelect.value = preset;
      
      // Trigger update events to refresh estimates and command preview
      if (epochsInput) {
        epochsInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
      if (batchSizeInput) {
        batchSizeInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
      if (imageRepeatsInput) {
        imageRepeatsInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
      if (lrInput) {
        lrInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
      if (rankInput) {
        rankInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
      if (dimsInput) {
        dimsInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
      
      presetDetectionDisabled = false;
      
      // Update estimate and command preview
      if (typeof updateEstimate === 'function') {
        updateEstimate();
      }
      if (typeof updateCommandPreview === 'function') {
        updateCommandPreview();
      }
    }
    
    // Update preset dropdown based on current values
    function updatePresetDropdown() {
      const presetSelect = document.getElementById('training-preset');
      if (!presetSelect) return;
      
      const detectedPreset = detectCurrentPreset();
      if (presetSelect.value !== detectedPreset) {
        presetSelect.value = detectedPreset;
      }
    }
    
    // Update preset values (epochs and repeats) when image count changes
    // This is called when images are added/removed and a preset is active
    function updatePresetValuesForImageCount() {
      const presetSelect = document.getElementById('training-preset');
      if (!presetSelect || presetSelect.value === 'custom') {
        return; // Don't update if custom is selected
      }
      
      const preset = presetSelect.value;
      const presetConfig = presetConfigs[preset];
      if (!presetConfig) return;
      
      // Get current number of images
      const uploadedCountEl = document.getElementById('uploaded_count');
      const imageCount = uploadedCountEl ? parseInt(uploadedCountEl.value || "0") : 0;
      
      // Calculate dynamic repeats, epochs, and learning rate
      const dynamic = calculateDynamicRepeatsAndEpochs(
        preset,
        presetConfig.epochs,
        presetConfig.image_repeats,
        presetConfig.lr,
        imageCount
      );
      
      // Update epochs, repeats, and learning rate
      const epochsInput = document.querySelector('input[name="epochs"]');
      const imageRepeatsInput = document.querySelector('input[name="image_repeats"]');
      const lrInput = document.querySelector('input[name="lr"]');
      
      if (epochsInput && parseInt(epochsInput.value) !== dynamic.epochs) {
        epochsInput.value = dynamic.epochs;
        epochsInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
      
      if (imageRepeatsInput && parseInt(imageRepeatsInput.value) !== dynamic.repeats) {
        imageRepeatsInput.value = dynamic.repeats;
        imageRepeatsInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
      
      if (lrInput && lrInput.value !== dynamic.lr) {
        lrInput.value = dynamic.lr;
        lrInput.dispatchEvent(new Event('input', { bubbles: true }));
      }
      
      // Update estimate and command preview
      if (typeof updateEstimate === 'function') {
        updateEstimate();
      }
      if (typeof updateCommandPreview === 'function') {
        updateCommandPreview();
      }
    }
    
    // Initialize page when DOM is ready
    let initDone = false;
    window.initializePage = function initializePage() {
      if (initDone) return;
      initDone = true;
      
      // Update start button state on page load
      updateStartButtonState();
      
      // Load saved draft from localStorage
      loadDraftFromStorage();
      
      // Setup auto-save to localStorage
      setupAutoSave();
      
      // Setup smooth scroll and active state for navigation
      setupNavigation();
      
      // Restore collapsed section states from localStorage
      restoreSectionStates();
      
      // Start system resources monitoring with delay to avoid blocking
      // Update more frequently during training (when it's most important)
      let resourcePollInterval = null;
      function startResourcePolling() {
        if (resourcePollInterval) return;
        resourcePollInterval = setInterval(function() {
          if (typeof updateSystemResources === 'function') {
        updateSystemResources();
          }
        }, 3000); // 3 seconds - more frequent updates for better live feedback
      }
      setTimeout(function() {
        try {
          if (typeof updateSystemResources === 'function') {
            updateSystemResources(); // Initial update
            startResourcePolling();
          }
      } catch (err) {
        console.error('Failed to initialize system resources monitoring:', err);
      }
      }, 500);
      
      try {
        // Safely check for spinner and start training if needed
      try {
        const spinner = document.getElementById("spinner");
          if (spinner && spinner.style.display === "block" && typeof startTraining === 'function') {
          startTraining();
        }
        } catch (err) {
          console.error('Error checking spinner:', err);
        }
        
        // Setup dropzone
        try {
          if (typeof setupDropzone === 'function') {
        setupDropzone();
          }
        } catch (err) {
          console.error('Error setting up dropzone:', err);
        }
        
        // Setup advanced toggle button
        const advancedToggleBtn = document.getElementById('advanced-toggle');
        if (advancedToggleBtn) {
          advancedToggleBtn.onclick = function() {
              window.toggleAdvancedFlags();
          };
            }
        
        // Attach listeners to keep command preview in sync with form values
        // Also reset preset to "custom" when user manually changes values
        try {
        const names = [
          "epochs","batch_size","image_repeats","lr","resolution",
          "seed","rank","dims","output_dir","output_name",
          "sample_every_epochs","sample_every_steps"
        ];
        names.forEach(n => {
            try {
          const el = document.querySelector('input[name="' + n + '"]');
              if (el && typeof updateCommandPreview === 'function') {
                el.addEventListener('input', function() {
                  // Remove field error styling when user starts typing
                  this.classList.remove('field-error');
                  
                  updateCommandPreview();
                  // Update preset dropdown when user manually changes training parameters
                  if (['epochs', 'batch_size', 'image_repeats', 'lr', 'rank', 'dims'].includes(n)) {
                    // Use setTimeout to allow value to update first
                    setTimeout(function() {
                      if (typeof updatePresetDropdown === 'function') {
                        updatePresetDropdown();
                      }
                    }, 10);
                  }
                });
                el.addEventListener('change', function() {
                  this.classList.remove('field-error');
                  updateCommandPreview();
                });
              }
            } catch (err) {
              console.error('Error attaching listener to ' + n + ':', err);
            }
          });
        } catch (err) {
          console.error('Error setting up form listeners:', err);
        }
        
        // Setup VRAM profile selector
        try {
        const vramSel = document.querySelector('select[name="vram_profile"]');
          if (vramSel && typeof syncAdvancedFlagsFromGui === 'function') {
          vramSel.addEventListener('change', function() {
            syncAdvancedFlagsFromGui(false);
          });
        }
        } catch (err) {
          console.error('Error setting up VRAM selector:', err);
        }
        
        // Setup optimizer selector
        try {
        const optSel = document.querySelector('select[name="optimizer_type"]');
          if (optSel && typeof updateCommandPreview === 'function') {
          optSel.addEventListener('change', updateCommandPreview);
        }
        } catch (err) {
          console.error('Error setting up optimizer selector:', err);
        }
        
        // Setup advanced flags textarea
        try {
        const advFlags = document.querySelector('textarea[name="advanced_flags"]');
          if (advFlags && typeof updateCommandPreview === 'function') {
          advFlags.addEventListener('input', function() {
            advFlagsDirty = true;
            updateCommandPreview();
          });
        }
        } catch (err) {
          console.error('Error setting up advanced flags:', err);
        }
        
        // Setup sample prompt textarea
        try {
        const samplePromptTextarea = document.getElementById('sample_prompt_text');
          if (samplePromptTextarea && typeof updateCommandPreview === 'function') {
          samplePromptTextarea.addEventListener('input', updateCommandPreview);
        }
        } catch (err) {
          console.error('Error setting up sample prompt textarea:', err);
        }
        
        // Setup samples checkbox
        try {
        const samplesCheckbox = document.getElementById('samples_enabled');
        if (samplesCheckbox) {
          samplesCheckbox.addEventListener('change', () => {
            if (window.toggleSampleOptions) {
              window.toggleSampleOptions();
            }
              if (typeof updateCommandPreview === 'function') {
            updateCommandPreview();
              }
            });
          }
        } catch (err) {
          console.error('Error setting up samples checkbox:', err);
        }
        
        // Setup sample at first checkbox
        try {
        const sampleAtFirstCheckbox = document.getElementById('sample_at_first');
          if (sampleAtFirstCheckbox && typeof updateCommandPreview === 'function') {
          sampleAtFirstCheckbox.addEventListener('change', updateCommandPreview);
        }
        } catch (err) {
          console.error('Error setting up sample at first checkbox:', err);
        }
        
        // Initialize form values - wrap each in try-catch
        try {
          if (typeof syncAdvancedFlagsFromGui === 'function') {
        syncAdvancedFlagsFromGui(true);
          }
        } catch (err) {
          console.error('Error syncing advanced flags:', err);
        }
        
        try {
        if (window.toggleSampleOptions) {
          window.toggleSampleOptions();
        }
        } catch (err) {
          console.error('Error toggling sample options:', err);
        }
        
        try {
          if (typeof updateCommandPreview === 'function') {
        updateCommandPreview();
          }
        } catch (err) {
          console.error('Error updating command preview:', err);
        }
        
        try {
          if (typeof updateCaptionLengthLabel === 'function') {
        updateCaptionLengthLabel();
          }
        } catch (err) {
          console.error('Error updating caption length label:', err);
        }
        
        try {
          if (typeof updateDetailLevelLabel === 'function') {
        updateDetailLevelLabel();
          }
        } catch (err) {
          console.error('Error updating detail level label:', err);
        }
        
        try {
          if (typeof updateFinalPath === 'function') {
        updateFinalPath();
          }
        } catch (err) {
          console.error('Error updating final path:', err);
        }
        
        // Detect and set initial preset based on current form values
        try {
          setTimeout(function() {
            if (typeof updatePresetDropdown === 'function') {
              updatePresetDropdown();
            }
          }, 100);
        } catch (err) {
          console.error('Error detecting initial preset:', err);
        }
        
        // Load sample gallery and README link with delay to avoid blocking
        setTimeout(function() {
          try {
            if (typeof refreshSampleGallery === 'function') {
          refreshSampleGallery(true);
            }
        } catch (err) {
          console.error('Failed to load sample gallery on page load:', err);
        }
        try {
            if (typeof refreshReadmeLink === 'function') {
          refreshReadmeLink();
            }
        } catch (err) {
          console.error('Failed to refresh README link on load:', err);
        }
        }, 200);
        
        
        // Load server info for footer
        setTimeout(function() {
          try {
            fetch('/server_info')
              .then(resp => resp.json())
              .then(data => {
                const infoEl = document.getElementById('server-info');
                if (infoEl && data) {
                  const parts = [];
                  if (data.version) parts.push('v' + data.version);
                  if (data.webserver) parts.push(data.webserver);
                  if (data.os) {
                    const osStr = data.os_release ? data.os + ' ' + data.os_release : data.os;
                    parts.push(osStr);
                  }
                  if (data.gpu) {
                    // Shorten GPU name if too long (e.g., "NVIDIA GeForce RTX 4090" -> "RTX 4090")
                    let gpuStr = data.gpu;
                    if (gpuStr.length > 20) {
                      const match = gpuStr.match(/(RTX|GTX|RX|A\d+)\s*[\d\w]+/i);
                      if (match) {
                        gpuStr = match[0];
            } else {
                        gpuStr = gpuStr.substring(0, 17) + '...';
                      }
                    }
                    parts.push(gpuStr);
                  }
                  if (data.pytorch_version) {
                    parts.push('PyTorch ' + data.pytorch_version.split('+')[0]); // Remove +cu118 etc.
                  }
                  if (data.cuda_version) {
                    parts.push('CUDA ' + data.cuda_version);
                  }
                  if (parts.length > 0) {
                    infoEl.textContent = '• ' + parts.join(' • ');
                  }
                }
              })
              .catch(err => {
                // Silently fail - server info is not critical
              });
          } catch (err) {
            // Silently fail
          }
        }, 300);
        
      } catch (err) {
        console.error('❌ Error during page initialization:', err);
      }
    }
    
    // Run initialization when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
      window.initializePage();
    });
  </script>
</body>
</html>
'''

def ensure_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d)

def resolve_output_path(rel_path: str) -> str:
    """
    Ensure a path stays inside the project output directory.
    Accepts strings that start with 'output'.
    """
    if not rel_path:
        raise ValueError("Empty path")
    normalized = rel_path.replace("\\", "/")
    if normalized.startswith("/"):
        normalized = normalized.lstrip("/")
    if not normalized.startswith("output"):
        raise ValueError("Path must stay inside output/")
    abs_path = os.path.abspath(os.path.join(PROJECT_ROOT, normalized))
    if not abs_path.startswith(OUTPUT_ROOT):
        raise ValueError("Path escapes output directory")
    return abs_path

def save_uploaded_images(files, captions, output_dir, trigger_word=""):
    ensure_dir(output_dir)
    img_files = []
    for idx, fileobj in enumerate(files):
        fname = secure_filename(fileobj.filename)
        filepath = os.path.join(output_dir, fname)
        fileobj.save(filepath)
        caption_file = os.path.join(output_dir, os.path.splitext(fname)[0] + ".txt")
        cap = captions[idx] if idx < len(captions) else ""
        full_caption = (cap or "").strip()
        tw = (trigger_word or "").strip()
        if tw:
            # Only prepend trigger word if it's not already present (case-insensitive)
            if tw.lower() not in full_caption.lower():
                full_caption = f"{tw} {full_caption}".strip()
        with open(caption_file, "w", encoding="utf-8") as cf:
            cf.write(full_caption)
        img_files.append(fname)
    return img_files

def generate_lora_readme(output_dir, output_name, trigger, num_images, epochs, batch_size, image_repeats, 
                          learning_rate, rank, dims, resolution, prompt, sample_prompt_text=None):
    """
    Generate a CivitAI-ready README.md file for the trained LoRA.
    Format inspired by CivitAI's LoRA descriptions with bullet points and example prompts.
    """
    readme_path = os.path.join(output_dir, "LORA_README.md")
    
    # Calculate total steps
    effective_images = num_images * image_repeats
    steps_per_epoch = (effective_images + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    
    # Generate title (use trigger word or output name)
    title = trigger if trigger and trigger.strip() else output_name
    title = title.replace("_", " ").title()
    
    # Generate description from prompt if available, otherwise create a default
    description = ""
    if prompt and prompt.strip():
        description = prompt.strip()
    else:
        description = f"A LoRA trained for {title}."
    
    # Extract example prompts from sample_prompt_text
    example_prompts = []
    if sample_prompt_text:
        lines = sample_prompt_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Replace {{trigger}} with actual trigger word
                prompt_line = line.replace("{{trigger}}", trigger if trigger else output_name)
                # Remove parameter flags for example display (keep the prompt part)
                # Extract just the prompt part (before --w, --h, etc.)
                prompt_part = prompt_line.split("--")[0].strip()
                if prompt_part:
                    example_prompts.append(prompt_part)
    
    # Generate feature bullets based on training settings
    features = []
    if num_images >= 20:
        features.append(f"Trained on {num_images}+ images for robust performance")
    elif num_images >= 10:
        features.append(f"Trained on {num_images} images")
    else:
        features.append(f"Trained on {num_images} images (consider adding more for better results)")
    
    if rank >= 32:
        features.append("High-capacity LoRA (rank {}) for capturing fine details".format(rank))
    else:
        features.append("Efficient LoRA (rank {}) with balanced detail".format(rank))
    
    if resolution and "1024" in str(resolution):
        features.append("Optimized for high-resolution generation ({} resolution)".format(resolution))
    
    features.append(f"Total training steps: {total_steps} steps over {epochs} epochs")
    
    # Generate README content in CivitAI-style format
    readme_content = f"""# {title} LoRA

{description}

"""
    
    # Add features as bullet points (like CivitAI format)
    if features:
        readme_content += "## Features\n\n"
        for feature in features:
            readme_content += f"- {feature}\n"
        readme_content += "\n"
    
    # Add example prompts if available
    if example_prompts:
        readme_content += "## Example Prompts\n\n"
        for i, example in enumerate(example_prompts[:5], 1):  # Limit to 5 examples
            readme_content += f"{i}. {example}\n\n"
    
    # Add usage instructions
    readme_content += f"""## Usage

Use the trigger word **`{trigger if trigger else output_name}`** in your prompts to activate this LoRA.

## Training Details

- **Base Model**: Qwen-Image
- **Training Images**: {num_images} images
- **Image Repeats**: {image_repeats}x per epoch
- **Total Training Steps**: {total_steps} steps ({epochs} epochs × {steps_per_epoch} steps/epoch)
- **Training Resolution**: {resolution}
- **LoRA Rank**: {rank}
- **LoRA Alpha**: {dims}
- **Learning Rate**: {learning_rate}
- **Batch Size**: {batch_size}
- **Optimizer**: AdamW (with bfloat16 mixed precision)
- **Training completed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This LoRA was trained using Musubi Tuner with MusubiTLX - A simple web GUI for training Qwen-Image LoRA models with Musubi Tuner.
"""
    
    try:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        return readme_path
    except Exception as e:
        return None

def write_dataset_config_toml(folder, batch_size, resolution, num_repeats):
    toml_path = os.path.join(folder, "dataset_config.toml")
    w, h = (int(x) for x in resolution.split("x"))
    try:
        repeats = int(str(num_repeats)) if str(num_repeats).strip() else 1
    except ValueError:
        repeats = 1
    config = {
        "general": {
            "resolution": [w, h],
            "batch_size": int(batch_size),
            "enable_bucket": True,
            "bucket_no_upscale": False
        },
        "datasets": [
            {
                "image_directory": folder,
                "caption_extension": ".txt",
                "num_repeats": repeats
            }
        ]
    }
    with open(toml_path, "w") as f:
        toml.dump(config, f)
    return toml_path

def run_cache_latents(dataset_config, vae_model, stream_output=False, log_file=None):
    cmd = [
        sys.executable, "src/musubi_tuner/qwen_image_cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_model
    ]
    if stream_output:
        # Create subprocess in new session so it survives SSH disconnect
        # start_new_session=True creates a new process group, preventing SIGHUP propagation
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=0, universal_newlines=True,
                                start_new_session=True if platform.system() != 'Windows' else False)
        register_process(proc)
        output = ""
        try:
            for line in iter(proc.stdout.readline, ''):
                if line:
                    # Normalize progress output: replace carriage return with newline
                    line = line.replace('\r\n', '\n').replace('\r', '\n')
                    output_queue.put(line)
                    if log_file:
                        log_file.write(line)
                        log_file.flush()
                    output += line
                    sys.stdout.flush()
            proc.wait()
            unregister_process(proc)  # Always unregister when process finishes
            # Check if process crashed or was killed
            if proc.returncode != 0:
                # SIGTERM (-15) means the process was cancelled gracefully - don't raise exception
                # On Windows, signal module may not have SIGTERM, so check for -15 directly
                try:
                    import signal
                    is_sigterm = proc.returncode == -signal.SIGTERM or proc.returncode == -15
                except (AttributeError, ImportError):
                    # Windows doesn't have SIGTERM in signal module, just check for -15
                    is_sigterm = proc.returncode == -15
                
                if is_sigterm:
                    # Process was cancelled by user - this is expected behavior
                    output_queue.put("\n⚠️ Cache latents was cancelled\n")
                    if log_file:
                        log_file.write("\n⚠️ Cache latents was cancelled\n")
                        log_file.flush()
                    return ""  # Return empty output to indicate cancellation
                error_msg = f"\n❌ Cache latents failed with exit code {proc.returncode}\n"
                if proc.returncode < 0:
                    error_msg += f"Process was killed by signal {abs(proc.returncode)}\n"
                output_queue.put(error_msg)
                if log_file:
                    log_file.write(error_msg)
                    log_file.flush()
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        except KeyboardInterrupt:
            error_msg = "\n❌ Cache latents interrupted by user\n"
            output_queue.put(error_msg)
            if log_file:
                log_file.write(error_msg)
                log_file.flush()
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            raise
        except Exception as e:
            error_msg = f"\n❌ ERROR DURING CACHE LATENTS: {str(e)}\n"
            output_queue.put(error_msg)
            if log_file:
                log_file.write(error_msg)
                import traceback
                log_file.write(traceback.format_exc())
                log_file.flush()
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            raise
        finally:
            unregister_process(proc)
        return output
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
        result = proc.stdout + "\n" + proc.stderr
        if log_file:
            log_file.write(result)
            log_file.flush()
        return result

def run_cache_textencoder(dataset_config, text_encoder, batch_size, vram_profile, stream_output=False, log_file=None):
    cmd = [
        sys.executable, "src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--text_encoder", text_encoder,
        "--batch_size", str(batch_size)
    ]
    # Use fp8_vl for text encoder when GPU has 16GB or less VRAM (recommended per documentation)
    if str(vram_profile) in ["12", "16"]:
        cmd.append("--fp8_vl")
    if stream_output:
        # Create subprocess in new session so it survives SSH disconnect
        # start_new_session=True creates a new process group, preventing SIGHUP propagation
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=0, universal_newlines=True,
                                start_new_session=True if platform.system() != 'Windows' else False)
        register_process(proc)
        output = ""
        try:
            for line in iter(proc.stdout.readline, ''):
                if line:
                    # Normalize progress output: replace carriage return with newline
                    line = line.replace('\r\n', '\n').replace('\r', '\n')
                    output_queue.put(line)
                    if log_file:
                        log_file.write(line)
                        log_file.flush()
                    output += line
                    sys.stdout.flush()
            proc.wait()
            unregister_process(proc)  # Always unregister when process finishes
            # Check if process crashed or was killed
            if proc.returncode != 0:
                # SIGTERM (-15) means the process was cancelled gracefully - don't raise exception
                # On Windows, signal module may not have SIGTERM, so check for -15 directly
                try:
                    import signal
                    is_sigterm = proc.returncode == -signal.SIGTERM or proc.returncode == -15
                except (AttributeError, ImportError):
                    # Windows doesn't have SIGTERM in signal module, just check for -15
                    is_sigterm = proc.returncode == -15
                
                if is_sigterm:
                    # Process was cancelled by user - this is expected behavior
                    output_queue.put("\n⚠️ Cache text encoder was cancelled\n")
                    if log_file:
                        log_file.write("\n⚠️ Cache text encoder was cancelled\n")
                        log_file.flush()
                    return ""  # Return empty output to indicate cancellation
                error_msg = f"\n❌ Cache text encoder failed with exit code {proc.returncode}\n"
                if proc.returncode < 0:
                    error_msg += f"Process was killed by signal {abs(proc.returncode)}\n"
                output_queue.put(error_msg)
                if log_file:
                    log_file.write(error_msg)
                    log_file.flush()
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        except KeyboardInterrupt:
            error_msg = "\n❌ Cache text encoder interrupted by user\n"
            output_queue.put(error_msg)
            if log_file:
                log_file.write(error_msg)
                log_file.flush()
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            raise
        except Exception as e:
            error_msg = f"\n❌ ERROR DURING CACHE TEXT ENCODER: {str(e)}\n"
            output_queue.put(error_msg)
            if log_file:
                log_file.write(error_msg)
                import traceback
                log_file.write(traceback.format_exc())
                log_file.flush()
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            raise
        finally:
            unregister_process(proc)
        return output
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
        result = proc.stdout + "\n" + proc.stderr
        if log_file:
            log_file.write(result)
            log_file.flush()
        return result

@app.route("/autocaption", methods=["POST"])
def autocaption():
    """
    Auto-caption uploaded images using a lightweight captioning model, if available.
    If transformers / pillow are missing, return a JSON error
    with clear installation instructions instead of crashing the app.
    """
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400

    caption_model_choice = (request.form.get("caption_model") or "vit-gpt2").strip().lower()
    # Optional max length slider from GUI
    max_len_str = (request.form.get("caption_max_length") or "").strip()
    try:
        max_len = int(max_len_str) if max_len_str else 0
    except ValueError:
        max_len = 0
    # Clamp to a safe range; fall back to sensible default if out of range
    # Higher default for BLIP and Qwen-VL to allow more detailed captions
    if max_len < 32 or max_len > 512:  # Increased max for Qwen-VL
        if caption_model_choice == "qwen-vl":
            max_len = 256  # Qwen-VL can handle longer captions
        elif caption_model_choice == "blip-large":
            max_len = 160
        else:
            max_len = 128
    
    # Optional detail level slider from GUI (1-5, only for BLIP)
    detail_level_str = (request.form.get("caption_detail_level") or "").strip()
    try:
        detail_level = int(detail_level_str) if detail_level_str else 3
        detail_level = max(1, min(5, detail_level))  # Clamp to 1-5
    except ValueError:
        detail_level = 3

    try:
        if caption_model_choice == "qwen-vl":
            model, processor = load_qwen_vl_caption_model()
            model_type = "qwen-vl"
        elif caption_model_choice == "blip-large":
            model, processor = load_blip_caption_model()
            model_type = "blip"
        else:
            model, feature_extractor, tokenizer = load_vit_gpt2_caption_model()
            model_type = "vit-gpt2"
    except RuntimeError as e:
        return jsonify({
            "error": str(e),
            "missing_dependencies": True
        }), 500
    except Exception as e:
        return jsonify({"error": f"Failed to initialize caption model: {e}"}), 500

    try:
        from PIL import Image  # type: ignore
        import io
        import torch
    except ImportError:
        return jsonify({
            "error": (
                "Pillow dependency is missing.\n"
                "Please install it in your virtual environment:\n"
                "  pip install pillow"
            ),
            "missing_dependencies": True
        }), 500

    device = next(model.parameters()).device
    captions = []

    for fileobj in files:
        image_bytes = fileobj.read()
        if not image_bytes:
            captions.append("")
            continue

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            captions.append("")
            continue

        if model_type == "blip":
            # BLIP unconditional captioning (no text parameter)
            # Adjust parameters based on detail level (1-5)
            # Level 1: Basic, Level 5: Extremely Detailed
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            # Map detail level to generation parameters
            # Note: BLIP-large has inherent limitations - it may not describe very fine details
            # like specific lighting conditions or facial features in extreme detail
            detail_params = {
                1: {"num_beams": 6, "length_penalty": 1.1, "repetition_penalty": 1.1, "temperature": 0.8, "max_tokens_multiplier": 1.0},
                2: {"num_beams": 8, "length_penalty": 1.2, "repetition_penalty": 1.2, "temperature": 0.75, "max_tokens_multiplier": 1.1},
                3: {"num_beams": 10, "length_penalty": 1.5, "repetition_penalty": 1.3, "temperature": 0.7, "max_tokens_multiplier": 1.2},
                4: {"num_beams": 15, "length_penalty": 1.8, "repetition_penalty": 1.4, "temperature": 0.6, "max_tokens_multiplier": 1.3},
                5: {"num_beams": 20, "length_penalty": 2.5, "repetition_penalty": 1.6, "temperature": 0.5, "max_tokens_multiplier": 1.5},  # Extremely aggressive settings
            }
            params = detail_params.get(detail_level, detail_params[3])
            
            # Increase max tokens for higher detail levels to allow more detailed descriptions
            effective_max_tokens = int(max_len * params["max_tokens_multiplier"])
            effective_max_tokens = min(effective_max_tokens, 256)  # Cap at 256 to avoid memory issues
            
            with torch.no_grad():
                # Generate caption with detail-level-adjusted parameters
                # Higher detail levels use more aggressive parameters to extract maximum detail
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=effective_max_tokens,
                    num_beams=params["num_beams"],
                    no_repeat_ngram_size=2 if detail_level >= 4 else 3,  # Less restrictive for level 4-5
                    length_penalty=params["length_penalty"],
                    repetition_penalty=params["repetition_penalty"],
                    temperature=params["temperature"],
                    early_stopping=False,  # Always allow full length for maximum detail
                    do_sample=True if detail_level >= 4 else False,  # Enable sampling for highest levels
                )
            caption_raw = processor.decode(output_ids[0], skip_special_tokens=True).strip()
            caption = _normalize_caption_text(caption_raw)
        elif model_type == "qwen-vl":
            # Qwen-VL uses prompt-based captioning for highly detailed descriptions
            from musubi_tuner.caption_images_by_qwen_vl import DEFAULT_PROMPT, resize_image
            DEFAULT_MAX_SIZE = 1280
            IMAGE_FACTOR = 28
            
            # Use detailed prompt for comprehensive descriptions
            detailed_prompt = DEFAULT_PROMPT
            
            # Prepare messages for Qwen-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": detailed_prompt},
                    ],
                }
            ]
            
            # Prepare inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs = resize_image(image, max_size=DEFAULT_MAX_SIZE)
            inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(device)
            
            # Use autocast with appropriate dtype based on model dtype
            model_dtype = next(caption_qwen_vl_model.parameters()).dtype
            use_fp8 = model_dtype == torch.float8_e4m3fn
            
            if use_fp8:
                # fp8 mode - use bfloat16 for autocast during generation
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=max_len, 
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
            else:
                # bfloat16 mode
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=max_len, 
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
            
            # Extract only the generated part (remove input tokens)
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            caption_raw = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            caption = caption_raw[0] if caption_raw else ""
            caption = _normalize_caption_text(caption)
        else:
            pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                # Use a longer, more descriptive caption with stronger beam search
                output_ids = model.generate(
                    pixel_values,
                    max_length=max_len,
                    min_length=max(20, max_len // 4),
                    num_beams=6,
                    no_repeat_ngram_size=3,
                    length_penalty=1.2,
                    early_stopping=True,
                )
            caption_raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            caption = _normalize_caption_text(caption_raw)

        captions.append(caption)

    return jsonify({"captions": captions})

@app.route("/cancel_training", methods=["POST"])
def cancel_training():
    """
    Request cancellation of any active training / caching subprocesses.
    Also unloads models from VRAM/RAM to free memory.
    """
    cancel_active_processes()
    
    # Unload models to free VRAM/RAM after canceling
    unload_caption_models()
    import gc
    gc.collect()
    
    output_queue.put("\n" + "="*60 + "\n")
    output_queue.put("⏹ TRAINING CANCEL REQUEST RECEIVED. Attempting to stop processes...\n")
    output_queue.put("Unloading models from VRAM/RAM...\n")
    output_queue.put("="*60 + "\n")
    output_queue.put("[TRAINING_FINISHED]\n")
    return jsonify({"status": "ok", "message": "Training canceled. Models unloaded from memory."})

@app.route("/force_unload_vram", methods=["POST"])
def force_unload_vram():
    """
    Force unload all models from VRAM (caption models and clear CUDA cache).
    """
    try:
        unload_caption_models()
        import gc
        gc.collect()
        
        # Additional aggressive CUDA cache clearing
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Force garbage collection on CUDA tensors
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        return jsonify({"status": "ok", "message": "VRAM unloaded. Caption models removed and CUDA cache cleared."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error unloading VRAM: {str(e)}"}), 500

@app.route("/force_unload_ram", methods=["POST"])
def force_unload_ram():
    """
    Force unload models from RAM by unloading caption models and running aggressive cleanup.
    Attempts to free system-level cached RAM (e.g. from ComfyUI) by triggering OS cache cleanup.
    """
    try:
        unload_caption_models()
        import gc
        import subprocess
        
        # Aggressive Python-level cleanup
        # Run multiple GC passes to ensure cleanup
        for _ in range(5):
            gc.collect()
        
        # Try to free PyTorch pinned memory if available
        try:
            import torch
            if torch.cuda.is_available():
                # Clear CUDA pinned memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Try to free memory allocated in pinned memory pools
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        
        # Additional aggressive GC after PyTorch cleanup
        for _ in range(3):
            gc.collect()
        
        # Try to trigger OS-level cache cleanup (Linux only, requires appropriate permissions)
        # This can help free RAM cached by other processes like ComfyUI
        # Windows doesn't have this capability, so this only works on Linux/Unix
        cache_freed = False
        import platform
        if platform.system() == 'Linux':
            try:
                # Try to drop page cache, dentries, and inodes (requires root or appropriate capabilities)
                # echo 3 > /proc/sys/vm/drop_caches clears page cache, dentries, and inodes
                # This is safe but requires elevated permissions
                result = subprocess.run(
                    ['sh', '-c', 'sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null'],
                    capture_output=True,
                    timeout=5,
                    check=False  # Don't fail if we can't do this
                )
                if result.returncode == 0:
                    cache_freed = True
            except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError, Exception):
                # If we can't drop caches (no permissions, not Linux, etc.), that's okay
                pass
        
        message = "RAM unloaded. Caption models removed and aggressive garbage collection run."
        if cache_freed:
            message += " System cache cleared (may help free RAM from other processes like ComfyUI)."
        else:
            message += " System cache clear skipped (requires root permissions or not available on this system)."
        
        return jsonify({"status": "ok", "message": message})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error unloading RAM: {str(e)}"}), 500

@app.route("/list_samples")
def list_samples():
    output_dir = (request.args.get("output_dir") or "").strip()
    if not output_dir:
        return jsonify({"files": []})
    try:
        sample_dir = resolve_output_path(os.path.join(output_dir, "sample"))
    except ValueError:
        return jsonify({"files": []}), 400
    if not os.path.isdir(sample_dir):
        return jsonify({"files": []})
    entries = []
    try:
        filenames = sorted(
            [f for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, f))],
            key=lambda name: os.path.getmtime(os.path.join(sample_dir, name)),
            reverse=True,
        )
    except OSError:
        filenames = []
    allowed_ext = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".mp4",
        ".mov",
        ".webm",
    }
    for fname in filenames[:36]:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in allowed_ext:
            continue
        rel_path = f"{output_dir}/sample/{fname}".replace("//", "/")
        try:
            abs_path = resolve_output_path(rel_path)
        except ValueError:
            continue
        try:
            stat = os.stat(abs_path)
            mtime_display = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            mtime_display = ""
        entries.append(
            {
                "name": fname,
                "kind": "video" if ext in {".mp4", ".mov", ".webm"} else "image",
                "url": f"/sample_file?path={urllib.parse.quote(rel_path)}",
                "mtime_display": mtime_display,
            }
        )
    return jsonify({"files": entries})

@app.route("/sample_file")
def sample_file():
    rel_path = (request.args.get("path") or "").strip()
    if not rel_path:
        return "Missing path", 400
    try:
        abs_path = resolve_output_path(rel_path)
    except ValueError:
        return "Invalid path", 400
    if not os.path.isfile(abs_path):
        return "Not found", 404
    mime, _ = mimetypes.guess_type(abs_path)
    return send_file(abs_path, mimetype=mime or "application/octet-stream")

@app.route("/lora_readme")
def lora_readme():
    """Serve the LORA_README.md file for a given output directory."""
    output_dir = request.args.get("output_dir", "").strip()
    if not output_dir:
        return "Missing output_dir parameter", 400
    try:
        # Use resolve_output_path for security
        readme_rel_path = os.path.join(output_dir, "LORA_README.md").replace("\\", "/")
        abs_readme_path = resolve_output_path(readme_rel_path)
        if not os.path.exists(abs_readme_path):
            return "README file not found", 404
        return send_file(abs_readme_path, mimetype="text/markdown")
    except ValueError:
        return "Invalid path", 400
    except Exception as e:
        return f"Error reading README: {str(e)}", 500


@app.route("/latest_readme_link")
def latest_readme_link():
    """Return metadata for the most recently generated README (if available)."""
    global latest_readme_output_dir
    if latest_readme_output_dir:
        readme_rel_path = os.path.join(latest_readme_output_dir, "LORA_README.md")
        if os.path.exists(readme_rel_path):
            return jsonify({
                "available": True,
                "output_dir": latest_readme_output_dir,
            })
    return jsonify({"available": False})

@app.route("/current_log")
def current_log():
    log_path = current_log_path
    log_content = ""
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                tail_bytes = 200000  # ~200 KB tail
                f.seek(max(0, size - tail_bytes))
                log_content = f.read()
        except Exception as e:
            return jsonify({"error": f"Failed to read log: {e}", "training_active": is_training_active()}), 500

    return jsonify(
        {
            "log_path": log_path,
            "log_content": log_content,
            "training_active": is_training_active(),
        }
    )

@app.route("/uploads/<filename>")
def uploaded_image(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))

@app.route("/system_resources")
def system_resources():
    """Return current RAM and VRAM usage as JSON."""
    try:
        import psutil
        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / (1024**3)
        ram_used_gb = ram.used / (1024**3)
        ram_available_gb = ram.available / (1024**3)
        ram_percent = ram.percent
    except ImportError:
        ram_total_gb = ram_used_gb = ram_available_gb = ram_percent = None
    
    # Use nvidia-smi to get system-level VRAM usage (sees all processes, not just this one)
    # Note: nvidia-smi comes automatically with NVIDIA drivers - no separate installation needed
    # If not available, we fall back to PyTorch (only sees current process, not training subprocess)
    vram_total_gb = None
    vram_used_gb = None
    vram_available_gb = None
    vram_percent = None
    
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                vram_total_mb = float(parts[0].strip())
                vram_used_mb = float(parts[1].strip())
                vram_free_mb = float(parts[2].strip())
                vram_total_gb = vram_total_mb / 1024
                vram_used_gb = vram_used_mb / 1024
                vram_available_gb = vram_free_mb / 1024
                vram_percent = (vram_used_mb / vram_total_mb) * 100 if vram_total_mb > 0 else 0
    except FileNotFoundError:
        # nvidia-smi not found - user doesn't have NVIDIA GPU or NVIDIA drivers installed
        # Fallback to PyTorch if available (only sees current process, not training subprocess)
        try:
            import torch
            if torch.cuda.is_available():
                vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_used_gb = torch.cuda.memory_allocated(0) / (1024**3)
                vram_reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
                vram_available_gb = vram_total_gb - vram_reserved_gb
                vram_percent = (vram_reserved_gb / vram_total_gb) * 100 if vram_total_gb > 0 else 0
        except ImportError:
            # PyTorch not available either - VRAM monitoring not possible
            pass
    except (subprocess.TimeoutExpired, Exception):
        # nvidia-smi timed out or errored - try PyTorch as fallback
        try:
            import torch
            if torch.cuda.is_available():
                vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_used_gb = torch.cuda.memory_allocated(0) / (1024**3)
                vram_reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
                vram_available_gb = vram_total_gb - vram_reserved_gb
                vram_percent = (vram_reserved_gb / vram_total_gb) * 100 if vram_total_gb > 0 else 0
        except ImportError:
            # PyTorch not available either - VRAM monitoring not possible
            pass
    
    return jsonify({
        "ram": {
            "total_gb": round(ram_total_gb, 1) if ram_total_gb else None,
            "used_gb": round(ram_used_gb, 1) if ram_used_gb else None,
            "available_gb": round(ram_available_gb, 1) if ram_available_gb else None,
            "percent": round(ram_percent, 1) if ram_percent is not None else None,
        },
        "vram": {
            "total_gb": round(vram_total_gb, 1) if vram_total_gb else None,
            "used_gb": round(vram_used_gb, 1) if vram_used_gb else None,
            "available_gb": round(vram_available_gb, 1) if vram_available_gb else None,
            "percent": round(vram_percent, 1) if vram_percent is not None else None,
        }
    })

@app.route("/caption_model_status")
def caption_model_status():
    """Check if caption models are already loaded and if they're cached on disk."""
    caption_model_choice = request.args.get("model", "vit-gpt2").strip().lower()
    
    model_loaded = False
    model_cached = False
    
    if caption_model_choice == "qwen-vl":
        model_loaded = caption_qwen_vl_model is not None
        # Qwen-VL is a local file, check if it exists in any of the possible locations
        # Use EXACTLY the same paths and logic as in load_qwen_vl_caption_model()
        webgui_dir = os.path.dirname(os.path.abspath(__file__))
        model_filename = "qwen_2.5_vl_7b.safetensors"
        
        # Build list of possible paths (EXACTLY same as load_qwen_vl_caption_model)
        possible_paths = []
        
        # 1. User override via environment variable
        env_path = os.environ.get("CAPTION_MODEL_PATH_QWEN_VL", "")
        if env_path:
            possible_paths.append(env_path)
        
        # 2. Same directory as webgui.py (MOST COMMON - check this first!)
        webgui_path = os.path.join(webgui_dir, model_filename)
        possible_paths.append(webgui_path)
        
        # 3. Current working directory
        cwd_path = os.path.join(os.getcwd(), model_filename)
        possible_paths.append(cwd_path)
        
        # 4. Relative path from current directory
        possible_paths.append(model_filename)
        
        # Check all possible paths (EXACTLY same logic as load_qwen_vl_caption_model)
        for path in possible_paths:
            if not path:
                continue
            # Try both absolute and relative path
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and os.path.isfile(abs_path):
                model_cached = True
                break
            if os.path.exists(path) and os.path.isfile(path):
                model_cached = True
                break
    elif caption_model_choice == "blip-large":
        model_loaded = caption_blip_model is not None
        # Check if BLIP model is cached in Hugging Face cache
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(cache_dir):
                # Look for blip or salesforce in cache directory names
                for item in os.listdir(cache_dir):
                    item_lower = item.lower()
                    if "blip" in item_lower or "salesforce" in item_lower:
                        # Check if it's a valid cache entry (has snapshots subdirectory or model files)
                        item_path = os.path.join(cache_dir, item)
                        if os.path.isdir(item_path):
                            # Check for snapshots directory or model files
                            has_snapshots = False
                            has_model_files = False
                            for subitem in os.listdir(item_path):
                                subitem_path = os.path.join(item_path, subitem)
                                if subitem == "snapshots" and os.path.isdir(subitem_path):
                                    has_snapshots = True
                                    break
                                if os.path.isfile(subitem_path) and (subitem.endswith('.bin') or subitem.endswith('.safetensors') or subitem.endswith('.pt')):
                                    has_model_files = True
                                    break
                            if has_snapshots or has_model_files:
                                model_cached = True
                                break
        except:
            pass
    else:  # vit-gpt2
        model_loaded = caption_vit_model is not None
        # Check if ViT-GPT2 model is cached in Hugging Face cache
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(cache_dir):
                # Look for nlpconnect or vit-gpt2 in cache directory names
                for item in os.listdir(cache_dir):
                    item_lower = item.lower()
                    if ("vit" in item_lower and "gpt2" in item_lower) or "nlpconnect" in item_lower:
                        # Check if it's a valid cache entry (has snapshots subdirectory or model files)
                        item_path = os.path.join(cache_dir, item)
                        if os.path.isdir(item_path):
                            # Check for snapshots directory or model files
                            has_snapshots = False
                            has_model_files = False
                            for subitem in os.listdir(item_path):
                                subitem_path = os.path.join(item_path, subitem)
                                if subitem == "snapshots" and os.path.isdir(subitem_path):
                                    has_snapshots = True
                                    break
                                if os.path.isfile(subitem_path) and (subitem.endswith('.bin') or subitem.endswith('.safetensors') or subitem.endswith('.pt')):
                                    has_model_files = True
                                    break
                            if has_snapshots or has_model_files:
                                model_cached = True
                                break
        except:
            pass
    
    return jsonify({
        "model_loaded": model_loaded,
        "model_cached": model_cached,
        "model": caption_model_choice
    })

@app.route("/server_info")
def server_info():
    """Return server information (version, webserver type, system info, etc.)"""
    # Detect which webserver is available (same logic as server startup)
    webserver = "Unknown"
    try:
        import gevent
        webserver = "gevent"
    except ImportError:
        try:
            import waitress
            webserver = "waitress"
        except ImportError:
            webserver = "Flask (dev)"
    
    # Get OS/platform
    os_name = platform.system()
    os_release = platform.release() if hasattr(platform, 'release') else None
    
    # Get GPU info
    gpu_name = None
    cuda_version = None
    pytorch_version = None
    
    try:
        import torch
        pytorch_version = torch.__version__
        if torch.cuda.is_available():
            # Get GPU name from PyTorch
            gpu_name = torch.cuda.get_device_properties(0).name
            # Get CUDA version
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else None
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_name = "Apple Silicon (MPS)"
    except ImportError:
        pass
    
    # Try to get GPU name from nvidia-smi as fallback (more detailed info)
    if not gpu_name:
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]  # Get first GPU name
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
    
    return jsonify({
        "version": "0.2.5",
        "webserver": webserver,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": os_name,
        "os_release": os_release,
        "gpu": gpu_name,
        "cuda_version": cuda_version,
        "pytorch_version": pytorch_version
    })

@app.route("/musubitlx_gui_readme")
def musubitlx_gui_readme():
    # Serve the GUI README markdown file so it can be opened from the footer link
    readme_path = os.path.join(os.path.dirname(__file__), "MUSUBITLX_GUI.md")
    if os.path.exists(readme_path):
        return send_file(readme_path, mimetype="text/markdown")
    return "MUSUBITLX_GUI.md not found.", 404

@app.route("/stream")
def stream():
    def generate():
        start_time = time.time()
        max_connection_time = 120  # Close connection after 2 minutes (client will auto-reconnect)
        while True:
            # Close connection periodically to prevent Waitress thread exhaustion
            if time.time() - start_time > max_connection_time:
                yield f"data: [SSE_RECONNECT]\n\n"
                return
            try:
                # Use shorter timeout for more responsive updates
                line = output_queue.get(timeout=0.5)
                if line:  # Only send non-empty lines
                    # Handle carriage returns (\r) - progress bars use \r to overwrite the same line
                    # Convert \r to \n so each progress update appears on a new line
                    if '\r' in line:
                        # Replace \r with \n, but keep \n if it exists
                        if '\n' in line:
                            # Has both \r and \n, remove \r
                            line = line.replace('\r', '')
                        else:
                            # Only \r, replace with \n
                            line = line.replace('\r', '\n')
                    # Ensure line ends with newline for proper display
                    if not line.endswith('\n'):
                        line = line + '\n'
                    yield f"data: {line}\n\n"
            except queue.Empty:
                # Send heartbeat to keep connection alive
                yield f"data: \n\n"
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    })

@app.route("/", methods=["GET", "POST"])
def gui():
    global current_log_path
    download_msg = ""
    uploaded_gallery = []
    trigger = "subjectword"
    # Default: Assume user has 12GB VRAM GPU (safest default)
    # Settings are automatically adjusted based on user's selection:
    # - 12GB VRAM GPU → blocks_to_swap 45 (~12GB VRAM usage, safest, most offload to RAM)
    # - 16GB VRAM GPU → blocks_to_swap 30 (~16-18GB VRAM usage, balanced performance)
    # - 24GB+ VRAM GPU → blocks_to_swap 16 (~24GB VRAM usage, fastest, minimal offload)
    vram_profile = "12"
    user_output_dir = ""  # Just the subdirectory name, not the full path
    # Epochs do not affect memory usage, only total training time
    epochs = 6
    # Important for memory usage: batch size 1 is safest for 16 GB at 1024x1024
    batch_size = 1
    lr = "5e-5"
    # Default optimizer: AdamW
    optimizer_type = "AdamW"
    prompt = "portrait of a person"
    # 1024x1024 works but is heavy – keep batch_size=1 by default
    resolution = "1024x1024"
    seed = 42
    # Lower rank/dims reduce VRAM pressure
    rank = 16
    dims = 128
    image_repeats = 10
    advanced_flags = ""
    samples_enabled = False
    sample_prompt_text = ""
    sample_every_epochs = 1
    sample_every_steps = 0
    sample_at_first = False
    output_dir = ""  # Empty means use default "output" directory
    output_name = "lora"
    saved_yaml = ""
    output_txt = ""
    if request.method == "POST":
        action = request.form.get("action", "")
        if action == "download":
            label = request.form.get("dl_label")
            url = request.form.get("dl_url")
            fname = urllib.parse.unquote(url.split('/')[-1].split('?')[0])
            if os.path.exists(fname):
                download_msg = f"File already exists: {fname}"
            else:
                try:
                    download_msg = f"Downloading {label} as {fname} ..."
                    download_with_progress(url, fname)
                    download_msg = f"Downloaded: {fname}"
                except Exception as e:
                    download_msg = (f"Download failed: {e}\n"
                        "If you see HTTP 401 Unauthorized, please check that the file is public "
                        "or download manually from " + url)
        else:
            trigger = request.form.get("trigger", trigger)
            vram_profile = request.form.get("vram_profile", vram_profile)
            epochs = request.form.get("epochs", epochs)
            batch_size = request.form.get("batch_size", batch_size)
            lr = request.form.get("lr", lr)
            prompt = request.form.get("prompt", prompt)
            resolution = request.form.get("resolution", resolution)
            seed = request.form.get("seed", seed)
            rank = request.form.get("rank", rank)
            dims = request.form.get("dims", dims)
            optimizer_type = request.form.get("optimizer_type", "AdamW")
            output_name = request.form.get("output_name", output_name)
            advanced_flags = request.form.get("advanced_flags", advanced_flags)
            samples_enabled = request.form.get("samples_enabled") == "on"
            sample_prompt_text = request.form.get("sample_prompt_text", sample_prompt_text)
            sample_every_epochs = request.form.get("sample_every_epochs", sample_every_epochs)
            sample_every_steps = request.form.get("sample_every_steps", sample_every_steps)
            sample_at_first = request.form.get("sample_at_first") == "on"
            try:
                sample_every_epochs = int(sample_every_epochs) if str(sample_every_epochs).strip() else 0
            except (ValueError, TypeError):
                sample_every_epochs = 0
            try:
                sample_every_steps = int(sample_every_steps) if str(sample_every_steps).strip() else 0
            except (ValueError, TypeError):
                sample_every_steps = 0
            # Always use "output" as base directory, and create subdirectory if specified
            base_output_dir = "output"
            user_output_dir = request.form.get("output_dir", "").strip()
            if user_output_dir:
                # If user specified a directory, use it as subdirectory under "output"
                output_dir = os.path.join(base_output_dir, user_output_dir)
            else:
                # Default to "output" directory
                output_dir = base_output_dir
            ensure_dir(output_dir)
            output = os.path.join(output_dir, output_name + ".safetensors")

            # If user requested to load last YAML config, override fields from saved file
            if action == "loadyaml":
                yaml_path = "configs/last_config.yaml"
                if os.path.exists(yaml_path):
                    try:
                        with open(yaml_path, "r", encoding="utf-8") as f:
                            cfg = yaml.safe_load(f) or {}
                        trigger = cfg.get("trigger_word", trigger)
                        vram_profile = str(cfg.get("vram_profile", vram_profile))
                        epochs = cfg.get("epochs", epochs)
                        batch_size = cfg.get("batch_size", batch_size)
                        lr = cfg.get("learning_rate", lr)
                        optimizer_type = cfg.get("optimizer_type", optimizer_type)
                        prompt = cfg.get("prompt", prompt)
                        resolution = cfg.get("resolution", resolution)
                        seed = cfg.get("seed", seed)
                        rank = cfg.get("lora_rank", rank)
                        dims = cfg.get("lora_dims", dims)
                        output_dir = cfg.get("output_dir", output_dir)
                        # Extract just the subdirectory name from full path for display
                        if output_dir and output_dir.startswith("output/"):
                            user_output_dir = output_dir.replace("output/", "", 1)
                        elif output_dir == "output":
                            user_output_dir = ""
                        else:
                            user_output_dir = output_dir
                        output_name = cfg.get("output_name", output_name)
                        advanced_flags = cfg.get("advanced_flags", advanced_flags)
                        samples_enabled = bool(cfg.get("samples_enabled", samples_enabled))
                        sample_prompt_text = cfg.get("sample_prompt_text", sample_prompt_text)
                        sample_every_epochs = cfg.get("sample_every_epochs", sample_every_epochs)
                        sample_every_steps = cfg.get("sample_every_steps", sample_every_steps)
                        sample_at_first = bool(cfg.get("sample_at_first", sample_at_first))
                        try:
                            sample_every_epochs = int(sample_every_epochs)
                        except (ValueError, TypeError):
                            sample_every_epochs = 0
                        try:
                            sample_every_steps = int(sample_every_steps)
                        except (ValueError, TypeError):
                            sample_every_steps = 0
                        # Optional: image_repeats, if present
                        if "image_repeats" in cfg:
                            image_repeats = cfg.get("image_repeats", image_repeats)
                        saved_yaml = yaml.dump(cfg, allow_unicode=True)
                    except Exception as e:
                        output_txt = f"Failed to load YAML config: {e}"
                else:
                    output_txt = "No saved YAML config found at configs/last_config.yaml"
            files = request.files.getlist("images")
            # Filter out empty file objects
            files = [f for f in files if f.filename]
            # Get uploaded_count, handle empty string
            uploaded_count_str = request.form.get("uploaded_count", "0") or "0"
            try:
                count = int(uploaded_count_str)
            except (ValueError, TypeError):
                # Fallback to actual file count if uploaded_count is invalid
                count = len(files)
            captions = [request.form.get(f"caption_{i}", "") for i in range(count)]
            img_files = []
            cache_latents_output = ""
            cache_textencoder_output = ""
            dit_model = "qwen_image_bf16.safetensors"
            vae_model = "diffusion_pytorch_model.safetensors"  # VAE from Qwen/Qwen-Image, not ComfyUI
            text_encoder = "qwen_2.5_vl_7b.safetensors"
            # Generate dataset TOML with image repeats
            image_repeats = request.form.get("image_repeats", image_repeats)
            dataset_config = write_dataset_config_toml(output_dir, batch_size, resolution, image_repeats)
            sample_prompt_path = None

            config_dict = dict(
                trigger_word=trigger,
                vram_profile=vram_profile,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=lr,
                optimizer_type=optimizer_type,
                prompt=prompt,
                resolution=resolution,
                seed=int(seed),
                lora_rank=int(rank),
                lora_dims=int(dims),
                output_dir=output_dir,
                output_name=output_name,
                dit_model=dit_model,
                vae_model=vae_model,
                text_encoder=text_encoder,
                advanced_flags=str(advanced_flags or ""),
                samples_enabled=bool(samples_enabled),
                sample_prompt_text=sample_prompt_text,
                sample_every_epochs=sample_every_epochs,
                sample_every_steps=sample_every_steps,
                sample_at_first=bool(sample_at_first),
            )
            # Extra args are now fully controlled by the Advanced flags textarea (kept in sync with VRAM profile by JS)
            extra_args = []
            if action == "saveyaml":
                yaml_path = "configs/last_config.yaml"
                os.makedirs("configs", exist_ok=True)
                with open(yaml_path, 'w') as f:
                    yaml.dump(config_dict, f, allow_unicode=True)
                saved_yaml = yaml.dump(config_dict, allow_unicode=True)
            elif action == "train":
                # Check if images were uploaded
                if not files:
                    output_txt = "ERROR: No images were uploaded. Please select at least one image file before starting training."
                else:
                    # Save uploaded images first (include trigger word in caption files)
                    img_files = save_uploaded_images(files, captions, output_dir, trigger)
                    if not img_files:
                        output_txt = "ERROR: Failed to save uploaded images. Please check file permissions and try again."
                    else:
                        validation_error = ""
                        sample_prompt_path = None
                        if samples_enabled:
                            cleaned_prompt = (sample_prompt_text or "").strip()
                            if not cleaned_prompt:
                                validation_error = "ERROR: Sample previews are enabled but no sample prompts were provided."
                            else:
                                sample_prompt_path = os.path.join(output_dir, "sample_prompts.txt")
                                ensure_dir(output_dir)
                                normalized_text = cleaned_prompt.replace("\r\n", "\n")
                                with open(sample_prompt_path, "w", encoding="utf-8") as sp_file:
                                    sp_file.write(normalized_text.rstrip() + "\n")
                        if validation_error:
                            output_txt = validation_error
                        else:
                            # Show uploaded images in gallery
                            for idx, fname in enumerate(img_files):
                                cap = captions[idx] if idx < len(captions) else ""
                                uploaded_gallery.append((fname, cap))
                            # Clear output queue
                            while not output_queue.empty():
                                try:
                                    output_queue.get_nowait()
                                except queue.Empty:
                                    break
                        
                        # Create log file with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        log_filename = f"training_log_{output_name}_{timestamp}.log"
                        log_path = os.path.join(output_dir, log_filename)
                        ensure_dir(output_dir)
                        global current_log_path, latest_readme_output_dir
                        current_log_path = log_path
                        latest_readme_output_dir = None
                        
                        # Unload caption models to free VRAM before training
                        output_queue.put("[Unloading caption models from VRAM...]\n")
                        with open(log_path, "w", encoding="utf-8") as log_file:
                            log_file.write(f"Training log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            log_file.write(f"Output directory: {output_dir}\n")
                            log_file.write(f"Output name: {output_name}\n")
                            log_file.write("="*60 + "\n\n")
                            log_file.write("[Unloading caption models from VRAM...]\n")
                            log_file.flush()
                        unload_caption_models()
                        output_queue.put("Caption models unloaded. Starting training process...\n")
                        with open(log_path, "a", encoding="utf-8") as log_file:
                            log_file.write("Caption models unloaded successfully.\n")
                            log_file.flush()
                        time.sleep(0.2)  # Small delay to allow VRAM to be freed
                        
                        # 1. Cache latents
                        output_queue.put("[Latent cache]\n")
                        with open(log_path, "a", encoding="utf-8") as log_file:
                            log_file.write("\n[Latent cache]\n")
                            log_file.flush()
                            cache_latents_output = run_cache_latents(dataset_config, vae_model, stream_output=True, log_file=log_file)
                        # Check if cache was cancelled - if so, abort training
                        if cache_latents_output == "":
                            output_queue.put("\n❌ Training cancelled: Cache latents was interrupted\n")
                            output_queue.put("[TRAINING_FINISHED]\n")
                        else:
                            output_queue.put("\n")
                            time.sleep(0.1)  # Small delay to ensure output is sent
                            
                            # 2. Cache textencoder
                            output_queue.put("[Text encoder cache]\n")
                            with open(log_path, "a", encoding="utf-8") as log_file:
                                log_file.write("\n[Text encoder cache]\n")
                                log_file.flush()
                                cache_textencoder_output = run_cache_textencoder(dataset_config, text_encoder, batch_size, vram_profile, stream_output=True, log_file=log_file)
                            # Check if cache was cancelled - if so, abort training
                            if cache_textencoder_output == "":
                                output_queue.put("\n❌ Training cancelled: Cache text encoder was interrupted\n")
                                output_queue.put("[TRAINING_FINISHED]\n")
                            else:
                                output_queue.put("\n")
                                time.sleep(0.1)  # Small delay to ensure output is sent
                                
                                # 3. Train LoRA
                                output_queue.put("[Training]\n")
                                cmd = [
                                    sys.executable, "src/musubi_tuner/qwen_image_train_network.py",
                                    "--dit", dit_model,
                                    "--vae", vae_model,
                                    "--text_encoder", text_encoder,
                                    "--dataset_config", dataset_config,
                                    "--max_train_epochs", str(epochs),
                                    "--save_every_n_epochs", "1",
                                    "--learning_rate", lr,
                                    "--network_dim", str(rank),
                                    "--network_alpha", str(dims),
                                    "--seed", str(seed),
                                    "--output_dir", output_dir,
                                    "--output_name", output_name,
                                    "--network_module", "networks.lora_qwen_image",
                                    "--optimizer_type", optimizer_type,
                                    "--mixed_precision", "bf16",  # Recommended for Qwen-Image
                                    "--sdpa",  # Use PyTorch's scaled dot product attention (requires PyTorch 2.0)
                                    "--timestep_sampling", "qwen_shift",  # Recommended for Qwen-Image - optimizes timestep sampling for better results
                                    "--max_data_loader_n_workers", "2",  # Recommended for performance per documentation
                                    "--persistent_data_loader_workers"  # Recommended for performance per documentation
                                ] + extra_args
                                
                                # Add --fp8_vl for text encoder (recommended for <16GB VRAM per documentation)
                                # Since we're using safe settings for 12GB and 16GB VRAM GPUs, add fp8_vl for both
                                if str(vram_profile) in ["12", "16"]:
                                    cmd.append("--fp8_vl")

                                # Add sample flags if samples are enabled
                                if samples_enabled and sample_prompt_path and os.path.exists(sample_prompt_path):
                                    cmd.append("--sample_prompts")
                                    cmd.append(sample_prompt_path)
                                    if sample_every_epochs and int(sample_every_epochs) > 0:
                                        cmd.append("--sample_every_n_epochs")
                                        cmd.append(str(sample_every_epochs))
                                    if sample_every_steps and int(sample_every_steps) > 0:
                                        cmd.append("--sample_every_n_steps")
                                        cmd.append(str(sample_every_steps))
                                    if sample_at_first:
                                        cmd.append("--sample_at_first")
                                
                                # Append any manually specified advanced flags from the GUI
                                manual_flags = []
                                if advanced_flags:
                                    try:
                                        manual_flags = shlex.split(str(advanced_flags))
                                    except ValueError:
                                        # Fallback: simple whitespace split if user entered something odd
                                        manual_flags = str(advanced_flags).split()
                                if manual_flags:
                                    cmd += manual_flags
                                
                                # Log the full command that will be executed (including all flags)
                                full_cmd_str = ' '.join([f'"{arg}"' if ' ' in arg else arg for arg in cmd])
                                output_queue.put(f"\n[Full training command]\n{full_cmd_str}\n")
                                output_queue.put("="*60 + "\n\n")
                                
                                env = dict(os.environ)
                                env["CUDA_VISIBLE_DEVICES"] = "0"
                                # Improve CUDA memory handling to reduce fragmentation (helps OOM)
                                # Use PYTORCH_ALLOC_CONF instead of deprecated PYTORCH_CUDA_ALLOC_CONF
                                if "PYTORCH_CUDA_ALLOC_CONF" in env:
                                    # Migrate old setting to new format
                                    old_value = env.pop("PYTORCH_CUDA_ALLOC_CONF")
                                    env.setdefault("PYTORCH_ALLOC_CONF", old_value)
                                else:
                                    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
                                
                                def run_training():
                                    global latest_readme_output_dir
                                    try:
                                        with open(log_path, "a", encoding="utf-8") as log_file:
                                            log_file.write("\n[Training]\n")
                                            log_file.write(f"VRAM Profile: {vram_profile} GB\n")
                                            log_file.write(f"Advanced flags: {advanced_flags}\n")
                                            log_file.write(f"Full command:\n{full_cmd_str}\n")
                                            log_file.write("="*60 + "\n\n")
                                            log_file.flush()
                                            
                                            # Create subprocess in new session so it survives SSH disconnect
                                            # start_new_session=True creates a new process group, preventing SIGHUP propagation
                                            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                                   text=True, env=env, bufsize=0, universal_newlines=True,
                                                                   start_new_session=True if platform.system() != 'Windows' else False)
                                            register_process(proc)
                                            
                                            # RAM monitoring: check every 10 seconds
                                            last_ram_check_time = time.time()
                                            RAM_CHECK_INTERVAL = 10.0  # seconds
                                            RAM_THRESHOLD_PERCENT = 95.0
                                            ram_aborted = False
                                            
                                            try:
                                                for line in iter(proc.stdout.readline, ''):
                                                    # Check RAM usage every RAM_CHECK_INTERVAL seconds
                                                    current_time = time.time()
                                                    if current_time - last_ram_check_time >= RAM_CHECK_INTERVAL:
                                                        last_ram_check_time = current_time
                                                        ram_percent = get_ram_percent()
                                                        if ram_percent is not None and ram_percent > RAM_THRESHOLD_PERCENT:
                                                            # RAM exceeded threshold - abort training
                                                            ram_aborted = True
                                                            error_msg = f"\n" + "="*60 + "\n"
                                                            error_msg += f"⚠️  CRITICAL: RAM usage is {ram_percent:.1f}% (threshold: {RAM_THRESHOLD_PERCENT}%)\n"
                                                            error_msg += f"Training aborted to prevent system lockup.\n"
                                                            error_msg += f"Unloading caption models to free memory...\n"
                                                            error_msg += "="*60 + "\n"
                                                            output_queue.put(error_msg)
                                                            log_file.write(error_msg)
                                                            log_file.flush()
                                                            
                                                            # Unload caption models to free RAM
                                                            try:
                                                                unload_caption_models()
                                                                unload_msg = "Caption models unloaded.\n"
                                                                output_queue.put(unload_msg)
                                                                log_file.write(unload_msg)
                                                                log_file.flush()
                                                            except Exception as unload_err:
                                                                unload_err_msg = f"Warning: Error unloading models: {str(unload_err)}\n"
                                                                output_queue.put(unload_err_msg)
                                                                log_file.write(unload_err_msg)
                                                                log_file.flush()
                                                            
                                                            # Terminate training process
                                                            if proc.poll() is None:
                                                                proc.terminate()
                                                                # Wait up to 5 seconds for graceful termination
                                                                try:
                                                                    proc.wait(timeout=5)
                                                                except subprocess.TimeoutExpired:
                                                                    # Force kill if it doesn't terminate gracefully
                                                                    proc.kill()
                                                                    proc.wait()
                                                            
                                                            # Mark return code to indicate RAM abort (process is already terminated)
                                                            proc.returncode = -99
                                                            break  # Exit the loop
                                                    
                                                    if line:
                                                        # Normalize progress output: replace carriage return with newline
                                                        line = line.replace('\r\n', '\n').replace('\r', '\n')
                                                        # Send to queue immediately
                                                        output_queue.put(line)
                                                        # Write to log file
                                                        log_file.write(line)
                                                        log_file.flush()
                                                        # Force flush stdout if possible
                                                        sys.stdout.flush()
                                                
                                                # Only wait if process wasn't aborted by RAM monitoring
                                                if not ram_aborted:
                                                    proc.wait()
                                                
                                                # If RAM aborted, skip normal error handling (already handled above)
                                                if ram_aborted:
                                                    output_queue.put("\n" + "="*60 + "\n")
                                                    output_queue.put("❌ TRAINING ABORTED: RAM usage exceeded 95%\n")
                                                    output_queue.put("="*60 + "\n")
                                                    output_queue.put("All caption models have been unloaded from memory.\n")
                                                    log_file.write("\n" + "="*60 + "\n")
                                                    log_file.write("❌ TRAINING ABORTED: RAM usage exceeded 95%\n")
                                                    log_file.write("="*60 + "\n")
                                                    log_file.write("All caption models have been unloaded from memory.\n")
                                                    log_file.flush()
                                                # Check if process crashed (only if not RAM aborted)
                                                elif proc.returncode != 0:
                                                    error_msg = f"\n❌ Process exited with code {proc.returncode}\n"
                                                    output_queue.put(error_msg)
                                                    log_file.write(error_msg)
                                                    log_file.flush()
                                            except Exception as e:
                                                error_msg = f"\n❌ ERROR DURING TRAINING: {str(e)}\n"
                                                output_queue.put(error_msg)
                                                log_file.write(error_msg)
                                                import traceback
                                                log_file.write(traceback.format_exc())
                                                log_file.flush()
                                                if proc.poll() is None:
                                                    proc.terminate()
                                                    proc.wait()
                                                raise
                                            
                                            # Skip success message if RAM aborted
                                            if ram_aborted:
                                                # Already logged above, just mark as finished
                                                pass
                                            elif proc.returncode == 0:
                                                output_queue.put("\n" + "="*60 + "\n")
                                                output_queue.put("✅ TRAINING COMPLETED!\n")
                                                output_queue.put("="*60 + "\n")
                                                final_path = os.path.join(output_dir, output_name + ".safetensors")
                                                epoch_path = os.path.join(output_dir, output_name + "-000001.safetensors")
                                                output_queue.put(f"LoRA file has been saved to: {final_path}\n")
                                                if os.path.exists(epoch_path):
                                                    output_queue.put(f"And epoch files such as: {epoch_path}\n")
                                                output_queue.put(f"Absolute path: {os.path.abspath(final_path)}\n")
                                                output_queue.put(f"Log file: {log_path}\n")
                                                
                                                # Generate LORA_README.md
                                                readme_path = None
                                                try:
                                                    # Ensure all required variables have default values
                                                    # Access variables from outer scope and provide defaults
                                                    try:
                                                        trigger_val = trigger or output_name
                                                    except NameError:
                                                        trigger_val = output_name
                                                    try:
                                                        epochs_val = int(epochs) if epochs else 6
                                                    except (NameError, TypeError, ValueError):
                                                        epochs_val = 6
                                                    try:
                                                        batch_size_val = int(batch_size) if batch_size else 1
                                                    except (NameError, TypeError, ValueError):
                                                        batch_size_val = 1
                                                    try:
                                                        image_repeats_val = int(image_repeats) if image_repeats else 1
                                                    except (NameError, TypeError, ValueError):
                                                        image_repeats_val = 1
                                                    try:
                                                        lr_val = lr or "5e-5"
                                                    except NameError:
                                                        lr_val = "5e-5"
                                                    try:
                                                        rank_val = int(rank) if rank else 16
                                                    except (NameError, TypeError, ValueError):
                                                        rank_val = 16
                                                    try:
                                                        dims_val = int(dims) if dims else 128
                                                    except (NameError, TypeError, ValueError):
                                                        dims_val = 128
                                                    try:
                                                        resolution_val = resolution or "1024x1024"
                                                    except NameError:
                                                        resolution_val = "1024x1024"
                                                    try:
                                                        prompt_val = prompt or ""
                                                    except NameError:
                                                        prompt_val = ""
                                                    try:
                                                        sample_prompt_text_val = sample_prompt_text if sample_prompt_text else None
                                                    except NameError:
                                                        sample_prompt_text_val = None
                                                    
                                                    # Count images in output directory (more robust than relying on img_files variable)
                                                    num_images = 0
                                                    if os.path.exists(output_dir):
                                                        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
                                                        for f in os.listdir(output_dir):
                                                            # Skip sample directory and other non-image files
                                                            full_path = os.path.join(output_dir, f)
                                                            if os.path.isfile(full_path) and any(f.lower().endswith(ext) for ext in image_extensions):
                                                                num_images += 1
                                                    
                                                    if num_images == 0:
                                                        output_queue.put("⚠️ Warning: No images found in output directory for README generation\n")
                                                    
                                                    readme_path = generate_lora_readme(
                                                        output_dir, output_name, trigger_val, num_images,
                                                        epochs_val, batch_size_val, image_repeats_val, lr_val,
                                                        rank_val, dims_val, resolution_val, prompt_val, sample_prompt_text_val
                                                    )
                                                    if readme_path and os.path.exists(readme_path):
                                                        output_queue.put(f"📄 LoRA README generated: {readme_path}\n")
                                                        output_queue.put("   (View README.md link will appear below when training completes)\n")
                                                        latest_readme_output_dir = output_dir
                                                    else:
                                                        output_queue.put(f"⚠️ Warning: README generation failed (file not created at {readme_path if readme_path else 'unknown path'})\n")
                                                        latest_readme_output_dir = None
                                                except Exception as e:
                                                    # Log error but don't fail training if README generation fails
                                                    import traceback
                                                    error_details = traceback.format_exc()
                                                    error_msg = f"⚠️ Warning: README generation failed with error: {str(e)}\n"
                                                    output_queue.put(error_msg)
                                                    try:
                                                        log_file.write(error_msg)
                                                        log_file.write(f"Traceback: {error_details}\n")
                                                    except:
                                                        pass
                                                
                                                output_queue.put("="*60 + "\n")
                                                
                                                log_file.write("\n" + "="*60 + "\n")
                                                log_file.write("✅ TRAINING COMPLETED!\n")
                                                log_file.write("="*60 + "\n")
                                                log_file.write(f"LoRA file has been saved to: {final_path}\n")
                                                if os.path.exists(epoch_path):
                                                    log_file.write(f"And epoch files such as: {epoch_path}\n")
                                                log_file.write(f"Absolute path: {os.path.abspath(final_path)}\n")
                                                try:
                                                    if readme_path and os.path.exists(readme_path):
                                                        log_file.write(f"LoRA README generated: {readme_path}\n")
                                                except:
                                                    pass
                                                log_file.write("="*60 + "\n")
                                            elif not ram_aborted:  # Don't show error message if RAM aborted (already handled above)
                                                # Check if process was cancelled (SIGTERM = -15)
                                                try:
                                                    is_sigterm = proc.returncode == -signal.SIGTERM or proc.returncode == -15
                                                except (AttributeError, ImportError):
                                                    # Windows doesn't have SIGTERM in signal module, just check for -15
                                                    is_sigterm = proc.returncode == -15
                                                
                                                if is_sigterm:
                                                    # Process was cancelled by user - this is expected behavior
                                                    output_queue.put("\n" + "="*60 + "\n")
                                                    output_queue.put("⏹ TRAINING CANCELLED BY USER\n")
                                                    output_queue.put("="*60 + "\n")
                                                    
                                                    log_file.write("\n" + "="*60 + "\n")
                                                    log_file.write("⏹ TRAINING CANCELLED BY USER\n")
                                                    log_file.write("="*60 + "\n")
                                                else:
                                                    # Actual error occurred
                                                    output_queue.put("\n" + "="*60 + "\n")
                                                    output_queue.put(f"❌ TRAINING ABORTED WITH ERROR (exit code {proc.returncode})\n")
                                                    output_queue.put("="*60 + "\n")
                                                    output_queue.put(f"See log file for details: {log_path}\n")
                                                    
                                                    log_file.write("\n" + "="*60 + "\n")
                                                    log_file.write(f"❌ TRAINING ABORTED WITH ERROR (exit code {proc.returncode})\n")
                                                    log_file.write("="*60 + "\n")
                                            
                                            log_file.write(f"\nTraining finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                            if ram_aborted:
                                                log_file.write("Reason: RAM usage exceeded 95% threshold.\n")
                                            log_file.flush()
                                            unregister_process(proc)
                                        output_queue.put("[TRAINING_FINISHED]\n")  # Signal to close connection
                                    except Exception as e:
                                        output_queue.put("\n" + "="*60 + "\n")
                                        output_queue.put(f"❌ ERROR: {str(e)}\n")
                                        output_queue.put("="*60 + "\n")
                                        output_queue.put(f"See log file for details: {log_path}\n")
                                        try:
                                            with open(log_path, "a", encoding="utf-8") as log_f2:
                                                log_f2.write("\n" + "="*60 + "\n")
                                                log_f2.write(f"❌ ERROR: {str(e)}\n")
                                                log_f2.write("="*60 + "\n")
                                                import traceback
                                                log_f2.write(traceback.format_exc())
                                        except:
                                            pass
                                        output_queue.put("[TRAINING_FINISHED]\n")
                                
                                # Run training in background thread
                                training_thread = threading.Thread(target=run_training, daemon=True)
                                training_thread.start()
                                
                                # Return immediately to show streaming output
                                output_txt = ""  # Will be streamed via SSE
            else:
                # For non-train actions, still show uploaded images if any
                if files:
                    img_files = save_uploaded_images(files, captions, output_dir, trigger)
                    for idx, fname in enumerate(img_files):
                        cap = captions[idx] if idx < len(captions) else ""
                        uploaded_gallery.append((fname, cap))
    return render_template_string(
        HTML,
        downloads=DOWNLOADS, download_msg=download_msg,
        trigger=trigger, vram_profile=vram_profile, epochs=epochs, batch_size=batch_size, lr=lr,
        prompt=prompt, resolution=resolution, seed=seed, rank=rank, dims=dims,
        output_dir=output_dir, user_output_dir=user_output_dir, output_name=output_name, image_repeats=image_repeats,
        advanced_flags=advanced_flags,
        samples_enabled=samples_enabled, sample_prompt_text=sample_prompt_text,
        sample_every_epochs=sample_every_epochs, sample_every_steps=sample_every_steps, sample_at_first=sample_at_first,
        saved_yaml=saved_yaml, output_txt=output_txt,
        uploaded_gallery=uploaded_gallery, optimizer_type=optimizer_type
    )

if __name__ == "__main__":
    # Ignore SIGHUP signal to prevent the server from being killed when SSH disconnects
    # This allows training to continue even if the SSH connection is lost
    def ignore_sighup(signum, frame):
        print("Received SIGHUP (SSH disconnect). Ignoring to keep server running...")
        pass
    
    # Only set signal handler on Unix-like systems (Linux, macOS)
    # Windows doesn't use SIGHUP in the same way
    if hasattr(signal, 'SIGHUP'):
        try:
            signal.signal(signal.SIGHUP, ignore_sighup)
            print("SIGHUP handler installed. Server will continue running if SSH disconnects.")
        except (ValueError, OSError) as e:
            print(f"Warning: Could not set SIGHUP handler: {e}")
    
    # Server selection for production use with SSE (Server-Sent Events) support
    # Priority: gevent (best for SSE) > Waitress (with more threads) > Flask dev server
    print("Starting MusubiTLX server...")
    print("Note: To run in background, use: nohup ./start_gui.sh > webgui.log 2>&1 &")
    print("Or use screen/tmux for better session management.")
    
    server_started = False
    
    # Try gevent first - best for SSE streaming (async-friendly)
    if not server_started:
        try:
            from gevent import monkey
            monkey.patch_all()
            from gevent.pywsgi import WSGIServer
            print("Using gevent WSGIServer (recommended for SSE streaming)")
            http_server = WSGIServer(('0.0.0.0', 5000), app, log=None)
            server_started = True
            http_server.serve_forever()
        except ImportError:
            print("gevent not installed. Install with: pip install gevent")
    
    # Fallback to Waitress with more threads for SSE
    if not server_started:
        try:
            from waitress import serve
            print("Using Waitress WSGI server (install gevent for better SSE: pip install gevent)")
            # Use many more threads to handle SSE connections without blocking
            # asyncore_use_poll=True helps with many concurrent connections
            serve(app, host="0.0.0.0", port=5000, threads=32, 
                  channel_timeout=300, asyncore_use_poll=True)
            server_started = True
        except ImportError:
            print("waitress not installed.")
    
    # Last resort: Flask development server
    if not server_started:
        print("WARNING: Using Flask development server (not recommended for production)")
        print("Install a production server: pip install gevent")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)