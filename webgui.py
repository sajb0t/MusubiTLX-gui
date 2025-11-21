from flask import Flask, render_template_string, request, send_file, Response, stream_with_context
import subprocess, os, yaml, urllib.request, urllib.parse
from werkzeug.utils import secure_filename
import toml
import threading
import queue
import time
from datetime import datetime

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Global queue for streaming output
output_queue = queue.Queue()

DOWNLOADS = {
    "DiT Model": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors?download=true",
    "Text Encoder": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors?download=true",
    "VAE Model": "https://huggingface.co/Qwen/Qwen-Image/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true",
}

def download_with_progress(url, fname):
    import sys
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

HTML = '''
<!DOCTYPE html>
<html>
<head>
  <title>MusubiTLX – Qwen LoRA Training Panel</title>
  <style>
    body { font-family: 'Inter', Arial, sans-serif; background: #222; color: #eee; margin:0;}
    .container { background: #2b2f33; box-shadow: 0 0 18px rgba(0,0,0,0.75); padding: 2.2em; max-width: 1100px; margin: 2.2em auto; border-radius: 22px;}
    .section { background: #23272E; border-radius:16px; margin-bottom: 24px; padding: 1.25em 1.7em;}
    .section h3 { margin-top:0; color: #86c6fe; font-size: 1.28em; font-weight:600; letter-spacing:0.02em;}
    input, select, textarea { width: 96%; padding: 8px; margin: 6px 0 16px; border-radius: 6px; border: 1px solid #666; background: #1e2023; color: #eee; font-size: 1.09rem;}
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
    .imgbox { background:#151618; border-radius:13px; box-shadow:0 0 8px #444 inset; padding:13px; text-align:center; width:190px;}
    .imgbox img { max-width:170px; max-height:170px; border-radius:7px; box-shadow:0 0 7px #262f2a;}
    .imgbox input[type=text] { width:97%; margin-top:10px; background:#222; color:#eee; border-radius:6px; border: 1px solid #666; }
    .title { font-size:2.1em; color:#a1e6ff; letter-spacing:1px; margin-bottom:0.3em; text-align:left;}
    .model-list { margin-bottom: 15px; }
    .model-list form { display: inline; }
    .model-list button { margin-right: 8px; }
    .footer { margin-top: 24px; font-size: 0.9em; color:#999; text-align:right; }
    .footer a { color:#7abfff; text-decoration:none; margin-left:12px; }
    .footer a:hover { text-decoration:underline; }
  </style>
</head>
<body>
  <div class="container">
    <div class="title">MusubiTLX – Qwen LoRA Training</div>
    <div class="section model-list">
      <h3>Model files (download from Hugging Face)</h3>
      {% for label, url in downloads.items() %}
        <form method="POST">
          <input type="hidden" name="dl_label" value="{{label}}">
          <input type="hidden" name="dl_url" value="{{url}}">
          <button type="submit" name="action" value="download">{{label}}</button>
        </form>
      {% endfor %}
      {% if download_msg %}<div style="margin-top:10px;">{{download_msg}}</div>{% endif %}
    </div>
    <form id="training-form" method="POST" enctype="multipart/form-data">
      <div class="section">
        <h3>1. Upload training images</h3>
        <label>Select images (multiple allowed)</label>
        <input type="file" id="images" name="images" multiple accept="image/*" onchange="previewImages(event)">
        <div class="imglist imglist-empty" id="preview">
          <div class="drop-hint">
            Drag &amp; drop images here or use the file picker above.
          </div>
        </div>
        <input type="hidden" name="uploaded_count" id="uploaded_count">
      </div>
      <div class="section">
        <h3>2. Training settings</h3>
        <div class="form-grid">
          <div class="field-group">
            <label>Trigger word (label used in captions)</label>
            <input name="trigger" type="text" value="{{trigger}}">
          </div>
          <div class="field-group field-group-half">
            <label>VRAM profile (GPU memory)</label>
            <select name="vram_profile" onchange="updateEstimate()">
              <option value="12" {% if vram_profile == '12' %}selected{% endif %}>12 GB – safe / recommended (most offload, slowest)</option>
              <option value="16" {% if vram_profile == '16' %}selected{% endif %}>16 GB – higher VRAM use (faster, may OOM)</option>
              <option value="24" {% if vram_profile == '24' %}selected{% endif %}>24+ GB – performance (fastest)</option>
            </select>
          </div>
          <div class="field-group field-group-half">
            <label>Epochs (how many passes over the dataset)</label>
            <input name="epochs" type="number" value="{{epochs}}" oninput="updateEstimate()">
          </div>
          <div class="field-group field-group-half">
            <label>Batch size</label>
            <input name="batch_size" type="number" value="{{batch_size}}" oninput="updateEstimate()">
          </div>
          <div class="field-group field-group-half">
            <label>Image repeats (how many times each image is seen)</label>
            <input name="image_repeats" type="number" value="{{image_repeats}}" oninput="updateEstimate()">
          </div>
          <div class="field-group field-group-half">
            <label>Learning rate</label>
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
            <input name="resolution" type="text" value="{{resolution}}" oninput="updateEstimate()">
          </div>
          <div class="field-group field-group-half">
            <label>Seed (same seed = repeatable result)</label>
            <input name="seed" type="number" value="{{seed}}">
          </div>
          <div class="field-group field-group-half">
            <label>LoRA rank (lower = less VRAM)</label>
            <input name="rank" type="number" value="{{rank}}">
          </div>
          <div class="field-group field-group-half">
            <label>LoRA dims (lower = less VRAM)</label>
            <input name="dims" type="number" value="{{dims}}">
          </div>
          <div class="field-group field-group-wide">
            <label>Output folder (under <code>output/</code>)</label>
            <input name="output_dir" type="text" value="{{output_dir}}" placeholder="empty = use 'output'">
            <small style="color: #888; display: block; margin-top: -8px; margin-bottom: 6px;">If you enter <code>art</code> files will be saved in <code>output/art/</code></small>
          </div>
          <div class="field-group field-group-wide">
            <label>LoRA filename (.safetensors)</label>
            <input name="output_name" type="text" value="{{output_name}}" placeholder="lora" required>
            <small style="color: #888; display: block; margin-top: -8px; margin-bottom: 4px;">Final LoRA file: <code>output/{{output_dir}}/{{output_name}}.safetensors</code></small>
          </div>
        </div>
        <div style="color:#aaa; font-size:0.9em; margin-bottom:6px; margin-top:4px;">
          <span id="time-estimate">Estimated training time: add images and adjust epochs/batch size.</span>
        </div>
        <div class="button-row">
          <button type="button" id="start-btn" onclick="startTraining()">Start training</button>
          <button type="submit" id="saveyaml-btn" name="action" value="saveyaml">Save configuration as YAML</button>
          <button type="submit" id="loadyaml-btn" name="action" value="loadyaml">Load last YAML config</button>
        </div>
      </div>
    </form>
    {% if uploaded_gallery %}
      <div class="section">
        <h3>3. Preview images & captions</h3>
        <div class="gallery">
        {% for fname, cap in uploaded_gallery %}
          <div class="imgbox">
            <img src="{{url_for('uploaded_image', filename=fname)}}">
            <div style="word-break:break-word; font-size:1.05em; margin-top:8px;">{{cap}}</div>
          </div>
        {% endfor %}
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
    {% if saved_yaml %}
      <div class="yaml-preview">
        <strong>Saved YAML configuration:</strong><br>
        <pre>{{saved_yaml}}</pre>
      </div>
    {% endif %}
    <div id="output-container" style="display:none;">
      <div class="output">
        <strong>Training log:</strong><br>
        <pre id="output-text" style="max-height: 500px; overflow-y: auto;"></pre>
      </div>
    </div>
    <div class="footer">
      MusubiTLX Web GUI created by TLX
      <a href="/musubitlx_gui_readme" target="_blank">GUI README</a>
    </div>
  </div>
  <script>
    function showSpinner() { 
      document.getElementById("spinner").style.display = "block";
      document.getElementById("output-container").style.display = "block";
      document.getElementById("output-text").textContent = "";
      const startBtn = document.getElementById("start-btn");
      const saveYamlBtn = document.getElementById("saveyaml-btn");
      const loadYamlBtn = document.getElementById("loadyaml-btn");
      if (startBtn) startBtn.disabled = true;
      if (saveYamlBtn) saveYamlBtn.disabled = true;
      if (loadYamlBtn) loadYamlBtn.disabled = true;
    }
    
    function updateEstimate() {
        const uploaded = parseInt(document.getElementById('uploaded_count').value || "0");
        const epochsInput = document.querySelector('input[name="epochs"]');
        const batchInput = document.querySelector('input[name="batch_size"]');
        const repeatsInput = document.querySelector('input[name="image_repeats"]');
        const vramSelect = document.querySelector('select[name="vram_profile"]');
        const resInput = document.querySelector('input[name="resolution"]');
        const estElem = document.getElementById('time-estimate');
        if (!epochsInput || !batchInput || !vramSelect || !resInput || !estElem) return;
        
        const epochs = parseInt(epochsInput.value || "0");
        const batchSize = parseInt(batchInput.value || "1");
        const repeats = parseInt(repeatsInput ? (repeatsInput.value || "1") : "1");
        const vramProfile = vramSelect.value || "16";
        const resText = resInput.value || "1024x1024";
        
        if (!uploaded || !epochs || !batchSize) {
            estElem.textContent = "Estimated training time: add images and set epochs/batch size.";
            return;
        }
        
        const effectiveImages = uploaded * (repeats > 0 ? repeats : 1);
        let stepsPerEpoch = Math.ceil(effectiveImages / batchSize);
        let totalSteps = stepsPerEpoch * epochs;
        
        // Parse resolution to scale estimate by pixel count
        let w = 1024, h = 1024;
        const m = resText.match(/(\\d+)\\s*x\\s*(\\d+)/i);
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
            text = `Estimated training time: ~${totalSec.toFixed(0)} seconds (${totalSteps} steps).`;
        } else if (totalSec < 3600) {
            const min = Math.round(totalSec / 60);
            text = `Estimated training time: ~${min} minutes (${totalSteps} steps).`;
        } else {
            const hours = (totalSec / 3600).toFixed(1);
            text = `Estimated training time: ~${hours} hours (${totalSteps} steps).`;
        }
        estElem.textContent = text;
    }
    function previewImages(event) {
        var preview = document.getElementById('preview');
        preview.innerHTML = '';
        var files = event.target && event.target.files ? event.target.files : (event.files || []);
        document.getElementById('uploaded_count').value = files.length;
        
        for (let i = 0; i < files.length; i++) {
            let f = files[i];
            let reader = new FileReader();

            // Create container and caption in the correct order immediately
            var div = document.createElement('div');
            div.className = "imgbox";
            div.innerHTML = '<img><br>' +
                '<input type="text" name="caption_' + i + '" placeholder="Caption for image ' + (i + 1) + '" style="width:97%; margin-top:10px; background:#222; color:#eee; border-radius:6px; border: 1px solid #666;">';
            let imgEl = div.querySelector('img');
            preview.appendChild(div);

            // When the file is loaded, just set the image source
            reader.onload = function(e) {
                if (imgEl) {
                    imgEl.src = e.target.result;
                }
            }
            reader.readAsDataURL(f);
        }
        updateEstimate();
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
          const dt = new DataTransfer();
          Array.from(fileInput.files || []).forEach(f => dt.items.add(f));
          imageFiles.forEach(f => dt.items.add(f));
          fileInput.files = dt.files;
          previewImages({ target: fileInput });
        });
    }
    
    // Live output updates via Server-Sent Events
    let eventSource = null;
    function startTraining() {
      showSpinner();
      
      // 1. Connect to stream immediately
      if (eventSource) eventSource.close();
      eventSource = new EventSource('/stream');
      
      eventSource.onmessage = function(event) {
        const outputText = document.getElementById("output-text");
        const data = event.data;
        
        // Skip empty messages
        if (!data || data.trim() === "") {
          return;
        }
        
        // If training aborted with error (including OOM), stop spinner and re‑enable buttons
        if (data.includes("TRAINING ABORTED WITH ERROR") || data.includes("torch.OutOfMemoryError")) {
          document.getElementById("spinner").style.display = "none";
          if (eventSource) {
            eventSource.close();
            eventSource = null;
          }
          const startBtn = document.getElementById("start-btn");
          const saveYamlBtn = document.getElementById("saveyaml-btn");
          const loadYamlBtn = document.getElementById("loadyaml-btn");
          if (startBtn) startBtn.disabled = false;
          if (saveYamlBtn) saveYamlBtn.disabled = false;
          if (loadYamlBtn) loadYamlBtn.disabled = false;
          const outputContainer = document.getElementById("output-container");
          outputContainer.style.border = "3px solid #ff5555";
          outputContainer.style.borderRadius = "10px";
          // fall through to still append the error line to the log
        }
        
        // Check if training is finished (success or handled error)
        if (data.includes("[TRAINING_FINISHED]")) {
          document.getElementById("spinner").style.display = "none";
          eventSource.close();
          eventSource = null;
          // Re‑enable buttons after training
          const startBtn = document.getElementById("start-btn");
          const saveYamlBtn = document.getElementById("saveyaml-btn");
          const loadYamlBtn = document.getElementById("loadyaml-btn");
          if (startBtn) startBtn.disabled = false;
          if (saveYamlBtn) saveYamlBtn.disabled = false;
          if (loadYamlBtn) loadYamlBtn.disabled = false;
          // Show completion highlight
          const outputContainer = document.getElementById("output-container");
          outputContainer.style.border = "3px solid #00ff00";
          outputContainer.style.borderRadius = "10px";
          return;
        }
        
        // Append each SSE message on its own line for readability
        const processedData = data;
        outputText.textContent += processedData + "\\n";
        outputText.scrollTop = outputText.scrollHeight;
      };
      
      eventSource.onerror = function(event) {
        if (eventSource.readyState === EventSource.CLOSED) {
             // Connection closed normally
        } else {
             // Error occurred
             console.error("EventSource error:", event);
             // Don't close immediately, it might reconnect
        }
      };

      // 2. Submit form via AJAX to prevent page reload
      var form = document.getElementById('training-form');
      var formData = new FormData(form);
      formData.append('action', 'train'); // Manually add action since we are not using submit button
      
      fetch('/', {
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (!response.ok) {
            document.getElementById("output-text").textContent += "❌ Server error when starting training: " + response.statusText + "\\n";
        }
      })
      .catch(error => {
        document.getElementById("output-text").textContent += "❌ Network error when starting training: " + error + "\\n";
      });
    }
    
    // Auto-start streaming when page loads with training in progress
    window.addEventListener('load', function() {
      const spinner = document.getElementById("spinner");
      if (spinner && spinner.style.display === "block") {
        startTraining();
      }
      setupDropzone();
    });
  </script>
</body>
</html>
'''

def ensure_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d)

def save_uploaded_images(files, captions, output_dir):
    ensure_dir(output_dir)
    img_files = []
    for idx, fileobj in enumerate(files):
        fname = secure_filename(fileobj.filename)
        filepath = os.path.join(output_dir, fname)
        fileobj.save(filepath)
        caption_file = os.path.join(output_dir, os.path.splitext(fname)[0] + ".txt")
        cap = captions[idx] if idx < len(captions) else ""
        with open(caption_file, "w", encoding="utf-8") as cf:
            cf.write(cap.strip())
        img_files.append(fname)
    return img_files

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
        "python", "src/musubi_tuner/qwen_image_cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_model
    ]
    if stream_output:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                text=True, bufsize=0, universal_newlines=True)
        output = ""
        import sys
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
            # Check if process crashed or was killed
            if proc.returncode != 0:
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
            error_msg = f"\n❌ FEL VID CACHE LATENTS: {str(e)}\n"
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
        "python", "src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--text_encoder", text_encoder,
        "--batch_size", str(batch_size)
    ]
    if int(str(vram_profile)) <= 12 or str(vram_profile) == "12":
        cmd.append("--fp8_vl")
    if stream_output:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                text=True, bufsize=0, universal_newlines=True)
        output = ""
        import sys
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
            # Check if process crashed or was killed
            if proc.returncode != 0:
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
            error_msg = f"\n❌ FEL VID CACHE TEXT ENCODER: {str(e)}\n"
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
        return output
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
        result = proc.stdout + "\n" + proc.stderr
        if log_file:
            log_file.write(result)
            log_file.flush()
        return result

@app.route("/uploads/<filename>")
def uploaded_image(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))

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
        while True:
            try:
                line = output_queue.get(timeout=1)
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
                yield f"data: \n\n"  # Keep connection alive
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    })

@app.route("/", methods=["GET", "POST"])
def gui():
    download_msg = ""
    uploaded_gallery = []
    trigger = "subjectword"
    # Default profile tuned for ~12 GB VRAM GPUs (safer for many 16 GB cards)
    vram_profile = "12"
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
                        output_name = cfg.get("output_name", output_name)
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
                text_encoder=text_encoder
            )
            extra_args = []
            if vram_profile == "12":
                # Aggressive VRAM saving: fp8, large block swap, gradient checkpointing + CPU offload of activations
                extra_args += [
                    "--fp8_base", "--fp8_scaled",
                    "--blocks_to_swap", "45",
                    "--gradient_checkpointing",
                    "--gradient_checkpointing_cpu_offload",
                ]
            elif vram_profile == "16":
                # Standard VRAM saving: fp8, some block swap, gradient checkpointing (no CPU offload)
                extra_args += [
                    "--fp8_base", "--fp8_scaled",
                    "--blocks_to_swap", "16",
                    "--gradient_checkpointing",
                ]
            elif vram_profile == "24":
                # High VRAM: no fp8 or block swap, only gradient checkpointing for safety
                extra_args += [
                    "--gradient_checkpointing",
                ]
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
                    # Save uploaded images first
                    img_files = save_uploaded_images(files, captions, output_dir)
                    if not img_files:
                        output_txt = "ERROR: Failed to save uploaded images. Please check file permissions and try again."
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
                        
                        # 1. Cache latents
                        output_queue.put("[Latent cache]\n")
                        with open(log_path, "w", encoding="utf-8") as log_file:
                            log_file.write(f"Training log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            log_file.write(f"Output directory: {output_dir}\n")
                            log_file.write(f"Output name: {output_name}\n")
                            log_file.write("="*60 + "\n\n")
                            log_file.write("[Latent cache]\n")
                            log_file.flush()
                            cache_latents_output = run_cache_latents(dataset_config, vae_model, stream_output=True, log_file=log_file)
                        output_queue.put("\n")
                        time.sleep(0.1)  # Small delay to ensure output is sent
                        
                        # 2. Cache textencoder
                        output_queue.put("[Text encoder cache]\n")
                        with open(log_path, "a", encoding="utf-8") as log_file:
                            log_file.write("\n[Text encoder cache]\n")
                            log_file.flush()
                            cache_textencoder_output = run_cache_textencoder(dataset_config, text_encoder, batch_size, vram_profile, stream_output=True, log_file=log_file)
                        output_queue.put("\n")
                        time.sleep(0.1)  # Small delay to ensure output is sent
                        
                        # 3. Train LoRA
                        output_queue.put("[Training]\n")
                        cmd = [
                            "python", "src/musubi_tuner/qwen_image_train_network.py",
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
                            "--sdpa"  # Use PyTorch's scaled dot product attention (requires PyTorch 2.0)
                        ] + extra_args
                        env = dict(os.environ)
                        env["CUDA_VISIBLE_DEVICES"] = "0"
                        # Improve CUDA memory handling to reduce fragmentation (helps OOM)
                        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                        
                        def run_training():
                            try:
                                with open(log_path, "a", encoding="utf-8") as log_file:
                                    log_file.write("\n[Training]\n")
                                    log_file.write(f"Command: {' '.join(cmd)}\n")
                                    log_file.write("="*60 + "\n\n")
                                    log_file.flush()
                                    
                                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                                           text=True, env=env, bufsize=0, universal_newlines=True)
                                    import sys
                                    try:
                                        for line in iter(proc.stdout.readline, ''):
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
                                        proc.wait()
                                        # Check if process crashed
                                        if proc.returncode != 0:
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
                                    
                                    if proc.returncode == 0:
                                        output_queue.put("\n" + "="*60 + "\n")
                                        output_queue.put("✅ TRAINING COMPLETED!\n")
                                        output_queue.put("="*60 + "\n")
                                        final_path = os.path.join(output_dir, output_name + ".safetensors")
                                        epoch_path = os.path.join(output_dir, output_name + "-000001.safetensors")
                                        output_queue.put(f"LoRA-filen har sparats i: {final_path}\n")
                                        if os.path.exists(epoch_path):
                                            output_queue.put(f"Och epoch-filer som: {epoch_path}\n")
                                        output_queue.put(f"Absolute path: {os.path.abspath(final_path)}\n")
                                        output_queue.put(f"Log file: {log_path}\n")
                                        output_queue.put("="*60 + "\n")
                                        
                                        log_file.write("\n" + "="*60 + "\n")
                                        log_file.write("✅ TRAINING COMPLETED!\n")
                                        log_file.write("="*60 + "\n")
                                        log_file.write(f"LoRA-filen har sparats i: {final_path}\n")
                                        if os.path.exists(epoch_path):
                                            log_file.write(f"Och epoch-filer som: {epoch_path}\n")
                                        log_file.write(f"Absolute path: {os.path.abspath(final_path)}\n")
                                        log_file.write("="*60 + "\n")
                                    else:
                                        output_queue.put("\n" + "="*60 + "\n")
                                        output_queue.put(f"❌ TRAINING ABORTED WITH ERROR (exit code {proc.returncode})\n")
                                        output_queue.put("="*60 + "\n")
                                        output_queue.put(f"See log file for details: {log_path}\n")
                                        
                                        log_file.write("\n" + "="*60 + "\n")
                                        log_file.write(f"❌ TRAINING ABORTED WITH ERROR (exit code {proc.returncode})\n")
                                        log_file.write("="*60 + "\n")
                                    log_file.write(f"\nTraining finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                    log_file.flush()
                                output_queue.put("[TRAINING_FINISHED]\n")  # Signal to close connection
                            except Exception as e:
                                output_queue.put("\n" + "="*60 + "\n")
                                output_queue.put(f"❌ FEL: {str(e)}\n")
                                output_queue.put("="*60 + "\n")
                                output_queue.put(f"See log file for details: {log_path}\n")
                                try:
                                    with open(log_path, "a", encoding="utf-8") as log_file:
                                        log_file.write("\n" + "="*60 + "\n")
                                        log_file.write(f"❌ FEL: {str(e)}\n")
                                        log_file.write("="*60 + "\n")
                                        import traceback
                                        log_file.write(traceback.format_exc())
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
                    img_files = save_uploaded_images(files, captions, output_dir)
                    for idx, fname in enumerate(img_files):
                        cap = captions[idx] if idx < len(captions) else ""
                        uploaded_gallery.append((fname, cap))
    return render_template_string(
        HTML,
        downloads=DOWNLOADS, download_msg=download_msg,
        trigger=trigger, vram_profile=vram_profile, epochs=epochs, batch_size=batch_size, lr=lr,
        prompt=prompt, resolution=resolution, seed=seed, rank=rank, dims=dims,
        output_dir=output_dir, output_name=output_name, image_repeats=image_repeats,
        saved_yaml=saved_yaml, output_txt=output_txt,
        uploaded_gallery=uploaded_gallery
    )

if __name__ == "__main__":
    # Prefer a production-ready WSGI server (waitress) if available,
    # but fall back to Flask's development server if not installed.
    try:
        from waitress import serve
        print("Starting MusubiTLX with waitress...")
        serve(app, host="0.0.0.0", port=5000)
    except ImportError:
        print("waitress is not installed, falling back to Flask development server.")
        app.run(host="0.0.0.0", port=5000, debug=True)
