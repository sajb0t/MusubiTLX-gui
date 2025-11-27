#!/bin/bash
# MusubiTLX GUI Starter for Linux/macOS
# Automatically starts the web GUI with SSH disconnect protection
#
# Usage:
#   ./start_gui.sh          - Start in background (recommended, survives SSH disconnect)
#   ./start_gui.sh --fg     - Start in foreground (interactive mode)
#   ./start_gui.sh --stop   - Stop the running server
#   ./start_gui.sh --status - Check if server is running

cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"

# Prevent infinite recursion when script calls itself
if [ "$MUSUBI_RECURSIVE" = "1" ]; then
    MUSUBI_RECURSIVE=""
else
    export MUSUBI_RECURSIVE="1"
fi

# Check if webgui.py exists (we're either in MusubiTLX-gui repo or musubi-tuner repo with webgui.py)
if [ ! -f "webgui.py" ]; then
    echo "ERROR: webgui.py not found!"
    echo "This script must be run from a directory containing webgui.py"
    echo ""
    echo "Please download MusubiTLX-gui from: https://github.com/sajb0t/MusubiTLX-gui"
    exit 1
fi

# Show status of what we found
echo ""
echo "Checking setup..."
echo "Current directory: $(pwd)"
if [ -d "src/musubi_tuner" ]; then
    echo "Musubi Tuner repository: ✅ Found"
else
    echo "Musubi Tuner repository: ❌ Not found"
fi
echo ""

# Check if musubi-tuner repository exists (src/musubi_tuner directory)
if [ ! -d "src/musubi_tuner" ]; then
    echo ""
    echo "========================================"
    echo "⚠️  Musubi Tuner repository not found!"
    echo "========================================"
    echo ""
    echo "This script detected webgui.py but musubi-tuner is not installed."
    echo "You need musubi-tuner for the GUI to work: https://github.com/kohya-ss/musubi-tuner"
    echo ""
    
    # Check if musubi-tuner exists in parent directory
    PARENT_DIR="$(dirname "$SCRIPT_DIR")"
    if [ -d "$PARENT_DIR/musubi-tuner/src/musubi_tuner" ]; then
        echo "Found musubi-tuner in parent directory: $PARENT_DIR/musubi-tuner"
        echo ""
        read -p "Copy webgui.py to musubi-tuner directory? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            cp webgui.py "$PARENT_DIR/musubi-tuner/"
            if [ -f "MUSUBITLX_GUI.md" ]; then
                cp MUSUBITLX_GUI.md "$PARENT_DIR/musubi-tuner/" 2>/dev/null || true
            fi
            if [ -f "start_gui.sh" ]; then
                cp start_gui.sh "$PARENT_DIR/musubi-tuner/" 2>/dev/null || true
                chmod +x "$PARENT_DIR/musubi-tuner/start_gui.sh" 2>/dev/null || true
            fi
            if [ -f "start_gui.bat" ]; then
                cp start_gui.bat "$PARENT_DIR/musubi-tuner/" 2>/dev/null || true
            fi
            if [ -f "stop_gui.sh" ]; then
                cp stop_gui.sh "$PARENT_DIR/musubi-tuner/" 2>/dev/null || true
                chmod +x "$PARENT_DIR/musubi-tuner/stop_gui.sh" 2>/dev/null || true
            fi
            if [ -f "stop_gui.bat" ]; then
                cp stop_gui.bat "$PARENT_DIR/musubi-tuner/" 2>/dev/null || true
            fi
            echo "✅ Files copied to $PARENT_DIR/musubi-tuner"
            echo ""
            
            # Store original directory for cleanup later
            export ORIG_DIR="$(pwd)"
            
            read -p "Change directory to musubi-tuner and continue? (Y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                cd "$PARENT_DIR/musubi-tuner"
                SCRIPT_DIR="$(pwd)"
            else
                exit 0
            fi
        else
            exit 1
        fi
    else
        echo ""
        echo "========================================"
        echo "MUSUBI-TUNER INSTALLATION REQUIRED"
        echo "========================================"
        echo ""
        echo "Would you like to automatically clone musubi-tuner here?"
        echo "This will create a 'musubi-tuner' subdirectory with the repository."
        echo ""
        echo "Repository: https://github.com/kohya-ss/musubi-tuner"
        echo ""
        echo "Note: After cloning, webgui.py and related files will be moved into the musubi-tuner folder."
        echo ""
        read -p "Clone musubi-tuner automatically? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            # Check if git is available
            if ! command -v git >/dev/null 2>&1; then
                echo "❌ Error: git is not installed. Please install git first."
                exit 1
            fi
            
            echo ""
            echo "Cloning musubi-tuner repository..."
            if git clone https://github.com/kohya-ss/musubi-tuner.git; then
                echo "✅ musubi-tuner cloned successfully!"
                echo ""
                
                # Move webgui.py and related files into musubi-tuner directory
                echo "Moving files into musubi-tuner directory..."
                mv webgui.py musubi-tuner/ 2>/dev/null || true
                if [ -f "MUSUBITLX_GUI.md" ]; then
                    mv MUSUBITLX_GUI.md musubi-tuner/ 2>/dev/null || true
                fi
                if [ -f "start_gui.sh" ]; then
                    mv start_gui.sh musubi-tuner/ 2>/dev/null || true
                    chmod +x musubi-tuner/start_gui.sh 2>/dev/null || true
                fi
                if [ -f "start_gui.bat" ]; then
                    mv start_gui.bat musubi-tuner/ 2>/dev/null || true
                fi
                if [ -f "stop_gui.sh" ]; then
                    mv stop_gui.sh musubi-tuner/ 2>/dev/null || true
                    chmod +x musubi-tuner/stop_gui.sh 2>/dev/null || true
                fi
                if [ -f "stop_gui.bat" ]; then
                    mv stop_gui.bat musubi-tuner/ 2>/dev/null || true
                fi
                echo "✅ Files moved to musubi-tuner/"
                echo ""
                
                # Store original directory for cleanup later
                export ORIG_DIR="$(pwd)"
                
                # Change to musubi-tuner directory
                cd musubi-tuner
                SCRIPT_DIR="$(pwd)"
                
                echo "Next steps:"
                echo "  1. Create virtual environment: python -m venv venv"
                echo "  2. Activate it: source venv/bin/activate"
                echo "  3. Install dependencies (follow musubi-tuner installation instructions)"
                echo ""
                read -p "Create virtual environment now? (Y/n) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                    # Use same Python detection/installation logic as below
                    # First, check for Python 3.10 or 3.11 explicitly
                    PYTHON_FOUND=""
                    PYTHON_FOUND_CMD=""
                    
                    # Check for Python 3.11 first (newer)
                    if command -v python3.11 >/dev/null 2>&1; then
                        PYTHON_VER_CHECK=$(python3.11 --version 2>&1 | awk '{print $2}')
                        if echo "$PYTHON_VER_CHECK" | grep -q "^3\.11"; then
                            PYTHON_FOUND="3.11"
                            PYTHON_FOUND_CMD="python3.11"
                        fi
                    fi
                    
                    # Check for Python 3.10 if 3.11 not found
                    if [ -z "$PYTHON_FOUND" ] && command -v python3.10 >/dev/null 2>&1; then
                        PYTHON_VER_CHECK=$(python3.10 --version 2>&1 | awk '{print $2}')
                        if echo "$PYTHON_VER_CHECK" | grep -q "^3\.10"; then
                            PYTHON_FOUND="3.10"
                            PYTHON_FOUND_CMD="python3.10"
                        fi
                    fi
                    
                    # If Python 3.10 or 3.11 found, ask user to confirm
                    if [ -n "$PYTHON_FOUND" ]; then
                        echo ""
                        echo "Found Python $PYTHON_FOUND"
                        read -p "Use Python $PYTHON_FOUND? (Y/n) " -n 1 -r
                        echo
                        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                            PYTHON_CMD="$PYTHON_FOUND_CMD"
                        fi
                    fi
                    
                    # If no Python 3.10/3.11 found, fall back to default
                    if [ -z "$PYTHON_CMD" ]; then
                        if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
                            echo "❌ Error: Python is not installed. Please install Python 3.10-3.12 first."
                            exit 1
                        fi
                        PYTHON_CMD="python3"
                        if ! command -v python3 >/dev/null 2>&1; then
                            PYTHON_CMD="python"
                        fi
                    fi
                    
                    # Check Python version (musubi-tuner requires Python 3.10-3.12, not 3.13+)
                    PYTHON_VER=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
                    PYTHON_MAJOR=$(echo "$PYTHON_VER" | cut -d. -f1)
                    PYTHON_MINOR=$(echo "$PYTHON_VER" | cut -d. -f2)
                    
                    # Check if version is 3.13 or higher
                    if [ "$PYTHON_MAJOR" = "3" ] && [ "$PYTHON_MINOR" -ge 13 ]; then
                        echo ""
                        echo "========================================"
                        echo "⚠️  WARNING: Python version not supported!"
                        echo "========================================"
                        echo ""
                        echo "Detected Python version: $PYTHON_VER"
                        echo ""
                        echo "musubi-tuner requires Python 3.10, 3.11, or 3.12."
                        echo "Python 3.13+ is NOT supported yet."
                        echo ""
                        
                        # Check if python3.12 is available
                        if command -v python3.12 >/dev/null 2>&1; then
                            echo "✅ This script detected that you have Python 3.12 installed!"
                            echo ""
                            read -p "Use Python 3.12 instead? (Y/n) " -n 1 -r
                            echo
                            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                                PYTHON_CMD="python3.12"
                                PYTHON_VER=$(python3.12 --version 2>&1 | awk '{print $2}')
                                echo ""
                                echo "✅ Will use Python $PYTHON_VER instead."
                                echo ""
                            else
                                echo ""
                                echo "Please install Python 3.12 from: https://www.python.org/downloads/"
                                echo "or use Python 3.11 or 3.10."
                                echo ""
                                exit 1
                            fi
                        else
                            echo "Python 3.12 was not found. Please install Python 3.12 from:"
                            echo "https://www.python.org/downloads/"
                            echo ""
                            echo "Or use Python 3.11 or 3.10."
                            echo ""
                            exit 1
                        fi
                    fi
                    
                    echo "Creating virtual environment..."
                    if $PYTHON_CMD -m venv venv; then
                        # Wait a moment for venv creation to fully complete
                        sleep 1
                        
                        # Verify venv was created successfully
                        if [ ! -f "venv/bin/activate" ]; then
                            echo "❌ Virtual environment creation may have failed. venv/bin/activate not found."
                            echo "Please create it manually: python -m venv venv"
                            exit 1
                        fi
                        
                        echo "✅ Virtual environment created!"
                        echo ""
                        
                        # Install dependencies for musubi-tuner
                        echo "Dependencies need to be installed for musubi-tuner to work."
                        echo ""
                        read -p "Install dependencies automatically now? (Y/n) " -n 1 -r
                        echo
                        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                            echo ""
                            echo "Installing dependencies..."
                            echo "This may take several minutes..."
                            echo ""
                            
                            # GPU Detection
                            echo "Checking for NVIDIA GPU..."
                            if command -v nvidia-smi >/dev/null 2>&1; then
                                echo "✅ NVIDIA GPU detected:"
                                nvidia-smi --query-gpu=name --format=csv,noheader
                                echo ""
                            else
                                echo "ℹ️  NVIDIA GPU not detected. Proceeding with CPU-only option recommended."
                                echo ""
                            fi
                            
                            # CUDA/MODEL PATH & TORCH SETUP - Interactive Menu
                            echo "========================================"
                            echo "Choose PyTorch installation:"
                            echo "========================================"
                            echo ""
                            echo "[1] CUDA 12.1 (Stable & recommended for RTX 30/40-series)"
                            echo "[2] CUDA 12.4 (Intermediate option, viable for RTX 50xx/Blackwell)"
                            echo "[3] CUDA 12.8 (Latest stable - recommended for RTX 50xx/Blackwell, PyTorch 2.7+)"
                            echo "[4] CPU only (No GPU acceleration)"
                            echo "[5] Skip PyTorch (I will install myself)"
                            echo ""
                            read -p "Enter your choice (1-5): " cuda_choice
                            
                            TORCH_INSTALLED=0
                            PYTORCH_INDEX_URL=""
                            CUDA_VERSION=""
                            
                            case "$cuda_choice" in
                                1)
                                    CUDA_VERSION="12.1"
                                    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
                                    ;;
                                2)
                                    CUDA_VERSION="12.4"
                                    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
                                    ;;
                                3)
                                    CUDA_VERSION="12.8"
                                    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
                                    echo ""
                                    echo "Installing CUDA 12.8 (latest stable for Blackwell GPUs)."
                                    echo ""
                                    ;;
                                4)
                                    CUDA_VERSION="CPU"
                                    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
                                    echo ""
                                    echo "⚠️  [WARNING] CPU-only installation selected. No GPU acceleration available."
                                    echo "Training/inference will be very slow."
                                    echo ""
                                    ;;
                                5)
                                    echo ""
                                    echo "⚠️  [WARNING] You have chosen to skip PyTorch installation."
                                    echo "You must install the correct version yourself before the GUI can run with GPU."
                                    echo "Consult PyTorch documentation for your CUDA version."
                                    echo "Continuing without PyTorch..."
                                    echo ""
                                    TORCH_INSTALLED=0
                                    ;;
                                *)
                                    echo ""
                                    echo "❌ [ERROR] Invalid choice. Please select 1-5."
                                    echo ""
                                    deactivate 2>/dev/null || true
                                    exit 1
                                    ;;
                            esac
                            
                            # Install PyTorch if not skipped
                            if [ "$cuda_choice" != "5" ]; then
                                echo "Step 1: Installing PyTorch and torchvision..."
                                source venv/bin/activate
                                if pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"; then
                                    TORCH_INSTALLED=1
                                    echo ""
                                    echo "✅ PyTorch installed successfully!"
                                    echo ""
                                    
                                    # PyTorch Validation
                                    echo "Validating PyTorch installation..."
                                    if python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A')" 2>/dev/null; then
                                        echo ""
                                        echo "✅ PyTorch validated successfully."
                                        python -c "import torch; print('CUDA support:', 'Available' if torch.cuda.is_available() else 'Not available (CPU-only mode)')" 2>/dev/null
                                        echo ""
                                    else
                                        echo ""
                                        echo "⚠️  [WARNING] PyTorch validation failed. You may need to troubleshoot installation manually."
                                        echo "Check PyTorch official documentation."
                                        echo ""
                                    fi
                                else
                                    echo ""
                                    echo "⚠️  [WARNING] PyTorch installation failed."
                                    read -p "Continue anyway? (y/N) " -n 1 -r
                                    echo
                                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                                        echo ""
                                        echo "Installation aborted. Please install PyTorch manually."
                                        echo ""
                                        deactivate
                                        exit 1
                                    fi
                                    echo ""
                                    echo "Continuing without PyTorch validation..."
                                    TORCH_INSTALLED=0
                                fi
                                deactivate
                            fi
                            
                            # Install musubi-tuner dependencies (from pyproject.toml or requirements.txt)
                            echo ""
                            echo "Step 2: Installing musubi-tuner dependencies..."
                            source venv/bin/activate
                            
                            # Check if pyproject.toml exists (preferred)
                            if [ -f "pyproject.toml" ]; then
                                echo "Installing from pyproject.toml..."
                                if pip install -e .; then
                                    echo ""
                                    echo "✅ musubi-tuner installed successfully!"
                                    echo ""
                                else
                                    echo ""
                                    echo "⚠️  [WARNING] musubi-tuner installation may have failed."
                                    read -p "Continue anyway? (y/N) " -n 1 -r
                                    echo
                                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                                        echo ""
                                        echo "Installation aborted. Please install musubi-tuner manually: pip install -e ."
                                        echo ""
                                        deactivate
                                        exit 1
                                    fi
                                    echo ""
                                fi
                            elif [ -f "requirements.txt" ]; then
                                echo "Installing from requirements.txt..."
                                if pip install -r requirements.txt; then
                                    echo ""
                                    echo "✅ Dependencies installed successfully!"
                                    echo ""
                                else
                                    echo ""
                                    echo "⚠️  [WARNING] Dependencies installation may have failed."
                                    read -p "Continue anyway? (y/N) " -n 1 -r
                                    echo
                                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                                        echo ""
                                        echo "Installation aborted. Please install dependencies manually."
                                        echo ""
                                        deactivate
                                        exit 1
                                    fi
                                    echo ""
                                fi
                            else
                                echo "ℹ️  [INFO] No requirements.txt found in musubi-tuner. Skipping Musubi dependency installation."
                                echo ""
                            fi
                            
                            # Install GUI dependencies (from webgui.py imports)
                            echo "Step 3: Installing GUI dependencies..."
                            echo "Installing Flask, PyYAML, toml, waitress, psutil..."
                            if pip install Flask PyYAML toml waitress psutil; then
                                echo ""
                                echo "✅ GUI dependencies installed successfully!"
                                echo ""
                            else
                                echo ""
                                echo "⚠️  [WARNING] Some GUI dependencies may have failed to install."
                                echo "You may need to install them manually: pip install Flask PyYAML toml waitress psutil"
                                echo ""
                            fi
                            
                            # Optional: gevent for better SSE streaming (live log updates)
                            echo "Optional: Install gevent for better live log streaming?"
                            echo "  - gevent improves real-time log updates during training"
                            echo "  - Without it, logs may update with slight delays"
                            echo ""
                            read -p "Install gevent? (y/N) " -n 1 -r
                            echo
                            if [[ $REPLY =~ ^[Yy]$ ]]; then
                                echo "Installing gevent..."
                                if pip install gevent; then
                                    echo ""
                                    echo "✅ gevent installed successfully!"
                                    echo ""
                                else
                                    echo ""
                                    echo "⚠️  gevent installation failed. Live logs will work but may be less responsive."
                                    echo ""
                                fi
                            fi
                            
                            # Optional dependencies
                            echo "Optional: Install additional dependencies for extra features?"
                            echo "  - ascii-magic (dataset verification)"
                            echo "  - matplotlib (timestep visualization)"
                            echo "  - tensorboard (training progress logging)"
                            echo "  - prompt-toolkit (interactive prompt editing)"
                            echo ""
                            read -p "Install optional dependencies? (y/N) " -n 1 -r
                            echo
                            if [[ $REPLY =~ ^[Yy]$ ]]; then
                                echo ""
                                echo "Installing optional dependencies..."
                                if pip install ascii-magic matplotlib tensorboard prompt-toolkit; then
                                    echo ""
                                    echo "✅ Optional dependencies installed successfully!"
                                    echo ""
                                else
                                    echo ""
                                    echo "⚠️  [WARNING] Some optional dependencies may have failed to install."
                                    echo ""
                                fi
                            fi
                            deactivate
                        else
                            echo ""
                            echo "To install dependencies manually, run:"
                            echo "  source venv/bin/activate"
                            echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
                            echo "  pip install -e ."
                            echo ""
                        fi
                    else
                        echo "❌ Failed to create virtual environment. Please create it manually."
                        exit 1
                    fi
                fi
            else
                echo "❌ Failed to clone musubi-tuner repository."
                echo "Please clone it manually: git clone https://github.com/kohya-ss/musubi-tuner.git"
                exit 1
            fi
        else
            echo ""
            echo "Please install musubi-tuner manually:"
            echo "  git clone https://github.com/kohya-ss/musubi-tuner.git"
            echo "  cd musubi-tuner"
            echo "  # Copy webgui.py here"
            exit 1
        fi
    fi
fi

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "⚠️  Virtual environment not found!"
    echo ""
    echo "Current directory: $(pwd)"
    if [ -d "src/musubi_tuner" ]; then
        echo "Musubi Tuner repository: ✅ Found"
    else
        echo "Musubi Tuner repository: ❌ Not found"
        echo ""
        echo "❌ Musubi Tuner repository is required before creating virtual environment!"
        echo ""
        echo "========================================"
        echo "MUSUBI-TUNER INSTALLATION REQUIRED"
        echo "========================================"
        echo ""
        echo "This script can automatically download and install musubi-tuner for you!"
        echo ""
        echo "Would you like to automatically clone musubi-tuner here?"
        echo "Repository: https://github.com/kohya-ss/musubi-tuner"
        echo ""
        echo "Note: After cloning, webgui.py and related files will be moved into the musubi-tuner folder."
        echo ""
        read -p "Clone musubi-tuner automatically? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            # Check if git is available
            if ! command -v git >/dev/null 2>&1; then
                echo "❌ Error: git is not installed. Please install git first."
                exit 1
            fi
            
            echo ""
            echo "Cloning musubi-tuner repository..."
            if git clone https://github.com/kohya-ss/musubi-tuner.git; then
                echo "✅ musubi-tuner cloned successfully!"
                echo ""
                
                # Move webgui.py and related files into musubi-tuner directory
                echo "Moving webgui.py into musubi-tuner directory..."
                mv webgui.py musubi-tuner/ 2>/dev/null || true
                if [ -f "MUSUBITLX_GUI.md" ]; then
                    mv MUSUBITLX_GUI.md musubi-tuner/ 2>/dev/null || true
                fi
                if [ -f "start_gui.sh" ]; then
                    mv start_gui.sh musubi-tuner/ 2>/dev/null || true
                    chmod +x musubi-tuner/start_gui.sh 2>/dev/null || true
                fi
                echo "✅ Files moved to musubi-tuner/"
                echo ""
                
                # Store original directory for cleanup later (though files are moved, not copied)
                export ORIG_DIR="$(pwd)"
                
                # Change to musubi-tuner directory
                cd musubi-tuner
                SCRIPT_DIR="$(pwd)"
                
                echo "✅ Changed to musubi-tuner directory."
                echo ""
                echo "Restarting setup check..."
                echo ""
                # Restart from the beginning by calling the script from the new directory
                if [ -f "start_gui.sh" ]; then
                    # Clear the recursive flag to allow the new script to run normally
                    unset MUSUBI_RECURSIVE
                    exec ./start_gui.sh
                else
                    echo "❌ Error: start_gui.sh not found in musubi-tuner directory!"
                    exit 1
                fi
            else
                echo "❌ Failed to clone musubi-tuner repository."
                echo "Please clone it manually: git clone https://github.com/kohya-ss/musubi-tuner.git"
                exit 1
            fi
        else
            echo ""
            echo "Musubi Tuner is required. Please install it first:"
            echo "  git clone https://github.com/kohya-ss/musubi-tuner.git"
            echo "  cd musubi-tuner"
            echo "  # Copy webgui.py here"
            echo ""
            exit 1
        fi
    fi
    echo ""
    echo "A virtual environment is required to run the GUI."
    echo ""
    read -p "Create virtual environment now? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        # First, check for Python 3.10 or 3.11 explicitly (as per requirements)
        PYTHON_FOUND=""
        PYTHON_FOUND_CMD=""
        
        # Check for Python 3.11 first (newer)
        if command -v python3.11 >/dev/null 2>&1; then
            PYTHON_VER_CHECK=$(python3.11 --version 2>&1 | awk '{print $2}')
            if echo "$PYTHON_VER_CHECK" | grep -q "^3\.11"; then
                PYTHON_FOUND="3.11"
                PYTHON_FOUND_CMD="python3.11"
            fi
        fi
        
        # Check for Python 3.10 if 3.11 not found
        if [ -z "$PYTHON_FOUND" ] && command -v python3.10 >/dev/null 2>&1; then
            PYTHON_VER_CHECK=$(python3.10 --version 2>&1 | awk '{print $2}')
            if echo "$PYTHON_VER_CHECK" | grep -q "^3\.10"; then
                PYTHON_FOUND="3.10"
                PYTHON_FOUND_CMD="python3.10"
            fi
        fi
        
        # If Python 3.10 or 3.11 found, ask user to confirm
        if [ -n "$PYTHON_FOUND" ]; then
            echo ""
            echo "Found Python $PYTHON_FOUND"
            read -p "Use Python $PYTHON_FOUND? (Y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                PYTHON_CMD="$PYTHON_FOUND_CMD"
            fi
        fi
        
        # If no Python 3.10/3.11 found or user declined, try to auto-install
        if [ -z "$PYTHON_CMD" ]; then
            echo ""
            echo "Python 3.10 or 3.11 not found."
            echo ""
            read -p "Would you like to automatically install Python 3.10? (Y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                echo ""
                echo "Attempting to install Python 3.10..."
                echo ""
                
                # Detect OS and use appropriate package manager
                OS_TYPE="unknown"
                if [ -f /etc/os-release ]; then
                    . /etc/os-release
                    OS_TYPE="$ID"
                elif [ "$(uname)" = "Darwin" ]; then
                    OS_TYPE="macos"
                fi
                
                INSTALL_SUCCESS=0
                
                case "$OS_TYPE" in
                    ubuntu|debian)
                        echo "Detected Ubuntu/Debian. Using apt-get..."
                        if command -v sudo >/dev/null 2>&1; then
                            if sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv; then
                                INSTALL_SUCCESS=1
                            fi
                        else
                            echo "⚠️  sudo not available. Please install Python 3.10 manually:"
                            echo "   apt-get update && apt-get install -y python3.10 python3.10-venv"
                        fi
                        ;;
                    fedora|rhel|centos)
                        echo "Detected Fedora/RHEL/CentOS. Using dnf..."
                        if command -v sudo >/dev/null 2>&1; then
                            if sudo dnf install -y python3.10 python3.10-devel; then
                                INSTALL_SUCCESS=1
                            fi
                        else
                            echo "⚠️  sudo not available. Please install Python 3.10 manually:"
                            echo "   dnf install -y python3.10 python3.10-devel"
                        fi
                        ;;
                    macos)
                        echo "Detected macOS. Using Homebrew..."
                        if command -v brew >/dev/null 2>&1; then
                            if brew install python@3.10; then
                                INSTALL_SUCCESS=1
                                # Homebrew installs to a specific path
                                if [ -f "/opt/homebrew/bin/python3.10" ]; then
                                    PYTHON_CMD="/opt/homebrew/bin/python3.10"
                                elif [ -f "/usr/local/bin/python3.10" ]; then
                                    PYTHON_CMD="/usr/local/bin/python3.10"
                                fi
                            fi
                        else
                            echo "⚠️  Homebrew not found. Please install Homebrew first:"
                            echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                            echo "   Then run: brew install python@3.10"
                        fi
                        ;;
                    *)
                        echo "⚠️  Unsupported OS detected ($OS_TYPE)."
                        echo "   Please install Python 3.10 manually from: https://www.python.org/downloads/"
                        ;;
                esac
                
                if [ $INSTALL_SUCCESS -eq 1 ]; then
                    echo ""
                    echo "✅ Python 3.10 installation completed!"
                    echo ""
                    # Verify installation
                    if [ -z "$PYTHON_CMD" ]; then
                        if command -v python3.10 >/dev/null 2>&1; then
                            PYTHON_CMD="python3.10"
                        fi
                    fi
                    if [ -n "$PYTHON_CMD" ] && $PYTHON_CMD --version >/dev/null 2>&1; then
                        echo "✅ Verified: $($PYTHON_CMD --version)"
                        echo ""
                    else
                        echo "⚠️  Python 3.10 installed but not found in PATH."
                        echo "   You may need to restart your terminal or add it to PATH manually."
                        echo ""
                    fi
                else
                    echo ""
                    echo "⚠️  Python 3.10 installation failed or was skipped."
                    echo ""
                    read -p "Continue without Python 3.10? (y/N) " -n 1 -r
                    echo
                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                        echo ""
                        echo "Python 3.10 is required. Please install it manually:"
                        echo "   Ubuntu/Debian: apt-get install python3.10 python3.10-venv"
                        echo "   Fedora/RHEL: dnf install python3.10 python3.10-devel"
                        echo "   macOS: brew install python@3.10"
                        echo "   Or download from: https://www.python.org/downloads/"
                        echo ""
                        exit 1
                    else
                        echo ""
                        echo "⚠️  [WARNING] Python 3.10 is required. You must install it manually before running the GUI."
                        echo ""
                    fi
                fi
            fi
        fi
        
        # Fall back to default python3/python if nothing found yet
        if [ -z "$PYTHON_CMD" ]; then
            if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
                echo "❌ Error: Python is not installed. Please install Python 3.10-3.12 first."
                exit 1
            fi
            PYTHON_CMD="python3"
            if ! command -v python3 >/dev/null 2>&1; then
                PYTHON_CMD="python"
            fi
        fi
        
        # Check Python version (musubi-tuner requires Python 3.10-3.12, not 3.13+)
        PYTHON_VER=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
        PYTHON_MAJOR=$(echo "$PYTHON_VER" | cut -d. -f1)
        PYTHON_MINOR=$(echo "$PYTHON_VER" | cut -d. -f2)
        
        # Check if version is 3.13 or higher
        if [ "$PYTHON_MAJOR" = "3" ] && [ "$PYTHON_MINOR" -ge 13 ]; then
            echo ""
            echo "========================================"
            echo "⚠️  WARNING: Python version not supported!"
            echo "========================================"
            echo ""
            echo "Detected Python version: $PYTHON_VER"
            echo ""
            echo "musubi-tuner requires Python 3.10, 3.11, or 3.12."
            echo "Python 3.13+ is NOT supported yet."
            echo ""
            
            # Check if python3.12 is available
            if command -v python3.12 >/dev/null 2>&1; then
                echo "✅ This script detected that you have Python 3.12 installed!"
                echo ""
                read -p "Use Python 3.12 instead? (Y/n) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                    PYTHON_CMD="python3.12"
                    PYTHON_VER=$(python3.12 --version 2>&1 | awk '{print $2}')
                    echo ""
                    echo "✅ Will use Python $PYTHON_VER instead."
                    echo ""
                else
                    echo ""
                    echo "Please install Python 3.12 from: https://www.python.org/downloads/"
                    echo "or use Python 3.11 or 3.10."
                    echo ""
                    exit 1
                fi
            else
                echo "Python 3.12 was not found. Please install Python 3.12 from:"
                echo "https://www.python.org/downloads/"
                echo ""
                echo "Or use Python 3.11 or 3.10."
                echo ""
                exit 1
            fi
        fi
        
        echo "Creating virtual environment..."
        if $PYTHON_CMD -m venv venv; then
            # Wait a moment for venv creation to fully complete
            sleep 1
            
            # Verify venv was created successfully
            if [ ! -f "venv/bin/activate" ]; then
                echo "❌ Virtual environment creation may have failed. venv/bin/activate not found."
                echo "Please create it manually: python -m venv venv"
                exit 1
            fi
            
            echo "✅ Virtual environment created!"
            echo ""
            
            # Install dependencies for musubi-tuner
            echo "Dependencies need to be installed for musubi-tuner to work."
            echo ""
            read -p "Install dependencies automatically now? (Y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                echo ""
                echo "Installing dependencies..."
                echo "This may take several minutes..."
                echo ""
                
                # GPU Detection
                echo "Checking for NVIDIA GPU..."
                if command -v nvidia-smi >/dev/null 2>&1; then
                    echo "✅ NVIDIA GPU detected:"
                    nvidia-smi --query-gpu=name --format=csv,noheader
                    echo ""
                else
                    echo "ℹ️  NVIDIA GPU not detected. Proceeding with CPU-only option recommended."
                    echo ""
                fi
                
                # CUDA/MODEL PATH & TORCH SETUP - Interactive Menu
                echo "========================================"
                echo "Choose PyTorch installation:"
                echo "========================================"
                echo ""
                echo "[1] CUDA 12.1 (Stable & recommended for RTX 30/40-series)"
                echo "[2] CUDA 12.4 (Intermediate option, viable for RTX 50xx/Blackwell)"
                echo "[3] CUDA 12.8 (Latest stable - recommended for RTX 50xx/Blackwell, PyTorch 2.7+)"
                echo "[4] CPU only (No GPU acceleration)"
                echo "[5] Skip PyTorch (I will install myself)"
                echo ""
                read -p "Enter your choice (1-5): " cuda_choice
                
                TORCH_INSTALLED=0
                PYTORCH_INDEX_URL=""
                CUDA_VERSION=""
                
                case "$cuda_choice" in
                    1)
                        CUDA_VERSION="12.1"
                        PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
                        ;;
                    2)
                        CUDA_VERSION="12.4"
                        PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
                        ;;
                    3)
                        CUDA_VERSION="12.8"
                        PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
                        echo ""
                        echo "Installing CUDA 12.8 (latest stable for Blackwell GPUs)."
                        echo ""
                        ;;
                    4)
                        CUDA_VERSION="CPU"
                        PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
                        echo ""
                        echo "⚠️  [WARNING] CPU-only installation selected. No GPU acceleration available."
                        echo "Training/inference will be very slow."
                        echo ""
                        ;;
                    5)
                        echo ""
                        echo "⚠️  [WARNING] You have chosen to skip PyTorch installation."
                        echo "You must install the correct version yourself before the GUI can run with GPU."
                        echo "Consult PyTorch documentation for your CUDA version."
                        echo "Continuing without PyTorch..."
                        echo ""
                        TORCH_INSTALLED=0
                        ;;
                    *)
                        echo ""
                        echo "❌ [ERROR] Invalid choice. Please select 1-5."
                        echo ""
                        deactivate 2>/dev/null || true
                        exit 1
                        ;;
                esac
                
                # Install PyTorch if not skipped
                if [ "$cuda_choice" != "5" ]; then
                    echo "Step 1: Installing PyTorch and torchvision..."
                    source venv/bin/activate
                    if pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"; then
                        TORCH_INSTALLED=1
                        echo ""
                        echo "✅ PyTorch installed successfully!"
                        echo ""
                        
                        # PyTorch Validation
                        echo "Validating PyTorch installation..."
                        if python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A')" 2>/dev/null; then
                            echo ""
                            echo "✅ PyTorch validated successfully."
                            python -c "import torch; print('CUDA support:', 'Available' if torch.cuda.is_available() else 'Not available (CPU-only mode)')" 2>/dev/null
                            echo ""
                        else
                            echo ""
                            echo "⚠️  [WARNING] PyTorch validation failed. You may need to troubleshoot installation manually."
                            echo "Check PyTorch official documentation."
                            echo ""
                        fi
                    else
                        echo ""
                        echo "⚠️  [WARNING] PyTorch installation failed."
                        read -p "Continue anyway? (y/N) " -n 1 -r
                        echo
                        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                            echo ""
                            echo "Installation aborted. Please install PyTorch manually."
                            echo ""
                            deactivate
                            exit 1
                        fi
                        echo ""
                        echo "Continuing without PyTorch validation..."
                        TORCH_INSTALLED=0
                    fi
                    deactivate
                fi
                
                # Install musubi-tuner dependencies (from pyproject.toml or requirements.txt)
                echo ""
                echo "Step 2: Installing musubi-tuner dependencies..."
                source venv/bin/activate
                
                # Check if pyproject.toml exists (preferred)
                if [ -f "pyproject.toml" ]; then
                    echo "Installing from pyproject.toml..."
                    if pip install -e .; then
                        echo ""
                        echo "✅ musubi-tuner installed successfully!"
                        echo ""
                    else
                        echo ""
                        echo "⚠️  [WARNING] musubi-tuner installation may have failed."
                        read -p "Continue anyway? (y/N) " -n 1 -r
                        echo
                        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                            echo ""
                            echo "Installation aborted. Please install musubi-tuner manually: pip install -e ."
                            echo ""
                            deactivate
                            exit 1
                        fi
                        echo ""
                    fi
                elif [ -f "requirements.txt" ]; then
                    echo "Installing from requirements.txt..."
                    if pip install -r requirements.txt; then
                        echo ""
                        echo "✅ Dependencies installed successfully!"
                        echo ""
                    else
                        echo ""
                        echo "⚠️  [WARNING] Dependencies installation may have failed."
                        read -p "Continue anyway? (y/N) " -n 1 -r
                        echo
                        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                            echo ""
                            echo "Installation aborted. Please install dependencies manually."
                            echo ""
                            deactivate
                            exit 1
                        fi
                        echo ""
                    fi
                else
                    echo "ℹ️  [INFO] No requirements.txt found in musubi-tuner. Skipping Musubi dependency installation."
                    echo ""
                fi
                
                # Install GUI dependencies (from webgui.py imports)
                echo "Step 3: Installing GUI dependencies..."
                echo "Installing Flask, PyYAML, toml, waitress, psutil..."
                if pip install Flask PyYAML toml waitress psutil; then
                    echo ""
                    echo "✅ GUI dependencies installed successfully!"
                    echo ""
                else
                    echo ""
                    echo "⚠️  [WARNING] Some GUI dependencies may have failed to install."
                    echo "You may need to install them manually: pip install Flask PyYAML toml waitress psutil"
                    echo ""
                fi
                
                # Optional: gevent for better SSE streaming (live log updates)
                echo "Optional: Install gevent for better live log streaming?"
                echo "  - gevent improves real-time log updates during training"
                echo "  - Without it, logs may update with slight delays"
                echo ""
                read -p "Install gevent? (y/N) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "Installing gevent..."
                    if pip install gevent; then
                        echo ""
                        echo "✅ gevent installed successfully!"
                        echo ""
                    else
                        echo ""
                        echo "⚠️  gevent installation failed. Live logs will work but may be less responsive."
                        echo ""
                    fi
                fi
                
                # Optional dependencies
                echo "Optional: Install additional dependencies for extra features?"
                echo "  - ascii-magic (dataset verification)"
                echo "  - matplotlib (timestep visualization)"
                echo "  - tensorboard (training progress logging)"
                echo "  - prompt-toolkit (interactive prompt editing)"
                echo ""
                read -p "Install optional dependencies? (y/N) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo ""
                    echo "Installing optional dependencies..."
                    if pip install ascii-magic matplotlib tensorboard prompt-toolkit; then
                        echo ""
                        echo "✅ Optional dependencies installed successfully!"
                        echo ""
                    else
                        echo ""
                        echo "⚠️  [WARNING] Some optional dependencies may have failed to install."
                        echo ""
                    fi
                fi
                deactivate
            else
                echo ""
                echo "To install dependencies manually, run:"
                echo "  source venv/bin/activate"
                echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
                echo "  pip install -e ."
                echo ""
            fi
        else
            echo "❌ Failed to create virtual environment. Please create it manually:"
            echo "  python -m venv venv"
            exit 1
        fi
    else
        echo ""
        echo "Virtual environment is required. Please create it manually:"
        echo "  python -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        echo "  pip install -e ."
        exit 1
    fi
fi

# Check for required model files
ALL_MODELS=(
    "qwen_image_bf16.safetensors:DiT Model:required"
    "diffusion_pytorch_model.safetensors:VAE Model:required"
    "qwen_2.5_vl_7b.safetensors:Text Encoder (for auto-captioning):optional"
)

MISSING_MODELS=()
HAS_MISSING_REQUIRED=0

for model_info in "${ALL_MODELS[@]}"; do
    model_file="${model_info%%:*}"
    rest="${model_info#*:}"
    model_name="${rest%%:*}"
    model_type="${rest##*:}"
    if [ ! -f "$model_file" ]; then
        MISSING_MODELS+=("$model_file:$model_name:$model_type")
        if [ "$model_type" = "required" ]; then
            HAS_MISSING_REQUIRED=1
        fi
    fi
done

# Show missing models and offer to download
if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo "⚠️  Missing model files detected:"
    echo ""
    for model_info in "${MISSING_MODELS[@]}"; do
        model_file="${model_info%%:*}"
        rest="${model_info#*:}"
        model_name="${rest%%:*}"
        model_type="${rest##*:}"
        if [ "$model_type" = "required" ]; then
            echo "  - $model_name ($model_file) - REQUIRED for training"
        else
            echo "  - $model_name ($model_file)"
        fi
    done
    echo ""
    
    echo "You can either:"
    echo "  1. Download them manually from the web GUI (after starting the server)"
    echo "  2. Let this script download them now (requires curl or wget)"
    echo ""
    read -p "Download missing models now? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        source venv/bin/activate
        
        # Download function
        download_model() {
            local model_file="$1"
            local model_name="$2"
            local model_type="$3"
            local url=""
            local size_hint=""
            
            case "$model_file" in
                "qwen_image_bf16.safetensors")
                    url="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors?download=true"
                    size_hint="~7GB"
                    ;;
                "diffusion_pytorch_model.safetensors")
                    url="https://huggingface.co/Qwen/Qwen-Image/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true"
                    size_hint="~335MB"
                    ;;
                "qwen_2.5_vl_7b.safetensors")
                    url="https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors?download=true"
                    size_hint="~16GB"
                    ;;
            esac
            
            if [ -z "$url" ]; then
                echo "  ❌ Unknown model: $model_file"
                return 1
            fi
            
            echo "  Downloading $model_name ($size_hint)..."
            if command -v curl >/dev/null 2>&1; then
                curl -L --progress-bar -o "$model_file" "$url" || return 1
            elif command -v wget >/dev/null 2>&1; then
                wget --progress=bar:force -O "$model_file" "$url" || return 1
            else
                echo "  ❌ Neither curl nor wget found. Please install one of them."
                return 1
            fi
            
            if [ -f "$model_file" ]; then
                echo "  ✅ Downloaded: $model_file"
                return 0
            else
                echo "  ❌ Download failed: $model_file"
                return 1
            fi
        }
        
        # Download all missing models
        for model_info in "${MISSING_MODELS[@]}"; do
            model_file="${model_info%%:*}"
            rest="${model_info#*:}"
            model_name="${rest%%:*}"
            model_type="${rest##*:}"
            if [ "$model_type" = "required" ]; then
                download_model "$model_file" "$model_name" "$model_type" || exit 1
            else
                download_model "$model_file" "$model_name" "$model_type" || echo "  ⚠️  Optional model download failed, continuing..."
            fi
        done
        
        echo ""
        echo "✅ Model download complete!"
        echo ""
    else
        if [ "$HAS_MISSING_REQUIRED" = "1" ]; then
            echo ""
            echo "⚠️  Required models are missing. Training will not work."
            echo "   You can download them from the web GUI after starting the server."
            echo ""
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            echo ""
            echo "Skipping model download. You can download models from the web GUI."
            echo ""
        fi
    fi
fi

source venv/bin/activate

# Check if already running with nohup (stdin is not a terminal)
if [ ! -t 0 ]; then
    # Already running in background, just execute
    python webgui.py
    exit $?
fi

# Check for --stop flag
if [ "$1" = "--stop" ]; then
    if ! pgrep -f "python webgui.py" > /dev/null; then
        echo "MusubiTLX GUI is not running."
        exit 0
    fi
    
    echo "Stopping MusubiTLX GUI server..."
    PIDS=$(pgrep -f "python webgui.py")
    for PID in $PIDS; do
        echo "  Stopping process $PID..."
        kill $PID 2>/dev/null
    done
    
    # Wait a moment and check if processes are still running
    sleep 2
    STILL_RUNNING=$(pgrep -f "python webgui.py" 2>/dev/null)
    if [ -n "$STILL_RUNNING" ]; then
        echo "  Some processes didn't stop gracefully. Force killing..."
        pkill -9 -f "python webgui.py" 2>/dev/null
        sleep 1
    fi
    
    if ! pgrep -f "python webgui.py" > /dev/null; then
        echo "✅ MusubiTLX GUI server stopped successfully."
    else
        echo "⚠️  Warning: Some processes may still be running."
        echo "   Check with: ps aux | grep webgui.py"
    fi
    exit 0
fi

# Check for --status flag
if [ "$1" = "--status" ]; then
    if pgrep -f "python webgui.py" > /dev/null; then
        PIDS=$(pgrep -f "python webgui.py")
        echo "✅ MusubiTLX GUI is running (PIDs: $PIDS)"
        echo "   Stop with: ./start_gui.sh --stop"
        exit 0
    else
        echo "MusubiTLX GUI is not running."
        exit 1
    fi
fi

# Check for --fg flag (foreground mode)
if [ "$1" = "--fg" ]; then
    echo "Starting MusubiTLX GUI in foreground mode..."
    echo "Server will listen on: http://0.0.0.0:5000"
    echo "Press Ctrl+C to stop"
    echo ""
    python webgui.py
    exit $?
fi

# Running interactively - automatically use nohup for SSH disconnect protection
echo "Starting MusubiTLX GUI in background (will survive SSH disconnect)..."
echo "(Use './start_gui.sh --fg' to run in foreground instead)"
echo ""

# Check if already running
if pgrep -f "python webgui.py" > /dev/null; then
    echo "⚠️  Warning: MusubiTLX GUI appears to already be running!"
    echo "   Check with: ps aux | grep webgui.py"
    echo "   Kill with: pkill -f webgui.py"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start with nohup in background
nohup python webgui.py > webgui.log 2>&1 &
PID=$!

echo "✅ Server started in background (PID: $PID)"
echo "   Server listening on: http://0.0.0.0:5000"
echo ""
echo "Useful commands:"
echo "   View logs:    tail -f webgui.log"
echo "   Stop server:  ./start_gui.sh --stop  (or: kill $PID)"
echo "   Check status: ./start_gui.sh --status"
echo ""
echo "Note: Press Ctrl+C to exit this terminal (server will keep running in background)"
echo ""

# Wait a moment to see if it starts successfully
sleep 2
if ! kill -0 $PID 2>/dev/null; then
    echo "❌ Server failed to start. Check webgui.log for errors:"
    echo ""
    tail -20 webgui.log 2>/dev/null || echo "   (log file not found)"
    exit 1
fi

# Clean up original files from parent directory AFTER server started
# Note: We do NOT delete start_gui.sh because it might be the script currently running
if [ -n "$ORIG_DIR" ] && [ -d "$ORIG_DIR" ]; then
    if [ "$(pwd)" != "$ORIG_DIR" ]; then
        if [ -f "$ORIG_DIR/webgui.py" ]; then
            echo ""
            echo "Clean up old installation files from parent directory?"
            echo "(Note: start_gui.sh will not be deleted as it may be running)"
            read -p "Delete old files from $ORIG_DIR? (Y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                echo ""
                echo "Cleaning up original installation files..."
                rm -f "$ORIG_DIR/webgui.py" 2>/dev/null || true
                rm -f "$ORIG_DIR/MUSUBITLX_GUI.md" 2>/dev/null || true
                rm -f "$ORIG_DIR/start_gui.bat" 2>/dev/null || true
                rm -f "$ORIG_DIR/stop_gui.sh" 2>/dev/null || true
                rm -f "$ORIG_DIR/stop_gui.bat" 2>/dev/null || true
                echo "✅ Cleaned up. You can manually delete $ORIG_DIR/start_gui.sh later."
                echo ""
            fi
        fi
    fi
fi

echo "Server is running. You can safely close this terminal or disconnect SSH."
echo "To view logs: tail -f webgui.log"
