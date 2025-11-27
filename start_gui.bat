@echo off
setlocal enabledelayedexpansion
REM MusubiTLX GUI Starter for Windows
REM Automatically starts the web GUI in the background
REM
REM Usage:
REM   start_gui.bat          - Start in background
REM   start_gui.bat --stop   - Stop the running server
REM   start_gui.bat --status - Check if server is running

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
REM Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Initialize PYTHON_CMD with default python command (will be updated if Python 3.13+ is detected)
set PYTHON_CMD=python

REM First check if webgui.py exists in current directory (we might already be in musubi-tuner)
if exist "webgui.py" (
    REM We're already in the right directory, stay here
    cd /d "%CD%"
    set SCRIPT_DIR=%CD%
    goto :skip_dir_check
)

REM Try to change to the script directory
cd /d "%SCRIPT_DIR%"

REM Verify we're in the right directory - check if webgui.py exists
if not exist "webgui.py" (
    REM If webgui.py doesn't exist, we might need to be in musubi-tuner subdirectory
    if exist "musubi-tuner\webgui.py" (
        cd /d "musubi-tuner"
        set SCRIPT_DIR=%CD%
    ) else (
        REM Check if we're already in a musubi-tuner directory by looking for src/musubi_tuner
        if exist "src\musubi_tuner" (
            REM We're already in musubi-tuner, stay here
            set SCRIPT_DIR=%CD%
        )
    )
)

:skip_dir_check

REM Check for --stop flag
if "%1"=="--stop" (
    tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
    if errorlevel 1 (
        echo MusubiTLX GUI is not running.
        echo.
        pause
        exit /b 0
    )
    
    echo Stopping MusubiTLX GUI server...
    REM Try to find and kill the webgui.py process
    for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr /I "PID"') do (
        REM Check if this process is running webgui.py (simplified check)
        wmic process where "ProcessId=%%i" get CommandLine 2>NUL | find /I "webgui.py" >NUL
        if not errorlevel 1 (
            echo   Stopping process %%i...
            taskkill /PID %%i /F >NUL 2>&1
        )
    )
    
    REM Wait a moment
    timeout /t 2 /nobreak >nul
    
    REM Check if still running
    tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
    if errorlevel 1 (
        echo [OK] MusubiTLX GUI server stopped successfully.
        echo.
        pause
    ) else (
        echo [WARNING]  Warning: Some processes may still be running.
        echo    Check with: tasklist ^| findstr python
        echo.
        pause
    )
    exit /b 0
)

REM Check for --status flag
if "%1"=="--status" (
    tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
    if errorlevel 1 (
        echo MusubiTLX GUI is not running.
        echo.
        pause
        exit /b 1
    )
    
    echo [OK] MusubiTLX GUI is running.
    echo    Stop with: start_gui.bat --stop
    echo.
    pause
    exit /b 0
)

:continue_setup
REM Check if webgui.py exists
if not exist "webgui.py" (
    echo.
    echo ========================================
    echo ERROR: webgui.py not found!
    echo ========================================
    echo.
    echo This script must be run from a directory containing webgui.py
    echo.
    echo Current directory: %CD%
    echo.
    echo Please download MusubiTLX-gui from: https://github.com/sajb0t/MusubiTLX-gui
    echo.
    echo.
    pause
    exit /b 1
)

REM Check if musubi-tuner repository exists
echo.
echo Checking setup...
echo Current directory: %CD%
if exist "src\musubi_tuner" (
    echo Musubi Tuner repository: [OK] Found
    goto :skip_musubi_check
)
echo Musubi Tuner repository: [ERROR] Not found
echo.
echo ========================================
echo [WARNING]  Musubi Tuner repository not found!
echo ========================================
echo.
echo This script detected webgui.py but musubi-tuner is not installed.
echo You need musubi-tuner for the GUI to work: https://github.com/kohya-ss/musubi-tuner
REM Check if musubi-tuner exists in parent directory
if exist "..\musubi-tuner\src\musubi_tuner" goto :found_in_parent
goto :not_found_anywhere

:found_in_parent
echo Found musubi-tuner in parent directory: ..\musubi-tuner
echo.
set /p copy_files="Copy webgui.py to musubi-tuner directory? (Y/n): "
    if /i not "%copy_files%"=="n" (
        copy webgui.py ..\musubi-tuner\ >nul 2>&1
        if exist "MUSUBITLX_GUI.md" (
            copy MUSUBITLX_GUI.md ..\musubi-tuner\ >nul 2>&1
        )
        if exist "start_gui.bat" (
            copy start_gui.bat ..\musubi-tuner\ >nul 2>&1
        )
        if exist "start_gui.sh" (
            copy start_gui.sh ..\musubi-tuner\ >nul 2>&1
        )
        if exist "stop_gui.bat" (
            copy stop_gui.bat ..\musubi-tuner\ >nul 2>&1
        )
        if exist "stop_gui.sh" (
            copy stop_gui.sh ..\musubi-tuner\ >nul 2>&1
        )
        echo [OK] Files copied to ..\musubi-tuner
        echo.
        set /p change_dir="Change directory to musubi-tuner and continue? (Y/n): "
        if /i not "%change_dir%"=="n" (
            cd /d "..\musubi-tuner"
            set SCRIPT_DIR=%CD%
        ) 
        if /i "%change_dir%"=="n" (
            echo.
            pause
            exit /b 0
        )
    )
    if /i "%copy_files%"=="n" (
        echo.
        pause
        exit /b 1
    )
    goto :skip_musubi_install

:not_found_anywhere
echo.
echo.
echo ========================================
echo MUSUBI-TUNER INSTALLATION REQUIRED
echo ========================================
echo.
echo This script can automatically download and install musubi-tuner for you!
        echo.
        echo Would you like to automatically clone musubi-tuner here?
        echo This will create a 'musubi-tuner' subdirectory with the repository.
        echo.
        echo Repository: https://github.com/kohya-ss/musubi-tuner
        echo.
        echo Note: After cloning, webgui.py and related files will be moved into the musubi-tuner folder.
        echo.
        echo IMPORTANT: You need musubi-tuner for the GUI to work!
        echo.
        echo.
        set /p clone_repo="Clone musubi-tuner automatically? (Y/n): "
        echo.
        if /i not "%clone_repo%"=="n" (
            REM Check if git is available
            where git >nul 2>&1
            if errorlevel 1 (
                echo [ERROR] Error: git is not installed. Please install git first.
                echo Download from: https://git-scm.com/download/win
                echo.
                pause
                exit /b 1
            )
            
            echo.
            echo Cloning musubi-tuner repository...
            git clone https://github.com/kohya-ss/musubi-tuner.git
            if errorlevel 1 (
                echo [ERROR] Failed to clone musubi-tuner repository.
                echo Please clone it manually: git clone https://github.com/kohya-ss/musubi-tuner.git
                echo.
                pause
                exit /b 1
            )
            
            echo [OK] musubi-tuner cloned successfully!
            echo.
            
            REM Move webgui.py and related files into musubi-tuner directory
            echo Moving files into musubi-tuner directory...
            move webgui.py musubi-tuner\ >nul 2>&1
            if exist "MUSUBITLX_GUI.md" (
                move MUSUBITLX_GUI.md musubi-tuner\ >nul 2>&1
            )
            REM Copy start_gui.bat instead of moving it (since we're running it)
            if exist "start_gui.bat" (
                copy start_gui.bat musubi-tuner\ >nul 2>&1
            )
            if exist "start_gui.sh" (
                move start_gui.sh musubi-tuner\ >nul 2>&1
            )
            if exist "stop_gui.bat" (
                move stop_gui.bat musubi-tuner\ >nul 2>&1
            )
            if exist "stop_gui.sh" (
                move stop_gui.sh musubi-tuner\ >nul 2>&1
            )
            echo [OK] Files moved to musubi-tuner\
            echo.
            
            REM Store original directory
            set ORIG_DIR=%CD%
            
            REM Change to musubi-tuner directory
            cd /d musubi-tuner
            set SCRIPT_DIR=%CD%
            set MUSUBI_DIR=%CD%
            
            echo.
            echo Files moved to musubi-tuner directory.
            echo.
            echo Restarting setup check in musubi-tuner directory...
            echo.
            timeout /t 2 /nobreak >nul
            echo Checking for start_gui.bat in: %CD%
            if exist "start_gui.bat" (
                echo Found start_gui.bat, continuing...
                REM We're already in the right directory
                REM Make sure we're in the right place by verifying webgui.py exists
                if exist "webgui.py" (
                    echo webgui.py found.
                    echo.
                    echo Restarting script from musubi-tuner directory...
                    REM Run the NEW script in musubi-tuner and exit this one
                    call "%CD%\start_gui.bat"
                    exit /b %ERRORLEVEL%
                ) else (
                    echo [ERROR] webgui.py not found in %CD%
                    echo Current directory: %CD%
                    pause
                    exit /b 1
                )
            )
            echo [ERROR] start_gui.bat not found in musubi-tuner directory!
            echo Current directory: %CD%
            echo Looking for: start_gui.bat
            dir /b *.bat
            pause
            exit /b 1
        ) else (
            echo.
            echo Please install musubi-tuner manually:
            echo   git clone https://github.com/kohya-ss/musubi-tuner.git
            echo   cd musubi-tuner
            echo   REM Copy webgui.py here
            echo.
            pause
            exit /b 1
        )
    )
)

:skip_musubi_install
:skip_musubi_check
REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    goto :venv_exists
)
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [WARNING]  Virtual environment not found!
    echo.
    echo Current directory: %CD%
    if exist "src\musubi_tuner" (
        echo Musubi Tuner repository: [OK] Found
        REM We have musubi-tuner, now ask about venv creation
        REM Close the inner if block first
        goto :ask_venv_creation
    ) else (
        echo Musubi Tuner repository: [ERROR] Not found
        echo.
        echo [ERROR] Musubi Tuner repository is required before creating virtual environment!
        echo.
        echo ========================================
        echo MUSUBI-TUNER INSTALLATION REQUIRED
        echo ========================================
        echo.
        echo This script can automatically download and install musubi-tuner for you!
        echo.
        echo Would you like to automatically clone musubi-tuner here?
        echo Repository: https://github.com/kohya-ss/musubi-tuner
        echo.
        echo Note: After cloning, webgui.py and related files will be moved into the musubi-tuner folder.
        echo.
        set /p clone_repo="Clone musubi-tuner automatically? (Y/n): "
        if /i not "%clone_repo%"=="n" (
            REM Check if git is available
            where git >nul 2>&1
            if errorlevel 1 (
                echo [ERROR] Error: git is not installed. Please install git first.
                echo Download from: https://git-scm.com/download/win
                echo.
                pause
                exit /b 1
            )
            
            echo.
            echo Cloning musubi-tuner repository...
            git clone https://github.com/kohya-ss/musubi-tuner.git
            if errorlevel 1 (
                echo [ERROR] Failed to clone musubi-tuner repository.
                echo Please clone it manually: git clone https://github.com/kohya-ss/musubi-tuner.git
                echo.
                pause
                exit /b 1
            )
            
            echo [OK] musubi-tuner cloned successfully!
            echo.
            
            REM Move webgui.py and related files into musubi-tuner directory
            echo Moving files into musubi-tuner directory...
            move webgui.py musubi-tuner\ >nul 2>&1
            if exist "MUSUBITLX_GUI.md" (
                move MUSUBITLX_GUI.md musubi-tuner\ >nul 2>&1
            )
            REM Copy start_gui.bat instead of moving it (since we're running it)
            if exist "start_gui.bat" (
                copy start_gui.bat musubi-tuner\ >nul 2>&1
            )
            if exist "start_gui.sh" (
                move start_gui.sh musubi-tuner\ >nul 2>&1
            )
            if exist "stop_gui.bat" (
                move stop_gui.bat musubi-tuner\ >nul 2>&1
            )
            if exist "stop_gui.sh" (
                move stop_gui.sh musubi-tuner\ >nul 2>&1
            )
            echo [OK] Files moved to musubi-tuner\
            echo.
            
            REM Store original directory
            set ORIG_DIR=%CD%
            
            REM Change to musubi-tuner directory
            cd /d musubi-tuner
            set SCRIPT_DIR=%CD%
            
            echo [OK] Changed to musubi-tuner directory.
            echo.
            echo Restarting setup check...
            echo.
            REM Restart from the beginning by calling the script from the new directory
            if exist "start_gui.bat" (
                call "start_gui.bat"
                exit /b %ERRORLEVEL%
            )
            if not exist "start_gui.bat" (
                echo [ERROR] start_gui.bat not found in musubi-tuner directory!
                pause
                exit /b 1
            )
        )
        if /i "%clone_repo%"=="n" (
            echo.
            echo Musubi Tuner is required. Please install it first:
            echo   git clone https://github.com/kohya-ss/musubi-tuner.git
            echo   cd musubi-tuner
            echo   REM Copy webgui.py here
            echo.
            pause
            exit /b 1
        )
    )
)
REM End of if not exist venv block

:ask_venv_creation
echo.
echo Would you like to create a virtual environment now?
set /p create_venv="Create virtual environment automatically? (Y/n): "

:create_venv_section
:venv_check_done
REM Check if user wants to create venv (default is yes if empty or anything other than "n")
if /i "%create_venv%"=="n" goto :skip_venv_creation

REM User wants to create venv (default or "y")
REM PYTHON_CMD is already initialized at the beginning of the script

REM First, check for Python 3.10/3.11 explicitly (as per requirements)
set PYTHON310_FOUND=0
set PYTHON311_FOUND=0
set PYTHON_FOUND_CMD=

REM Check for Python 3.11 first (newer)
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 --version > "%TEMP%\py311_ver.txt" 2>&1
    if exist "%TEMP%\py311_ver.txt" (
        findstr /I /C:"not installed" /C:"not found" /C:"Requested Python version" "%TEMP%\py311_ver.txt" >nul
        if errorlevel 1 (
            findstr /R /C:"Python 3\.11" "%TEMP%\py311_ver.txt" >nul
            if not errorlevel 1 (
                set PYTHON311_FOUND=1
                set PYTHON_FOUND_CMD=py -3.11
                set PYTHON_FOUND_VER=3.11
            )
        )
        del "%TEMP%\py311_ver.txt" >nul 2>&1
    )
)

REM Check for Python 3.10 if 3.11 not found
if !PYTHON311_FOUND!==0 (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.10 --version > "%TEMP%\py310_ver.txt" 2>&1
        if exist "%TEMP%\py310_ver.txt" (
            findstr /I /C:"not installed" /C:"not found" /C:"Requested Python version" "%TEMP%\py310_ver.txt" >nul
            if errorlevel 1 (
                findstr /R /C:"Python 3\.10" "%TEMP%\py310_ver.txt" >nul
                if not errorlevel 1 (
                    set PYTHON310_FOUND=1
                    set PYTHON_FOUND_CMD=py -3.10
                    set PYTHON_FOUND_VER=3.10
                )
            )
            del "%TEMP%\py310_ver.txt" >nul 2>&1
        )
    )
)

REM If Python 3.10 or 3.11 found, ask user to confirm
if !PYTHON311_FOUND!==1 (
    echo.
    echo Found Python 3.11
    set /p use_python="Use Python 3.11? (Y/n): "
    if /i not "!use_python!"=="n" (
        set PYTHON_CMD=!PYTHON_FOUND_CMD!
        goto :python_detection_done
    )
)
if !PYTHON310_FOUND!==1 (
    echo.
    echo Found Python 3.10
    set /p use_python="Use Python 3.10? (Y/n): "
    if /i not "!use_python!"=="n" (
        set PYTHON_CMD=!PYTHON_FOUND_CMD!
        goto :python_detection_done
    )
)

REM If no Python 3.10/3.11 found and user declined, try to auto-install
if not defined PYTHON_CMD (
    echo.
    echo Python 3.10 or 3.11 not found.
    echo.
    set /p auto_install="Would you like to automatically install Python 3.10? (Y/n): "
    if /i not "!auto_install!"=="n" (
        echo.
        echo Attempting to install Python 3.10...
        echo.
        
        set INSTALL_SUCCESS=0
        
        REM Check if Chocolatey is available
        where choco >nul 2>&1
        if not errorlevel 1 (
            echo Detected Chocolatey. Using choco to install Python 3.10...
            choco install python310 -y
            if not errorlevel 1 (
                set INSTALL_SUCCESS=1
                REM Refresh environment variables
                call refreshenv >nul 2>&1
                REM Check if py launcher can now find Python 3.10
                py -3.10 --version >nul 2>&1
                if not errorlevel 1 (
                    set PYTHON_CMD=py -3.10
                )
            )
        ) else (
            REM Check if winget is available (Windows 10/11)
            where winget >nul 2>&1
            if not errorlevel 1 (
                echo Detected winget. Attempting to install Python 3.10...
                winget install Python.Python.3.10 --accept-package-agreements --accept-source-agreements
                if not errorlevel 1 (
                    set INSTALL_SUCCESS=1
                    REM Refresh PATH (may need manual refresh)
                    echo.
                    echo [INFO] Python 3.10 installed. You may need to restart this script.
                    echo.
                    REM Try to find Python 3.10 after installation
                    timeout /t 3 /nobreak >nul
                    py -3.10 --version >nul 2>&1
                    if not errorlevel 1 (
                        set PYTHON_CMD=py -3.10
                    )
                )
            ) else (
                REM Neither Chocolatey nor winget available - prompt for manual installation
                echo.
                echo [INFO] No automatic installer found (Chocolatey or winget).
                echo.
                echo Please install Python 3.10 manually:
                echo   1. Download from: https://www.python.org/downloads/release/python-31011/
                echo   2. Run the installer
                echo   3. Make sure to check "Add Python to PATH" during installation
                echo   4. Restart this script after installation
                echo.
                echo Alternatively, install Chocolatey and try again:
                echo   https://chocolatey.org/install
                echo.
                set /p continue_manual="Continue anyway (Python installation required)? (y/N): "
                if /i not "!continue_manual!"=="y" (
                    pause
                    exit /b 1
                )
                echo.
                echo [WARNING] Python 3.10 is required. You must install it manually before running the GUI.
                echo.
            )
        )
        
        if !INSTALL_SUCCESS!==1 (
            echo.
            echo [OK] Python 3.10 installation completed!
            echo.
            if defined PYTHON_CMD (
                echo [OK] Verified: Python 3.10 is available.
                echo.
            ) else (
                echo [WARNING] Python 3.10 installed but not found in PATH.
                echo    You may need to restart this script or add it to PATH manually.
                echo.
            )
        )
    ) else (
        REM User declined auto-installation
        echo.
        echo [WARNING] Python 3.10 installation was skipped.
        echo.
        set /p continue_skip="Continue without Python 3.10? (y/N): "
        if /i not "!continue_skip!"=="y" (
            echo.
            echo Python 3.10 is required. Please install it manually:
            echo   Download from: https://www.python.org/downloads/
            echo.
            pause
            exit /b 1
        ) else (
            echo.
            echo [WARNING] Python 3.10 is required. You must install it manually before running the GUI.
            echo.
        )
    )
)

REM Fall back to default python command if still not set
if not defined PYTHON_CMD (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Error: Python is not installed. Please install Python 3.10-3.12 first.
        echo.
        pause
        exit /b 1
    )
    set PYTHON_CMD=python
)

:python_detection_done

REM If PYTHON_CMD is already set (from Python 3.10/3.11 detection), verify it and skip to venv creation
if defined PYTHON_CMD (
    REM Verify the Python command works
    !PYTHON_CMD! --version >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=2" %%v in ('!PYTHON_CMD! --version 2^>^&1') do set PYTHON_VER=%%v
        REM Verify it's a supported version (3.10, 3.11, or 3.12)
        echo !PYTHON_VER! | findstr /R "3\.1[0-2]" >nul
        if not errorlevel 1 (
            echo.
            echo [OK] Will use Python !PYTHON_VER!
            echo.
            goto :python_check_done
        )
    )
)

REM Check Python version (musubi-tuner requires Python 3.10-3.12, not 3.13+)

REM Check if PYTHON_CMD is set, otherwise use default python
if not defined PYTHON_CMD set PYTHON_CMD=python

!PYTHON_CMD! --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to check Python version.
    echo.
    pause
    exit /b 1
)

REM Extract Python version string (use PYTHON_CMD)
for /f "tokens=2" %%v in ('!PYTHON_CMD! --version 2^>^&1') do set PYTHON_VER=%%v

REM Check if version contains "3.13" or higher (use delayed expansion)
echo !PYTHON_VER! | findstr /R "3\.1[3-9]" >nul
if not errorlevel 1 (
    echo.
    echo ========================================
    echo [WARNING] Python version not supported!
    echo ========================================
    echo.
    echo Detected Python version: !PYTHON_VER!
    echo.
    echo musubi-tuner requires Python 3.10, 3.11, or 3.12.
    echo Python 3.13+ is NOT supported yet.
    echo.
    REM Check if py launcher is available and can find Python 3.12
    set PY312_FOUND=0
    set PYTHON312_CMD=
    
    REM First try: Check if py launcher exists and can find Python 3.12
    where py >nul 2>&1
    if not errorlevel 1 (
        REM Try to get Python 3.12 version - capture both stdout and stderr
        py -3.12 --version > "%TEMP%\py312_ver.txt" 2>&1
        if exist "%TEMP%\py312_ver.txt" (
            REM First check if output contains error messages indicating Python 3.12 doesn't exist
            findstr /I /C:"not installed" /C:"not found" /C:"Requested Python version" "%TEMP%\py312_ver.txt" >nul
            if errorlevel 1 (
                REM No error messages found, check if it's actually Python 3.12
                findstr /R /C:"Python 3\.12" "%TEMP%\py312_ver.txt" >nul
                if not errorlevel 1 (
                    set PY312_FOUND=1
                    set PYTHON312_CMD=py -3.12
                )
            )
        )
        if exist "%TEMP%\py312_ver.txt" del "%TEMP%\py312_ver.txt" >nul 2>&1
    )
    
    REM Second try: Check if python3.12 exists directly
    if !PY312_FOUND!==0 (
        where python3.12 >nul 2>&1
        if not errorlevel 1 (
            python3.12 --version > "%TEMP%\py312_ver.txt" 2>&1
            if exist "%TEMP%\py312_ver.txt" (
                findstr /R /C:"Python 3\.12" "%TEMP%\py312_ver.txt" >nul
                if not errorlevel 1 (
                    set PY312_FOUND=1
                    set PYTHON312_CMD=python3.12
                )
            )
            if exist "%TEMP%\py312_ver.txt" del "%TEMP%\py312_ver.txt" >nul 2>&1
        )
    )
    
    REM If Python 3.12 found, ask user
    if !PY312_FOUND!==1 (
        echo This script detected that you have Python 3.12 installed!
        echo.
        set /p use_py312="Use Python 3.12 instead? (Y/n): "
        if /i not "!use_py312!"=="n" (
            set PYTHON_CMD=!PYTHON312_CMD!
            for /f "tokens=2" %%v in ('!PYTHON312_CMD! --version 2^>^&1') do set PYTHON_VER=%%v
            echo.
            echo [OK] Will use Python !PYTHON_VER! instead.
            echo.
        ) else (
            echo.
            echo Please install Python 3.12 from: https://www.python.org/downloads/
            echo or use Python 3.11 or 3.10.
            echo.
            pause
            exit /b 1
        )
        goto :python_check_done
    ) else (
        REM Python 3.12 not found automatically - but ask user if they want to try anyway
        echo Python 3.12 was not found automatically.
        echo.
        echo However, if you have Python 3.12 installed, you can still try to use it.
        echo.
        where py >nul 2>&1
        if not errorlevel 1 (
            echo py launcher is available. You can try manually:
            echo   py -3.12 -m venv venv
            echo.
            set /p try_py312="Would you like to try 'py -3.12' anyway? (y/N): "
            if /i "!try_py312!"=="y" (
                REM Test if it actually works - capture both stdout and stderr
                py -3.12 --version > "%TEMP%\py312_test.txt" 2>&1
                REM Check if command succeeded (errorlevel 0) AND output contains valid Python 3.12 version
                if errorlevel 1 (
                    REM Command failed
                    if exist "%TEMP%\py312_test.txt" del "%TEMP%\py312_test.txt" >nul 2>&1
                    echo.
                    echo [ERROR] py -3.12 did not work. Python 3.12 is not installed.
                    echo.
                ) else (
                    REM Command succeeded, check if output is valid Python 3.12
                    if exist "%TEMP%\py312_test.txt" (
                        REM Check if output contains error messages indicating Python 3.12 doesn't exist
                        findstr /I /C:"not found" /C:"not installed" /C:"Requested Python version" "%TEMP%\py312_test.txt" >nul
                        if not errorlevel 1 (
                            REM Output contains error message, Python 3.12 doesn't exist
                            del "%TEMP%\py312_test.txt" >nul 2>&1
                            echo.
                            echo [ERROR] py -3.12 did not work. Python 3.12 is not installed.
                            echo.
                        ) else (
                            REM Check if output is valid Python 3.12 version
                            findstr /R /C:"Python 3.12" "%TEMP%\py312_test.txt" >nul
                            if not errorlevel 1 (
                                REM Valid Python 3.12 found!
                                set PYTHON_CMD=py -3.12
                                for /f "tokens=2" %%v in ('py -3.12 --version 2^>^&1') do set PYTHON_VER=%%v
                                echo.
                                echo [OK] Will use Python !PYTHON_VER!.
                                echo.
                                del "%TEMP%\py312_test.txt" >nul 2>&1
                                goto :python_check_done
                            ) else (
                                REM Output exists but doesn't match Python 3.12
                                del "%TEMP%\py312_test.txt" >nul 2>&1
                                echo.
                                echo [ERROR] py -3.12 did not return a valid Python 3.12 version.
                                echo.
                            )
                        )
                    ) else (
                        REM File doesn't exist, command failed
                        echo.
                        echo [ERROR] py -3.12 did not work. Python 3.12 is not installed.
                        echo.
                    )
                )
            )
        )
        
        echo.
        echo You can try:
        echo   1. Install Python 3.12 from: https://www.python.org/downloads/
        echo   2. Or use Python 3.11 or 3.10 if available
        echo.
        echo Checking for alternative Python versions...
        echo.
        
        REM Check for py -3.11
        where py >nul 2>&1
        if not errorlevel 1 (
            for /f "tokens=*" %%i in ('py -3.11 --version 2^>^&1') do (
                echo Found: %%i (use with: py -3.11 -m venv venv)
            )
        )
        
        REM Check for py -3.10
        where py >nul 2>&1
        if not errorlevel 1 (
            for /f "tokens=*" %%i in ('py -3.10 --version 2^>^&1') do (
                echo Found: %%i (use with: py -3.10 -m venv venv)
            )
        )
        
        echo.
        pause
        exit /b 1
    )
)

:python_check_done
echo Creating virtual environment...
%PYTHON_CMD% -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment. Please create it manually:
    echo    %PYTHON_CMD% -m venv venv
    echo.
    pause
    exit /b 1
)

REM Wait a moment for venv creation to fully complete
timeout /t 1 /nobreak >nul

REM Verify venv was created successfully
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment creation may have failed. venv\Scripts\activate.bat not found.
    echo Please create it manually: %PYTHON_CMD% -m venv venv
    echo.
    pause
    exit /b 1
)

echo [OK] Virtual environment created!
echo.

REM Install dependencies for musubi-tuner
echo Dependencies need to be installed for musubi-tuner to work.
echo.
set /p install_deps="Install dependencies automatically now? (Y/n): "
if /i "%install_deps%"=="n" goto :skip_deps_install

REM User wants to install dependencies (default or "y")
echo.
echo Installing dependencies...
echo This may take several minutes...
echo.

REM GPU Detection
echo Checking for NVIDIA GPU...
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [OK] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name --format=csv,noheader | findstr /N /C:".*" | findstr /V "^$"
    echo.
) else (
    echo [INFO] NVIDIA GPU not detected. Proceeding with CPU-only option recommended.
    echo.
)

REM CUDA/MODEL PATH & TORCH SETUP - Interactive Menu
echo ========================================
echo Choose PyTorch installation:
echo ========================================
echo.
echo [1] CUDA 12.1 (Stable ^& recommended for RTX 30/40-series)
echo [2] CUDA 12.4 (Intermediate option, viable for RTX 50xx/Blackwell)
echo [3] CUDA 12.8 (Latest stable - recommended for RTX 50xx/Blackwell, PyTorch 2.7+^)
echo [4] CPU only (No GPU acceleration)
echo [5] Skip PyTorch (I will install myself)
echo.
set /p cuda_choice="Enter your choice (1-5): "
REM Trim whitespace from user input (delayed expansion already enabled)
set cuda_choice=!cuda_choice: =!

set TORCH_INSTALLED=0
set PYTORCH_INDEX_URL=
set CUDA_VERSION=

if "!cuda_choice!"=="1" (
    set CUDA_VERSION=12.1
    set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
    goto :install_pytorch
)
if "!cuda_choice!"=="2" (
    set CUDA_VERSION=12.4
    set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
    goto :install_pytorch
)
if "!cuda_choice!"=="3" (
    set CUDA_VERSION=12.8
    set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
    echo.
    echo Installing CUDA 12.8 ^(latest stable for Blackwell GPUs^).
    echo.
    goto :install_pytorch
)
if "!cuda_choice!"=="4" (
    set CUDA_VERSION=CPU
    set PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    echo.
    echo [WARNING] CPU-only installation selected. No GPU acceleration available.
    echo Training/inference will be very slow.
    echo.
    goto :install_pytorch
)
if "!cuda_choice!"=="5" (
    echo.
    echo [WARNING] You have chosen to skip PyTorch installation.
    echo You must install the correct version yourself before the GUI can run with GPU.
    echo Consult PyTorch documentation for your CUDA version.
    echo Continuing without PyTorch installation...
    echo.
    set TORCH_INSTALLED=0
    goto :skip_pytorch_install
)

REM Invalid choice
echo.
echo [ERROR] Invalid choice. Please select 1-5.
echo.
pause
call venv\Scripts\deactivate.bat >nul 2>&1
goto :deps_install_done

:install_pytorch
echo Step 1: Installing PyTorch and torchvision...
call venv\Scripts\activate.bat
if "%CUDA_VERSION%"=="CPU" (
    pip install torch torchvision torchaudio --index-url %PYTORCH_INDEX_URL%
) else (
    pip install torch torchvision torchaudio --index-url %PYTORCH_INDEX_URL%
)
if errorlevel 1 (
    echo.
    echo [WARNING] PyTorch installation failed.
    set /p continue_pytorch="Continue anyway? (y/N): "
    if /i not "!continue_pytorch!"=="y" (
        echo.
        echo Installation aborted. Please install PyTorch manually.
        echo.
        call venv\Scripts\deactivate.bat
        pause
        exit /b 1
    )
    echo.
    echo Continuing without PyTorch validation...
    set TORCH_INSTALLED=0
    call venv\Scripts\deactivate.bat
    goto :skip_pytorch_validation
)

set TORCH_INSTALLED=1
echo.
echo [OK] PyTorch installed successfully!
echo.

REM PyTorch Validation
echo Validating PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A')" 2>nul
if errorlevel 1 (
    echo.
    echo [WARNING] PyTorch validation failed. You may need to troubleshoot installation manually.
    echo Check PyTorch official documentation.
    echo.
) else (
    echo.
    echo [OK] PyTorch validated successfully.
    python -c "import torch; print('CUDA support:', 'Available' if torch.cuda.is_available() else 'Not available (CPU-only mode)')" 2>nul
    echo.
)
call venv\Scripts\deactivate.bat

:skip_pytorch_validation
:skip_pytorch_install

REM Install musubi-tuner dependencies (from pyproject.toml or requirements.txt)
echo.
echo Step 2: Installing musubi-tuner dependencies...
call venv\Scripts\activate.bat

REM Check if pyproject.toml exists (preferred)
if exist "pyproject.toml" (
    echo Installing from pyproject.toml...
    pip install -e .
    if errorlevel 1 (
        echo.
        echo [WARNING] musubi-tuner installation may have failed.
        set /p continue_musubi="Continue anyway? (y/N): "
        if /i not "!continue_musubi!"=="y" (
            echo.
            echo Installation aborted. Please install musubi-tuner manually: pip install -e .
            echo.
            call venv\Scripts\deactivate.bat
            pause
            exit /b 1
        )
        echo.
    ) else (
        echo.
        echo [OK] musubi-tuner installed successfully!
        echo.
    )
) else if exist "requirements.txt" (
    echo Installing from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [WARNING] Dependencies installation may have failed.
        set /p continue_reqs="Continue anyway? (y/N): "
        if /i not "!continue_reqs!"=="y" (
            echo.
            echo Installation aborted. Please install dependencies manually.
            echo.
            call venv\Scripts\deactivate.bat
            pause
            exit /b 1
        )
        echo.
    ) else (
        echo.
        echo [OK] Dependencies installed successfully!
        echo.
    )
) else (
    echo [INFO] No requirements.txt found in musubi-tuner. Skipping Musubi dependency installation.
    echo.
)

REM Install GUI dependencies (from webgui.py imports)
echo Step 3: Installing GUI dependencies...
echo Installing Flask, PyYAML, toml, waitress, psutil...
pip install Flask PyYAML toml waitress psutil
if errorlevel 1 (
    echo.
    echo [WARNING] Some GUI dependencies may have failed to install.
    echo You may need to install them manually: pip install Flask PyYAML toml waitress psutil
    echo.
) else (
    echo.
    echo [OK] GUI dependencies installed successfully!
    echo.
)

REM Optional: gevent for better SSE streaming (live log updates)
echo Optional: Install gevent for better live log streaming?
echo   - gevent improves real-time log updates during training
echo   - Without it, logs may update with slight delays
echo.
set /p install_gevent="Install gevent? (y/N): "
if /i "%install_gevent%"=="y" (
    echo Installing gevent...
    pip install gevent
    if errorlevel 1 (
        echo.
        echo [WARNING] gevent installation failed. Live logs will work but may be less responsive.
        echo.
    ) else (
        echo.
        echo [OK] gevent installed successfully!
        echo.
    )
)

REM Optional dependencies
echo Optional: Install additional dependencies for extra features?
echo   - ascii-magic (dataset verification)
echo   - matplotlib (timestep visualization)
echo   - tensorboard (training progress logging)
echo   - prompt-toolkit (interactive prompt editing)
echo.
set /p install_opt="Install optional dependencies? (y/N): "
if /i "%install_opt%"=="y" (
    echo.
    echo Installing optional dependencies...
    pip install ascii-magic matplotlib tensorboard prompt-toolkit
    if errorlevel 1 (
        echo.
        echo [WARNING] Some optional dependencies may have failed to install.
        echo.
    ) else (
        echo.
        echo [OK] Optional dependencies installed successfully!
        echo.
    )
)
call venv\Scripts\deactivate.bat
goto :deps_install_done

:skip_deps_install
echo.
echo To install dependencies manually, run:
echo   venv\Scripts\activate
echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo   pip install -e .
echo.

:deps_install_done
goto :venv_exists

:skip_venv_creation
echo.
echo Virtual environment is required. Please create it manually:
echo   %PYTHON_CMD% -m venv venv
echo   venv\Scripts\activate
echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo   pip install -e .
echo.
pause
exit /b 1

:venv_exists
REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check for required model files
set MISSING_MODELS=0

if not exist "qwen_image_bf16.safetensors" (
    set MISSING_MODELS=1
    set MISSING_DIT=1
)
if not exist "diffusion_pytorch_model.safetensors" (
    set MISSING_MODELS=1
    set MISSING_VAE=1
)
if not exist "qwen_2.5_vl_7b.safetensors" (
    set MISSING_MODELS=1
    set MISSING_TEXT_ENCODER=1
)

REM Show missing models and offer to download
if %MISSING_MODELS%==1 goto :show_missing_models
goto :models_ok

:show_missing_models
echo.
echo [WARNING]  Missing model files detected:
echo.
if defined MISSING_DIT echo   - DiT Model ^(qwen_image_bf16.safetensors^) - REQUIRED for training
if defined MISSING_VAE echo   - VAE Model ^(diffusion_pytorch_model.safetensors^) - REQUIRED for training
if defined MISSING_TEXT_ENCODER echo   - Text Encoder ^(qwen_2.5_vl_7b.safetensors^) - needed for auto-captioning
echo.
echo You can either:
echo   1. Download them manually from the web GUI ^(after starting the server^)
echo   2. Let this script download them now ^(requires PowerShell^)
echo.
set /p download="Download missing models now? (Y/n): "
if /i "%download%"=="n" goto :skip_model_download

echo.
echo Downloading models...
echo.

if defined MISSING_DIT (
    echo Downloading DiT Model ^(this may take a while, ~7GB^)...
    powershell -Command "Invoke-WebRequest -Uri 'https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors?download=true' -OutFile 'qwen_image_bf16.safetensors'"
    if exist "qwen_image_bf16.safetensors" (
        echo   [OK] Downloaded: qwen_image_bf16.safetensors
    ) else (
        echo   [ERROR] Download failed: qwen_image_bf16.safetensors
        exit /b 1
    )
)

if defined MISSING_VAE (
    echo Downloading VAE Model...
    powershell -Command "Invoke-WebRequest -Uri 'https://huggingface.co/Qwen/Qwen-Image/resolve/main/vae/diffusion_pytorch_model.safetensors?download=true' -OutFile 'diffusion_pytorch_model.safetensors'"
    if exist "diffusion_pytorch_model.safetensors" (
        echo   [OK] Downloaded: diffusion_pytorch_model.safetensors
    ) else (
        echo   [ERROR] Download failed: diffusion_pytorch_model.safetensors
        exit /b 1
    )
)

if defined MISSING_TEXT_ENCODER (
    echo Downloading Text Encoder ^(this may take a while, ~16GB^)...
    powershell -Command "Invoke-WebRequest -Uri 'https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors?download=true' -OutFile 'qwen_2.5_vl_7b.safetensors'"
    if exist "qwen_2.5_vl_7b.safetensors" (
        echo   [OK] Downloaded: qwen_2.5_vl_7b.safetensors
    ) else (
        echo   [WARNING] Text Encoder download failed. Auto-captioning will not work.
        echo            You can download it later from the web GUI.
    )
)

echo.
echo [OK] Model download complete!
echo.
goto :models_ok

:skip_model_download
if defined MISSING_DIT goto :warn_required_missing
if defined MISSING_VAE goto :warn_required_missing
echo.
echo Skipping model download. You can download models from the web GUI.
echo.
goto :models_ok

:warn_required_missing
echo.
echo [WARNING]  Required models are missing. Training will not work.
echo    You can download them from the web GUI after starting the server.
echo.
set /p continue_anyway="Continue anyway? (y/N): "
if /i not "%continue_anyway%"=="y" exit /b 1
goto :models_ok

:models_ok

REM Check if server is already running
tasklist /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq webgui.py*" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Warning: MusubiTLX GUI appears to already be running!
    echo Check with: tasklist ^| findstr python
    echo Kill with: taskkill /F /FI "WINDOWTITLE eq webgui.py*"
    echo.
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" (
        echo.
        pause
        exit /b 1
    )
)

REM Start server in background with output to log file
echo Starting MusubiTLX GUI in background...
echo.
echo Server will listen on: http://localhost:5000
echo Logs will be written to: webgui.log
echo.
echo Useful commands:
echo    View logs:    type webgui.log
echo    View live:    powershell -Command "Get-Content webgui.log -Wait -Tail 50"
echo    Stop server:  start_gui.bat --stop  (or: stop_gui.bat)
echo    Check status: start_gui.bat --status
echo.

REM Start Python process in a new minimized window with venv activated
REM Using 'start /MIN' creates a separate process that won't die when parent closes
REM We need to activate venv in the new cmd window before running python
start "MusubiTLX Server" /MIN cmd /c "cd /d "%CD%" && call venv\Scripts\activate.bat && python webgui.py > webgui.log 2>&1"

REM Wait a moment to check if it started
timeout /t 2 /nobreak > nul

REM Check if process is running (simplified check)
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] Server is running in background.
    echo You can safely close this window.
    echo To view logs: type webgui.log
) else (
    echo Server may have failed to start. Check webgui.log for errors:
    echo.
    if exist webgui.log (
        type webgui.log
    ) else (
        echo ^(log file not found^)
    )
    echo.
    pause
    exit /b 1
)

REM Clean up original files from parent directory AFTER server started
REM Note: We do NOT delete start_gui.bat because it might be the file currently running
if defined ORIG_DIR (
    if not "%CD%"=="%ORIG_DIR%" (
        if exist "%ORIG_DIR%\webgui.py" (
            echo.
            echo Clean up old installation files from parent directory?
            echo ^(Note: start_gui.bat will not be deleted as it may be running^)
            set /p cleanup_confirm="Delete old files from %ORIG_DIR%? (Y/n): "
            if /i not "!cleanup_confirm!"=="n" (
                echo.
                echo Cleaning up original installation files...
                del "%ORIG_DIR%\webgui.py" >nul 2>&1
                if exist "%ORIG_DIR%\MUSUBITLX_GUI.md" del "%ORIG_DIR%\MUSUBITLX_GUI.md" >nul 2>&1
                if exist "%ORIG_DIR%\start_gui.sh" del "%ORIG_DIR%\start_gui.sh" >nul 2>&1
                if exist "%ORIG_DIR%\stop_gui.bat" del "%ORIG_DIR%\stop_gui.bat" >nul 2>&1
                if exist "%ORIG_DIR%\stop_gui.sh" del "%ORIG_DIR%\stop_gui.sh" >nul 2>&1
                echo [OK] Cleaned up. You can manually delete %ORIG_DIR%\start_gui.bat later.
                echo.
            )
        )
    )
)

REM Keep window open so user can see status and easily view logs
REM Server runs in background, so window can be kept open without blocking
echo.
echo ========================================
echo Server is running in the background.
echo You can keep this window open to monitor, or close it safely.
echo ========================================
echo.
echo To stop the server, run: start_gui.bat --stop
echo Or close this window and use stop_gui.bat
echo.
pause

