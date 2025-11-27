@echo off
REM MusubiTLX GUI Stopper for Windows
REM Simple script to stop the running GUI server

cd /d "%~dp0"

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
    REM Check if this process is running webgui.py
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
    echo ✅ MusubiTLX GUI server stopped successfully.
    echo.
    pause
    exit /b 0
) else (
    echo ⚠️  Warning: Some processes may still be running.
    echo    Check with: tasklist ^| findstr python
    echo.
    pause
    exit /b 1
)

