#!/bin/bash
# MusubiTLX GUI Stopper
# Simple script to stop the running GUI server

cd "$(dirname "$0")"

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
    exit 1
fi

