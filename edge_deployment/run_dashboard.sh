#!/usr/bin/env bash
# ==============================================================================
# CARE Wind Turbine Prognostics Dashboard Launcher for Raspberry Pi (Linux/ARM)
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "======================================================================"
echo "  CARE WIND TURBINE PROGNOSTICS • RASPBERRY PI EDGE RUNTIME"
echo "======================================================================"

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "[!] Python 3 not found. Please run: sudo apt update && sudo apt install -y python3 python3-pip python3-venv"
    exit 1
fi

# Create virtualenv if not exists
if [ ! -d ".venv" ]; then
    echo "[*] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install requirements if torch not installed
if ! python3 -c "import torch" &> /dev/null; then
    echo "[*] Installing edge runtime dependencies (torch, numpy, pandas, flask)..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "[*] Launching Prognostics Dashboard on port 5000..."
echo "[*] Access locally: http://localhost:5000"
echo "[*] Access from other devices on Wi-Fi: http://$(hostname -I | cut -d' ' -f1):5000"
echo "======================================================================"

python3 app.py
