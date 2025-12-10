#!/bin/bash

# ==========================
# CONFIG
# ==========================
PROJECT_DIR="$(pwd)"
VENV_NAME="harmone_env"
VENV_PATH="$PROJECT_DIR/$VENV_NAME/bin/activate"

echo "=============================="
echo "  HarmonE: Setup + Launch"
echo "=============================="

# ---------------------------
# Detect a usable terminal
# ---------------------------
detect_terminal() {
    if command -v gnome-terminal &>/dev/null; then
        echo "gnome-terminal"
    elif command -v konsole &>/dev/null; then
        echo "konsole"
    elif command -v xfce4-terminal &>/dev/null; then
        echo "xfce4-terminal"
    elif command -v tilix &>/dev/null; then
        echo "tilix"
    elif command -v xterm &>/dev/null; then
        echo "xterm"
    else
        echo ""
    fi
}

TERMINAL=$(detect_terminal)

if [[ -z "$TERMINAL" ]]; then
    echo "No supported terminal found! Install xterm or GNOME terminal."
    exit 1
else
    echo "[✔] Using terminal: $TERMINAL"
fi

# Helper to launch commands in whichever terminal the user has
launch_terminal() {
    CMD="$1"

    case "$TERMINAL" in
        gnome-terminal)
            gnome-terminal -- bash -c "$CMD; exec bash"
            ;;
        konsole)
            konsole -e bash -c "$CMD; exec bash"
            ;;
        xfce4-terminal)
            xfce4-terminal --hold -e "bash -c '$CMD; exec bash'"
            ;;
        tilix)
            tilix -e "bash -c '$CMD; exec bash'"
            ;;
        xterm)
            xterm -hold -e "bash -c '$CMD; exec bash'"
            ;;
    esac
}

# ---------------------------
# Step 1: Environment Setup
# ---------------------------
echo "[1] Navigating to project directory..."
cd "$PROJECT_DIR" || { echo "Directory not found!"; exit 1; }

echo "[2] Creating virtual environment..."
if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv "$VENV_NAME"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo "[3] Activating environment..."
source "$VENV_PATH"

echo "[4] Installing dependencies..."
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

# ---------------------------
# Step 2: PyRAPL Permissions
# ---------------------------
echo "[5] Setting PyRAPL energy permissions..."
sudo chmod -R 777 /sys/class/powercap/intel-rapl/ 2>/dev/null
echo "[✔] Energy permissions applied."

# ---------------------------
# Step 3: Launching Terminals
# ---------------------------

echo "[6] Launching ACP Server..."
launch_terminal "
cd $PROJECT_DIR;
source $VENV_PATH;
python3 app.py
"

echo "[7] Launching Dashboard..."
launch_terminal "
cd $PROJECT_DIR/frontend;
python3 -m http.server 8000
"

echo "[8] Opening Managed System Console..."
launch_terminal "
cd $PROJECT_DIR;
source $VENV_PATH;
echo 'Run: python3 run_managed_system.py'
exec bash
"

echo "=================================="
echo "   All systems launched!"
echo "   Dashboard: http://localhost:8000/dashboard.html"
echo "=================================="
