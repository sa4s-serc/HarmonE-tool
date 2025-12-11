
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
# Detect Python interpreter
# ---------------------------
detect_python() {
    for cmd in python3 python python3.12 python3.11; do
        if command -v "$cmd" &>/dev/null; then
            echo "$cmd"
            return
        fi
    done
    echo ""
}

PYTHON_CMD=$(detect_python)

if [[ -z "$PYTHON_CMD" ]]; then
    echo "No valid Python interpreter found!"
    echo "Install Python 3 first."
    exit 1
else
    echo "[✔] Using Python: $PYTHON_CMD"
fi

# pip command mapped to python -m pip (always available)
PIP_CMD="$PYTHON_CMD -m pip"

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
    echo "No supported terminal found!"
    echo "Install xterm or GNOME Terminal."
    exit 1
else
    echo "[✔] Using terminal: $TERMINAL"
fi

# Helper to launch commands in user’s terminal
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
    $PYTHON_CMD -m venv "$VENV_NAME"
    echo "[✔] Virtual environment created."
else
    echo "[✔] Virtual environment already exists."
fi

echo "[3] Activating environment..."
source "$VENV_PATH"

echo "[4] Installing dependencies..."
$PIP_CMD install -r requirements.txt \
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
$PYTHON_CMD app.py
"

echo "[7] Launching Dashboard..."
launch_terminal "
cd $PROJECT_DIR/frontend;
$PYTHON_CMD -m http.server 8000
"

echo "[8] Opening Managed System Console..."
launch_terminal "
cd $PROJECT_DIR;
source $VENV_PATH;
echo 'Run: $PYTHON_CMD run_managed_system.py'
exec bash
"

echo "=================================="
echo "   All systems launched!"
echo "   Dashboard: http://localhost:8000/dashboard.html"
echo "=================================="
