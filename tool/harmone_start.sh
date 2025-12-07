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
# Step 1: Environment Setup
# ---------------------------
echo "[1] Navigating to project directory..."
cd "$PROJECT_DIR" || { echo "Directory not found!"; exit 1; }

echo "[2] Creating virtual environment (if not exists)..."
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
echo "[5] Setting PyRAPL energy permissions (requires sudo)..."
sudo chmod 777 /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj 2>/dev/null
sudo chmod 777 /sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj 2>/dev/null
sudo chmod 777 /sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:1/energy_uj 2>/dev/null
sudo chmod -R 777 /sys/class/powercap/intel-rapl/ 2>/dev/null

echo "[✔] Energy permissions applied."

# ---------------------------
# Step 3: Launching Terminals
# ---------------------------

echo "[6] Launching ACP Server + Manager..."
gnome-terminal -- bash -c "
    cd $PROJECT_DIR;
    source $VENV_PATH;
    echo 'Starting app.py...';
    python3 app.py;
    exec bash
"

echo "[7] Launching Dashboard..."
gnome-terminal -- bash -c "
    cd $PROJECT_DIR/frontend;
    echo 'Starting dashboard on port 8000...';
    python3 -m http.server 8000;
    exec bash
"

echo "[8] Preparing Managed System Terminal..."
gnome-terminal -- bash -c "
    cd $PROJECT_DIR;
    source $VENV_PATH;
    echo 'Managed System Console Ready.';
    echo 'Start after selecting policy in UI:';
    echo '    python3 run_managed_system.py';
    exec bash
"

echo "=================================="
echo "   All systems launched!"
echo "   Dashboard: http://localhost:8000/acp_dashboard.html"
echo "   ACP: Running on ports 5000 (API) & 8080 (Manager)"
echo "=================================="
