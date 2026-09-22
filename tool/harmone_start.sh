#!/usr/bin/env bash

# ==========================
# Harmonica: one-shot setup + launch.
#
# Lives in tool/ and is meant to be run from there (./harmone_start.sh),
# same as before. The one thing it now reaches outside tool/ is the small
# sample dataset/images checked into ../examples/ - REPO_DIR below is that
# one level up, kept separate from PROJECT_DIR (tool/) so the two are never
# confused.
#
# Idempotent: safe to re-run. Every step checks whether its output already
# exists before doing anything, so re-running after a fresh `git pull`
# usually does nothing at all.
# ==========================
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$PROJECT_DIR")"
VENV_NAME="harmone_env"
VENV_PATH="$PROJECT_DIR/$VENV_NAME/bin/activate"
VENV_PYTHON="$PROJECT_DIR/$VENV_NAME/bin/python"
VENV_MLFLOW="$PROJECT_DIR/$VENV_NAME/bin/mlflow"
LOG_DIR="$PROJECT_DIR/.harmone_logs"

cd "$PROJECT_DIR" || { echo "Directory not found: $PROJECT_DIR"; exit 1; }

echo "=============================="
echo "  Harmonica: Setup + Launch"
echo "=============================="

# ---------------------------
# Detect a Python interpreter to CREATE the venv with. Once the venv exists,
# every step below calls $VENV_PYTHON directly (its absolute path) rather
# than re-deriving an interpreter from PATH - a bare name like "python3.11"
# can silently resolve to a *different* interpreter than the one the venv
# was built with (e.g. if the venv is 3.10-based but 3.11 is also on PATH),
# which sends pip installing into that other interpreter's site-packages
# instead of the venv. Generic names first, exact version names last, since
# every venv guarantees `python`/`python3` but not `python3.11` specifically.
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
    echo "No Python 3 interpreter found. Install Python 3.8+ first."
    exit 1
fi
echo "[✔] Using Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"

# ---------------------------
# Step 1: venv + dependencies
# ---------------------------
echo
echo "[1/7] Virtual environment..."
if [ ! -d "$VENV_NAME" ]; then
    $PYTHON_CMD -m venv "$VENV_NAME"
    echo "  [✔] Created $VENV_NAME"
else
    echo "  [✔] $VENV_NAME already exists"
fi

echo
echo "[2/7] Installing dependencies (first run can take a few minutes - PyTorch/Ultralytics are large)..."
"$VENV_PYTHON" -m pip install --upgrade pip -q
"$VENV_PYTHON" -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# ---------------------------
# Step 2: RAPL energy-counter permissions (Linux only, best-effort)
# ---------------------------
echo
echo "[3/7] Energy monitoring permissions..."
RAPL_DIR="/sys/class/powercap/intel-rapl"
if [ -d "$RAPL_DIR" ]; then
    if [ -r "$RAPL_DIR/intel-rapl:0/energy_uj" ] 2>/dev/null; then
        echo "  [✔] RAPL energy counters already readable"
    else
        echo "  Requesting sudo to make RAPL energy counters readable (real hardware"
        echo "  energy readings instead of the CPU-utilization estimate)."
        echo "  You'll be prompted for YOUR password below, in this terminal - it's"
        echo "  your own login password, not a project password:"
        # No stderr redirect here: sudo's password prompt itself goes straight to
        # /dev/tty (not through stdout/stderr), so it shows up either way - but
        # if sudo fails for some OTHER reason (no controlling terminal at all,
        # an askpass helper missing, etc.), that real reason must stay visible
        # instead of being swallowed, or "declined/no sudo" below is misleading.
        if sudo chmod -R a+r "$RAPL_DIR"; then
            echo "  [✔] Done"
        else
            echo "  [!] Could not set RAPL permissions (see any sudo error above,"
            echo "      or you declined/have no sudo access) - falling back to the"
            echo "      CPU-utilization energy estimate. The system still runs fine."
        fi
    fi
else
    echo "  [i] No $RAPL_DIR on this machine (not Linux, or no RAPL support)."
    echo "      Energy will use the CPU-utilization estimate - this is expected"
    echo "      on Windows/macOS and is not an error."
fi

# ---------------------------
# Step 3: seed sample datasets (idempotent - only copies what's missing).
# All destinations here are inside tool/ (i.e. relative to $PROJECT_DIR,
# where this script already `cd`'d); sources live one level up in
# $REPO_DIR/examples/.
# ---------------------------
echo
echo "[4/7] Sample datasets..."

REG_KNOWLEDGE="managed_system_regression/knowledge"
REG_TRAIN_DATA="managed_system_regression/data/pems"
mkdir -p "$REG_KNOWLEDGE" "$REG_TRAIN_DATA"

if [ ! -f "$REG_KNOWLEDGE/dataset.csv" ]; then
    if cp "$REPO_DIR/examples/HarmonE/dataset.csv" "$REG_KNOWLEDGE/dataset.csv"; then
        echo "  [✔] Seeded regression inference stream (knowledge/dataset.csv)"
    else
        echo "  [!] Could not seed knowledge/dataset.csv - regression will fail to start"
    fi
else
    echo "  [✔] Regression inference stream already present"
fi

if [ ! -f "$REG_TRAIN_DATA/flow_data_train.csv" ]; then
    if cp "$REPO_DIR/examples/HarmonE/dataset.csv" "$REG_TRAIN_DATA/flow_data_train.csv"; then
        echo "  [✔] Seeded regression training sample (data/pems/flow_data_train.csv)"
    else
        echo "  [!] Could not seed data/pems/flow_data_train.csv - initial training will fail"
    fi
else
    echo "  [✔] Regression training sample already present"
fi

CV_TEST_DIR="managed_system_cv/data/bdd100k/images/test"
mkdir -p "$CV_TEST_DIR"
if [ -z "$(ls -A "$CV_TEST_DIR" 2>/dev/null)" ]; then
    if cp "$REPO_DIR"/examples/cv_sample_images/*.jpg "$CV_TEST_DIR/" 2>/dev/null; then
        echo "  [✔] Seeded $(ls "$CV_TEST_DIR" | wc -l) CV demo images (data/bdd100k/images/test/)"
    else
        echo "  [!] Could not seed CV demo images - CV approaches will have nothing to run inference on"
    fi
else
    echo "  [✔] CV demo images already present ($(ls "$CV_TEST_DIR" | wc -l) files)"
fi

# ---------------------------
# Step 4: seed per-system MAPE-loop config (idempotent)
# ---------------------------
echo
echo "[5/7] MAPE loop config (thresholds.json / mape_info.json / model.csv)..."
for SYS in regression cv; do
    SEED_DIR="managed_system_${SYS}/knowledge_seed"
    DEST_DIR="managed_system_${SYS}/knowledge"
    mkdir -p "$DEST_DIR"
    for f in thresholds.json mape_info.json model.csv; do
        if [ ! -f "$DEST_DIR/$f" ]; then
            if cp "$SEED_DIR/$f" "$DEST_DIR/$f"; then
                echo "  [✔] Seeded $SYS/knowledge/$f"
            else
                echo "  [!] Could not seed $SYS/knowledge/$f"
            fi
        fi
    done
done
echo "  [✔] Done"

# ---------------------------
# Step 5: base models (train / download only if missing)
# ---------------------------
echo
echo "[6/7] Model weights..."

REG_MODELS="managed_system_regression/models"
if [ ! -f "$REG_MODELS/lstm.pth" ] || [ ! -f "$REG_MODELS/svm.pkl" ] || [ ! -f "$REG_MODELS/linear.pkl" ]; then
    echo "  Training initial LSTM/SVM/Linear regression models (~2 min on CPU)..."
    if (cd managed_system_regression && "$VENV_PYTHON" train.py); then
        echo "  [✔] Regression models trained"
    else
        echo "  [!] Regression model training failed - see output above. The"
        echo "      dashboard's regression approaches won't produce telemetry"
        echo "      until managed_system_regression/models/ has lstm.pth,"
        echo "      svm.pkl and linear.pkl (rerun this script to retry)."
    fi
else
    echo "  [✔] Regression models already present"
fi

CV_MODELS="managed_system_cv/models"
if [ ! -f "$CV_MODELS/yolo_n.pt" ] || [ ! -f "$CV_MODELS/yolo_s.pt" ] || [ ! -f "$CV_MODELS/yolo_m.pt" ]; then
    echo "  Downloading YOLOv8 base weights (~80MB total, needs internet)..."
    if (cd managed_system_cv && "$VENV_PYTHON" utility/get_models.py); then
        echo "  [✔] YOLO weights downloaded"
    else
        echo "  [!] YOLO weight download failed - check your internet connection."
        echo "      The CV approaches won't produce telemetry until"
        echo "      managed_system_cv/models/ has yolo_n.pt, yolo_s.pt and"
        echo "      yolo_m.pt (rerun this script to retry)."
    fi
else
    echo "  [✔] CV models already present"
fi

# Default approach, only if nothing has picked one yet - the dashboard
# overwrites this the moment someone selects an approach; this is purely so
# `python run_managed_system.py` has something valid on a bare clone.
[ -f approach.conf ] || echo "reg_harmone" > approach.conf

# ---------------------------
# Step 6: launch servers
# ---------------------------
echo
echo "[7/7] Launching servers..."
mkdir -p "$LOG_DIR"

detect_terminal() {
    for t in gnome-terminal konsole xfce4-terminal tilix xterm; do
        if command -v "$t" &>/dev/null; then
            echo "$t"
            return
        fi
    done
    echo ""
}
TERMINAL=$(detect_terminal)

# Launches $2 in its own terminal window when one is available, or as a
# background process logging to .harmone_logs/$1.log otherwise (headless
# servers, CI, SSH sessions without X, etc. - the original script just gave
# up here with "No supported terminal found!").
launch() {
    local NAME="$1" CMD="$2"
    if [[ -n "$TERMINAL" ]]; then
        case "$TERMINAL" in
            gnome-terminal) gnome-terminal --title="$NAME" -- bash -c "$CMD; exec bash" ;;
            konsole)        konsole -e bash -c "$CMD; exec bash" ;;
            xfce4-terminal) xfce4-terminal --title="$NAME" --hold -e "bash -c '$CMD; exec bash'" ;;
            tilix)          tilix -e "bash -c '$CMD; exec bash'" ;;
            xterm)          xterm -T "$NAME" -hold -e "bash -c '$CMD; exec bash'" ;;
        esac
    else
        nohup bash -c "$CMD" >"$LOG_DIR/$NAME.log" 2>&1 &
        echo "  [✔] $NAME -> background (pid $!), log: $LOG_DIR/$NAME.log"
    fi
}

if [[ -n "$TERMINAL" ]]; then
    echo "  [✔] Using terminal: $TERMINAL (one window per server)"
else
    echo "  [i] No GUI terminal emulator found - running servers in the background instead."
fi

# Absolute venv binaries throughout, not `source activate; bare-command` -
# a freshly spawned terminal/subshell doesn't inherit this script's sourced
# venv, so a bare `python`/`mlflow` there falls back to whatever's on the
# system PATH instead (the same class of bug this script's own pip install
# above avoids by calling "$VENV_PYTHON" directly).
launch "ACP-Server" "cd '$PROJECT_DIR'; '$VENV_PYTHON' app.py"
sleep 2
launch "MLflow-UI" "cd '$PROJECT_DIR'; '$VENV_MLFLOW' ui --backend-store-uri sqlite:///mlflow.db --port 5001"
launch "Dashboard" "cd '$PROJECT_DIR/frontend'; '$VENV_PYTHON' -m http.server 8000"

echo
echo "=================================="
echo "  All set!"
echo "  Dashboard  : http://localhost:8000/dashboard.html"
echo "  MLflow UI  : http://localhost:5001"
echo "  ACP server : http://localhost:5000"
echo
echo "  Pick an approach on the dashboard and click 'Start Managed System' -"
echo "  it launches run_managed_system.py for you, no manual step needed."
if [[ -z "$TERMINAL" ]]; then
echo
echo "  No GUI terminal was found, so servers are running in the background."
echo "  Logs: $LOG_DIR/*.log"
echo "  Stop them with:"
echo "    pkill -f 'tool/app.py'; pkill -f 'mlflow ui'; pkill -f 'http.server 8000'"
fi
echo "=================================="
