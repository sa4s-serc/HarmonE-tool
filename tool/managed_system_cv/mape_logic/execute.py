import os
import shutil
import time
import re
import json
import csv
import logging # <-- Added logging

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
event_log_file = os.path.join(KNOWLEDGE_DIR, "event_log.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")

# --- IMPORT ALL THREE planners ---
from plan import plan_mape, plan_drift, plan_simple_switch

models_dir = "models"

# --- (Helper functions: load_mape_info, save_mape_info, get_last_prediction_line, log_event are all fine and unchanged) ---

def load_mape_info():
    """Loads the mape_info JSON file, handling potential errors."""
    try:
        with open(mape_info_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "last_line": 0,
            "current_energy_threshold": 0.6,
            "ema_scores": {"yolo_n": 0.5, "yolo_s": 0.5, "yolo_m": 0.5},
            "recovery_cycles": 0
        }

def save_mape_info(data):
    """Saves data to the mape_info JSON file."""
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def get_last_prediction_line():
    try:
        with open(predictions_file, "r") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def log_event(event_type, model=None, version=None, details=None):
    last_line = get_last_prediction_line()
    log_entry = {
        "event_type": event_type,
        "last_line": last_line,
        "model": model or "",
        "version": version or "",
        "details": details or ""
    }
    file_exists = os.path.isfile(event_log_file)
    with open(event_log_file, "a", newline="") as csvfile:
        fieldnames = ["event_type", "last_line", "model", "version", "details"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

# --- MODIFIED FUNCTION ---
def execute_mape(trigger="local"):
    """Execute a model switch based on the MAPE plan."""
    print("[MAPE-EXEC] Planning model switch...")
    # --- Pass the trigger down to the planner ---
    decision = plan_mape(trigger=trigger)
    if not decision:
        print("[MAPE-EXEC] No model switch needed.")
        return

    print(f"[MAPE-EXEC] Executing switch to model: {decision.upper()}")
    with open(model_file, "w") as f:
        f.write(decision)
    print(f"⚡ Switched active model to {decision.upper()}")
    log_event("switch", model=decision)

# --- MODIFIED FUNCTION ---
def execute_drift(trigger="local"):
    """Execute the drift response: switch to a previous version or trigger retraining."""
    print("[DRIFT-EXEC] Planning drift response...")
    # --- Pass the trigger down to the planner ---
    decision = plan_drift(trigger=trigger)
    if not decision:
        print("[DRIFT-EXEC] No drift action needed.")
        return

    action = decision.get("action")
    print(f"[DRIFT-EXEC] Drift action planned: {action}")

    if action == "switch_version":
        version_path = decision["version_path"]
        if not os.path.exists(version_path):
            print(f"[DRIFT-EXEC] Error: Version path '{version_path}' does not exist. Cannot switch.")
            return

        # ... (rest of the switch_version logic is unchanged) ...
        base_name_match = re.search(r'(yolo_[nsm])', os.path.basename(version_path))
        if not base_name_match:
            print(f"[DRIFT-EXEC] Error: Could not determine base model name from '{version_path}'.")
            return
        base_name = base_name_match.group(1)
        destination_path = os.path.join(models_dir, f"{base_name}.pt")
        shutil.copy(version_path, destination_path)
        print(f"[DRIFT-EXEC] Copied '{version_path}' to '{destination_path}'.")
        with open(model_file, "w") as f:
            f.write(base_name)
        print(f" Switched active model to version: {os.path.basename(version_path)}")
        log_event("vmr", model=base_name, version=os.path.basename(version_path), details=f"Switched to versioned model at {version_path}")
        
        print(f"[DRIFT-EXEC] Inflating EMA score for {base_name.upper()} to ensure stability...")
        mape_info = load_mape_info()
        current_score = mape_info["ema_scores"].get(base_name, 0.5)
        new_score = min(1.0, current_score + 0.1)
        mape_info["ema_scores"][base_name] = new_score
        save_mape_info(mape_info)
        print(f"[DRIFT-EXEC] EMA score for {base_name.upper()} updated from {current_score:.4f} to {new_score:.4f}.")
        time.sleep(20)  # Reduced simulation time

    elif action == "retrain":
        print("[DRIFT-EXEC] Triggering retraining...")
        # Note: os.system() is simple but blocks. In a real system, you might
        # use subprocess.Popen() to run this in the background.
        # For this design, blocking is acceptable as it's a single-threaded listener.
        os.system("python retrain.py") 
        log_event("retrain", details="Retraining triggered by drift detection.")
        time.sleep(20) # Reduced simulation time

# --- NEW FUNCTION ---
def execute_simple_switch(trigger="local"):
    """
    Executes a simple model switch based on the R2 baseline plan.
    'trigger' is accepted for consistency but not used.
    """
    logging.info("Executing Simple Switch (R² Baseline)...")
    
    # Call the simple planner
    decision = plan_simple_switch() 
    
    if not decision:
        logging.info("EXECUTE (Simple Switch): No action needed.")
        return

    logging.info(f"⚡ EXECUTE (Simple Switch): Switching model to {decision.upper()} in {model_file}")
    try:
        with open(model_file, "w") as f:
            f.write(decision)
        logging.info("EXECUTE (Simple Switch): Model switch successful.")
        log_event("switch", model=decision, details="R2_baseline_switch")
    except Exception as e:
        logging.error(f"EXECUTE (Simple Switch): Failed to write to {model_file}: {e}")