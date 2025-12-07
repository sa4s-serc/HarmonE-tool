import os
import shutil
import time
import re
import json
import csv
import logging
import pyRAPL

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

# Initialize PyRAPL for MAPE-K energy monitoring
pyRAPL.setup()

def load_mape_info():
    """Loads the mape_info JSON file with event counters and energy tracking."""
    try:
        with open(mape_info_file, "r") as f:
            info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        info = {
            "last_line": 0,
            "current_energy_threshold": 0.6,
            "ema_scores": {"yolo_n": 0.5, "yolo_s": 0.5, "yolo_m": 0.5},
            "recovery_cycles": 0
        }
    
    # Ensure event counters exist
    if "event_counters" not in info:
        info["event_counters"] = {
            "model_switches": 0,
            "retrains": 0,
            "vmr_events": 0,
            "mape_k_energy_uJ": 0.0
        }
    
    # Ensure simple switch counters exist (separate from MAPE counters)
    if "simple_switch_counters" not in info:
        info["simple_switch_counters"] = {
            "simple_switches": 0
        }
    
    return info

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

def record_event(event_type, energy_consumed=0.0, details=None):
    """Record an event and update counters."""
    info = load_mape_info()
    
    # Update counters
    if event_type == "switch":
        info["event_counters"]["model_switches"] += 1
        logging.info(f"📊 Event recorded: Model switch #{info['event_counters']['model_switches']}")
    elif event_type == "retrain":
        info["event_counters"]["retrains"] += 1
        logging.info(f"📊 Event recorded: Retrain #{info['event_counters']['retrains']}")
    elif event_type == "vmr":
        info["event_counters"]["vmr_events"] += 1
        logging.info(f"📊 Event recorded: VMR event #{info['event_counters']['vmr_events']}")
    
    # Add MAPE-K energy consumption
    info["event_counters"]["mape_k_energy_uJ"] += energy_consumed
    
    if details:
        logging.info(f"📊 Event details: {details}")
    if energy_consumed > 0:
        logging.info(f"⚡ MAPE-K energy consumed: {energy_consumed:.2f} µJ (Total: {info['event_counters']['mape_k_energy_uJ']:.2f} µJ)")
    
    save_mape_info(info)

def record_simple_switch():
    """Record a simple switch event (no energy tracking, just count)."""
    info = load_mape_info()
    
    # Update simple switch counter
    info["simple_switch_counters"]["simple_switches"] += 1
    
    logging.info(f"📊 Simple switch recorded: #{info['simple_switch_counters']['simple_switches']}")
    
    save_mape_info(info)

def execute_mape(trigger="local"):
    """Execute a model switch based on the MAPE plan."""
    # Start energy monitoring for MAPE-K loop
    energy_meter = pyRAPL.Measurement("mape_k_cv_execution")
    energy_meter.begin()
    
    print("[MAPE-EXEC] Planning model switch...")
    
    # Pass the trigger down to the planner
    decision = plan_mape(trigger=trigger)
    
    if not decision:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("switch", energy_consumed, "No switch needed - planning returned no decision")
        print("[MAPE-EXEC] No model switch needed.")
        return

    print(f"[MAPE-EXEC] Executing switch to model: {decision.upper()}")
    
    try:
        # Get current model for logging
        try:
            with open(model_file, "r") as f:
                old_model = f.read().strip()
        except FileNotFoundError:
            old_model = "unknown"
        
        with open(model_file, "w") as f:
            f.write(decision)
        
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        
        # Record both old CSV log and new event counter
        log_event("switch", model=decision)
        record_event("switch", energy_consumed, f"Model switched from {old_model} to {decision}")
        
        print(f"⚡ Switched active model to {decision.upper()}")
        
    except Exception as e:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("switch", energy_consumed, f"Failed to write model file: {e}")
        print(f"[MAPE-EXEC] Error switching model: {e}")

def execute_drift(trigger="local"):
    """Execute the drift response: switch to a previous version or trigger retraining."""
    # Start energy monitoring for MAPE-K loop
    energy_meter = pyRAPL.Measurement("mape_k_cv_drift_execution")
    energy_meter.begin()
    
    print("[DRIFT-EXEC] Planning drift response...")
    
    # Pass the trigger down to the planner
    decision = plan_drift(trigger=trigger)
    
    if not decision:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("vmr", energy_consumed, "No drift action needed")
        print("[DRIFT-EXEC] No drift action needed.")
        return

    action = decision.get("action")
    print(f"[DRIFT-EXEC] Drift action planned: {action}")

    if action == "switch_version":
        version_path = decision["version_path"]
        if not os.path.exists(version_path):
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("vmr", energy_consumed, f"Version path does not exist: {version_path}")
            print(f"[DRIFT-EXEC] Error: Version path '{version_path}' does not exist. Cannot switch.")
            return

        base_name_match = re.search(r'(yolo_[nsm])', os.path.basename(version_path))
        if not base_name_match:
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("vmr", energy_consumed, f"Could not determine base model name from {version_path}")
            print(f"[DRIFT-EXEC] Error: Could not determine base model name from '{version_path}'.")
            return
        
        base_name = base_name_match.group(1)
        destination_path = os.path.join(models_dir, f"{base_name}.pt")
        
        try:
            shutil.copy(version_path, destination_path)
            print(f"[DRIFT-EXEC] Copied '{version_path}' to '{destination_path}'.")
            
            with open(model_file, "w") as f:
                f.write(base_name)
            
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            
            # Record both old CSV log and new event counter
            log_event("vmr", model=base_name, version=os.path.basename(version_path), 
                     details=f"Switched to versioned model at {version_path}")
            record_event("vmr", energy_consumed, f"VMR: Switched to version {version_path}")
            
            print(f"⚡ Switched active model to version: {os.path.basename(version_path)}")
            
            # Inflate EMA score for stability
            print(f"[DRIFT-EXEC] Inflating EMA score for {base_name.upper()} to ensure stability...")
            mape_info = load_mape_info()
            current_score = mape_info["ema_scores"].get(base_name, 0.5)
            new_score = min(1.0, current_score + 0.1)
            mape_info["ema_scores"][base_name] = new_score
            save_mape_info(mape_info)
            print(f"[DRIFT-EXEC] EMA score for {base_name.upper()} updated from {current_score:.4f} to {new_score:.4f}.")
            time.sleep(20)  # Reduced simulation time
            
        except Exception as e:
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("vmr", energy_consumed, f"Failed to copy versioned model: {e}")
            print(f"[DRIFT-EXEC] Error copying versioned model: {e}")

    elif action == "retrain":
        print("[DRIFT-EXEC] Triggering retraining...")
        try:
            # Note: os.system() is simple but blocks. In a real system, you might
            # use subprocess.Popen() to run this in the background.
            os.system("python retrain.py")
            
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            
            # Record both old CSV log and new event counter
            log_event("retrain", details="Retraining triggered by drift detection.")
            record_event("retrain", energy_consumed, "Model retrained due to drift")
            
            time.sleep(20) # Reduced simulation time
            
        except Exception as e:
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("retrain", energy_consumed, f"Retraining failed: {e}")
            print(f"[DRIFT-EXEC] Error during retraining: {e}")

def execute_simple_switch(trigger="local"):
    """Executes a simple model switch based on the confidence baseline plan."""
    logging.info("Executing Simple Switch (Confidence Baseline)...")
    
    # Call the simple planner
    decision = plan_simple_switch() 
    
    if not decision:
        logging.info("EXECUTE (Simple Switch): No action needed.")
        return

    logging.info(f"⚡ EXECUTE (Simple Switch): Switching model to {decision.upper()} in {model_file}")
    
    try:
        # Get current model for logging
        try:
            with open(model_file, "r") as f:
                old_model = f.read().strip()
        except FileNotFoundError:
            old_model = "unknown"
        
        with open(model_file, "w") as f:
            f.write(decision)
        
        # Record both old CSV log and new simple switch counter
        log_event("switch", model=decision, details="confidence_baseline_switch")
        record_simple_switch()
        
        logging.info(f"📊 Simple switch: {old_model} → {decision}")
        logging.info("EXECUTE (Simple Switch): Model switch successful.")
        
    except Exception as e:
        logging.error(f"EXECUTE (Simple Switch): Failed to write to {model_file}: {e}")