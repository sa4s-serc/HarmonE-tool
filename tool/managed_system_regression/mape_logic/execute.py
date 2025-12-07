import os
import shutil
import logging
import sys
import json
import pyRAPL
from plan import plan_mape, plan_drift, plan_simple_switch

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

MODEL_FILE = os.path.join(KNOWLEDGE_DIR, "model.csv")
MAPE_INFO_FILE = os.path.join(KNOWLEDGE_DIR, "mape_info.json")

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [execute.py] - %(levelname)-8s - %(message)s',
    stream=sys.stdout
)

# Initialize PyRAPL for MAPE-K energy monitoring
pyRAPL.setup()

def load_mape_info():
    """Load MAPE info with event counters and energy tracking."""
    try:
        with open(MAPE_INFO_FILE, "r") as f:
            info = json.load(f)
    except FileNotFoundError:
        info = {}
    
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

def save_mape_info(info):
    """Save updated MAPE info including event counters."""
    with open(MAPE_INFO_FILE, "w") as f:
        json.dump(info, f, indent=4)

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

# --- Tactic Execution ---
def execute_mape(trigger="local"):
    """Switch to the best model based on planning."""
    # Start energy monitoring for MAPE-K loop
    energy_meter = pyRAPL.Measurement("mape_k_execution")
    energy_meter.begin()
    
    logging.info("Executing MAPE (model switch)...")
    
    # Call plan_mape with the trigger
    decision = plan_mape(trigger=trigger)
    
    if not decision:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("switch", energy_consumed, "No switch needed - planning returned no decision")
        logging.info("EXECUTE: No action needed (plan was empty).")
        return

    logging.info(f"⚡ EXECUTE: Switching model to {decision.upper()} in {MODEL_FILE}")
    try:
        # Get current model for logging
        try:
            with open(MODEL_FILE, "r") as f:
                old_model = f.read().strip()
        except FileNotFoundError:
            old_model = "unknown"
        
        with open(MODEL_FILE, "w") as f:
            f.write(decision)
        
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        
        # Record the switch event
        record_event("switch", energy_consumed, f"Model switched from {old_model} to {decision}")
        
        logging.info("EXECUTE: Model switch successful.")
    except Exception as e:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("switch", energy_consumed, f"Failed to write model file: {e}")
        logging.error(f"EXECUTE: Failed to write to {MODEL_FILE}: {e}")

def execute_drift(trigger="local"):
    """Replaces model with best version or retrains if necessary."""
    # Start energy monitoring for MAPE-K loop
    energy_meter = pyRAPL.Measurement("mape_k_drift_execution")
    energy_meter.begin()
    
    logging.info("Executing Drift handling...")
    
    decision = plan_drift(trigger=trigger)
    
    if not decision:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("vmr", energy_consumed, "No drift action needed")
        logging.info("EXECUTE (Drift): No action needed.")
        return

    if decision["action"] == "replace":
        best_version_path = decision["version"]
        # Basic validation
        if not best_version_path or "version" not in best_version_path:
             energy_meter.end()
             energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
             record_event("vmr", energy_consumed, f"Invalid version path: {best_version_path}")
             logging.warning(f"EXECUTE (Drift): Invalid version path provided: {best_version_path}")
             return

        model_name = os.path.basename(os.path.dirname(best_version_path))
        model_extension = ".pkl" if model_name in ["linear", "svm"] else ".pth"
        model_target_path = os.path.join(BASE_DIR, "..", "models", f"{model_name}{model_extension}")
        
        try:
            shutil.copy(best_version_path, model_target_path)
            
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            
            # Record VMR event
            record_event("vmr", energy_consumed, f"VMR: Switched to version {best_version_path}")
            
            logging.info(f"✔ EXECUTE (Drift): Switched to lower KL divergence model: {best_version_path}")
        except Exception as e:
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("vmr", energy_consumed, f"Failed to copy model: {e}")
            logging.error(f"EXECUTE (Drift): Failed to copy model: {e}")

    elif decision["action"] == "retrain":
        logging.info("🚀 EXECUTE (Drift): Triggering retraining...")
        try:
            # Ensure retrain.py exists
            if os.path.exists("retrain.py"):
                os.system("python retrain.py") 
                
                energy_meter.end()
                energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
                
                # Record retrain event
                record_event("retrain", energy_consumed, "Model retrained due to drift")
                
                logging.info("EXECUTE (Drift): Retraining script finished.")
            else:
                energy_meter.end()
                energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
                record_event("retrain", energy_consumed, "Retrain.py not found")
                logging.warning("EXECUTE (Drift): 'retrain.py' not found. Skipping.")
        except Exception as e:
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("retrain", energy_consumed, f"Retraining failed: {e}")
            logging.error(f"EXECUTE (Drift): Retraining failed: {e}")

def execute_simple_switch(trigger="local"):
    """Switches model based on the simple_switch plan."""
    logging.info("Executing Simple Switch (R² Baseline)...")
    
    decision = plan_simple_switch(trigger=trigger) # Call the new planner
    
    if not decision:
        logging.info("EXECUTE (Simple Switch): No action needed.")
        return

    logging.info(f"⚡ EXECUTE (Simple Switch): Switching model to {decision.upper()} in {MODEL_FILE}")
    try:
        # Get current model for logging
        try:
            with open(MODEL_FILE, "r") as f:
                old_model = f.read().strip()
        except FileNotFoundError:
            old_model = "unknown"
        
        with open(MODEL_FILE, "w") as f:
            f.write(decision)
        
        # Record the simple switch event (no energy tracking)
        record_simple_switch()
        
        logging.info(f"📊 Simple switch: {old_model} → {decision}")
        logging.info("EXECUTE (Simple Switch): Model switch successful.")
    except Exception as e:
        logging.error(f"EXECUTE (Simple Switch): Failed to write to {MODEL_FILE}: {e}")