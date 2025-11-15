import os
import shutil
import logging
import sys
from plan import plan_mape, plan_drift, plan_simple_switch

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [execute.py] - %(levelname)-8s - %(message)s',
    stream=sys.stdout
)

MODEL_FILE = "knowledge/model.csv"

# --- Tactic Execution ---
def execute_mape(trigger="local"):
    """Switch to the best model based on planning."""
    logging.info("Executing MAPE (model switch)...")
    
    # Call plan_mape with the trigger
    decision = plan_mape(trigger=trigger)
    
    if not decision:
        logging.info("EXECUTE: No action needed (plan was empty).")
        return

    logging.info(f"⚡ EXECUTE: Switching model to {decision.upper()} in {MODEL_FILE}")
    try:
        with open(MODEL_FILE, "w") as f:
            f.write(decision)
        logging.info("EXECUTE: Model switch successful.")
    except Exception as e:
        logging.error(f"EXECUTE: Failed to write to {MODEL_FILE}: {e}")

def execute_drift(trigger="local"):
    """Replaces model with best version or retrains if necessary."""
    logging.info("Executing Drift handling...")
    
    decision = plan_drift(trigger=trigger)
    
    if not decision:
        logging.info("EXECUTE (Drift): No action needed.")
        return

    if decision["action"] == "replace":
        best_version_path = decision["version"]
        # Basic validation
        if not best_version_path or "version" not in best_version_path:
             logging.warning(f"EXECUTE (Drift): Invalid version path provided: {best_version_path}")
             return

        model_name = os.path.basename(os.path.dirname(best_version_path))
        model_extension = ".pkl" if model_name in ["linear", "svm"] else ".pth"
        model_target_path = os.path.join("models", f"{model_name}{model_extension}")
        
        try:
            shutil.copy(best_version_path, model_target_path)
            logging.info(f"✔ EXECUTE (Drift): Switched to lower KL divergence model: {best_version_path}")
        except Exception as e:
            logging.error(f"EXECUTE (Drift): Failed to copy model: {e}")

    elif decision["action"] == "retrain":
        logging.info("🚀 EXECUTE (Drift): Triggering retraining...")
        try:
            # Ensure retrain.py exists
            if os.path.exists("retrain.py"):
                os.system("python retrain.py") 
                logging.info("EXECUTE (Drift): Retraining script finished.")
            else:
                logging.warning("EXECUTE (Drift): 'retrain.py' not found. Skipping.")
        except Exception as e:
            logging.error(f"EXECUTE (Drift): Retraining failed: {e}")

def execute_simple_switch():
    """Switches model based on the simple_switch plan."""
    logging.info("Executing Simple Switch (R² Baseline)...")
    
    decision = plan_simple_switch() # Call the new planner
    
    if not decision:
        logging.info("EXECUTE (Simple Switch): No action needed.")
        return

    logging.info(f"⚡ EXECUTE (Simple Switch): Switching model to {decision.upper()} in {MODEL_FILE}")
    try:
        with open(MODEL_FILE, "w") as f:
            f.write(decision)
        logging.info("EXECUTE (Simple Switch): Model switch successful.")
    except Exception as e:
        logging.error(f"EXECUTE (Simple Switch): Failed to write to {MODEL_FILE}: {e}")