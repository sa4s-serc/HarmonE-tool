import threading
import time
import os
import sys
import logging
from execute import execute_mape, execute_drift, execute_simple_switch

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [manage.py] - %(levelname)-8s - %(message)s',
    stream=sys.stdout
)

# File paths
COMMAND_FILE_PATH = "knowledge/command.txt"
APPROACH_CONFIG_FILE = "approach.conf"
LOG_FILE = "knowledge/mape_log.csv"
PREDICTIONS_FILE = "knowledge/predictions.csv"
DRIFT_FILE = "knowledge/drift.csv"

# --- Local Tactic Execution ---
def execute_tactic_locally(tactic_id):
    """Executes the correct local logic based on the tactic_id."""
    logging.info(f"Command '{tactic_id}' received. Triggering local logic...")
    
    if tactic_id == "execute_mape_plan":
        # This calls plan.py with the "acp" trigger
        execute_mape(trigger="acp")
        
    elif tactic_id == "handle_data_drift":
        # This calls plan.py (for drift) and then execute_drift
        execute_drift(trigger="acp")
        
    else:
        logging.warning(f"Unknown local tactic_id: '{tactic_id}'")

# --- Main MAPE Loop ---
def run_mape_loop(approach):
    """The main loop that drives the local MAPE logic."""
    
    if approach == "harmone_local":
        # --- Original HarmonE Logic ---
        logging.info("Running in 'harmone_local' mode. Using internal timer.")
        while True:
            time.sleep(40) # Original 40-second timer
            logging.info("Local timer triggered. Running MAPE plan...")
            execute_mape(trigger="local")
            
            # Add drift check logic here if needed
            # time.sleep(400)
            # execute_drift(trigger="local")

    elif approach in ["harmone_acp", "switch_acp"]:
        # --- ACP-Driven Logic ---
        logging.info("Running in 'harmone_acp' mode. Listening for commands...")
        while True:
            try:
                # Check for the command file created by the wrapper
                if os.path.exists(COMMAND_FILE_PATH):
                    with open(COMMAND_FILE_PATH, 'r') as f:
                        tactic_id = f.read().strip()
                    
                    # Command processed, delete the file so it doesn't run again
                    os.remove(COMMAND_FILE_PATH)
                    
                    if tactic_id:
                        execute_tactic_locally(tactic_id)
                        
            except FileNotFoundError:
                # This is normal, means the file was deleted before we could read it
                pass 
            except Exception as e:
                logging.error(f"Error in ACP command loop: {e}")
                
            time.sleep(5) # Check for a new command every 5 seconds
    else:
        logging.info(f"Mode '{approach}' requires no local MAPE loop. Exiting.")
        
def execute_tactic_locally(tactic_id):
    """Executes the correct local logic based on the tactic_id."""
    logging.info(f"Command '{tactic_id}' received. Triggering local logic...")
    
    if tactic_id == "execute_mape_plan":
        execute_mape(trigger="acp")
        
    elif tactic_id == "handle_data_drift":
        execute_drift(trigger="acp")
    
    # --- ADD THIS ELIF BLOCK ---
    elif tactic_id == "switch_model_r2_baseline":
        execute_simple_switch()
    # --- END OF ADDITION ---
        
    else:
        logging.warning(f"Unknown local tactic_id: '{tactic_id}'")

# --- Startup ---
if __name__ == "__main__":
    try:
        with open(APPROACH_CONFIG_FILE, 'r') as f:
            approach = f.read().strip().lower()
    except FileNotFoundError:
        logging.error(f"'{APPROACH_CONFIG_FILE}' not found. Cannot start.")
        sys.exit(1)
        
    run_mape_loop(approach)

