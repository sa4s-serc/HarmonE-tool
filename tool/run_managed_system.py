import time
import requests
import threading
import logging
import subprocess
import os
import json
import importlib.util
import sys
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MasterWrapper] - %(levelname)s - %(message)s')

# --- Configuration ---
ACP_SERVER_URL = "http://localhost:5000"
APPROACH_CONFIG_FILE = "approach.conf"
POLICY_DIR = "policies"
HANDLER_PORT = 8080

# --- Dynamically set logic paths ---
LOGIC_PATH = ""
COMMAND_FILE_PATH = ""
monitor_mape = None
monitor_drift = None

# --- Adaptation Handler API (Listens for commands from ACP) ---
handler_app = Flask(__name__)

@handler_app.route('/adaptor/tactic', methods=['POST'])
def execute_tactic_from_acp():
    """Receives a command from the ACP and writes it to the correct command file."""
    data = request.json
    tactic_id = data.get("tactic_id")
    logging.info(f"[ACP_Handler] Command received: '{tactic_id}'")
    try:
        # COMMAND_FILE_PATH is set dynamically in __main__
        with open(COMMAND_FILE_PATH, "w") as f:
            f.write(tactic_id)
        logging.info(f"[ACP_Handler] Command '{tactic_id}' queued in {COMMAND_FILE_PATH}.")
        return jsonify({"message": "Command queued."}), 200
    except Exception as e:
        logging.error(f"[ACP_Handler] Failed to write command file: {e}")
        return jsonify({"error": "Failed to queue command"}), 500

def run_handler_api():
    handler_app.run(host='0.0.0.0', port=HANDLER_PORT)

# --- Telemetry & Policy Functions ---
def push_telemetry():
    """Dynamically pushes telemetry from the correct monitor."""
    global monitor_mape, monitor_drift
    if not monitor_mape and not monitor_drift:
        logging.critical("Monitors not imported. Exiting telemetry thread.")
        return

    logging.info(f"[Monitor] Telemetry thread started.")
    time.sleep(10) # Initial delay
    
    while True:
        try:
            telemetry_payload = {"timestamp": time.time()}
            
            # Dynamically call the correct monitor
            if monitor_mape:
                mape_metrics = monitor_mape()
                if mape_metrics:
                    telemetry_payload.update(mape_metrics)

            if monitor_drift:
                drift_metrics = monitor_drift()
                if drift_metrics:
                    telemetry_payload.update(drift_metrics)

            if len(telemetry_payload) > 1: # More than just timestamp
                logging.info(f"[Monitor] Pushing telemetry: {telemetry_payload}")
                requests.post(f"{ACP_SERVER_URL}/api/telemetry", json=telemetry_payload, timeout=3)
            else:
                logging.info("[Monitor] No new data from monitors.")

        except Exception as e:
            logging.error(f"[Monitor] Error in telemetry loop: {e}", exc_info=True)
        
        time.sleep(15)

def register_policies_with_acp(policy_prefix):
    """
    Registers policies from the POLICY_DIR that match the prefix
    (e.g., 'cv_harmone' prefix will load 'cv_harmone_score.json').
    """
    # if 'single' in policy_prefix:
    #     logging.info("Running in 'single' mode. No policies will be registered.")
    #     return True

    try:
        policy_files = [
            f for f in os.listdir(POLICY_DIR) 
            if f.endswith('.json') and f.startswith(policy_prefix)
        ]
    except FileNotFoundError:
        logging.error(f"FATAL: Policy directory '{POLICY_DIR}' not found.")
        return False
        
    if not policy_files:
        logging.warning(f"No policies found in '{POLICY_DIR}' with prefix '{policy_prefix}'.")
        return True # Not fatal

    logging.info(f"Found {len(policy_files)} policies to register: {policy_files}")
    
    for policy_file in policy_files:
        try:
            policy_path = os.path.join(POLICY_DIR, policy_file)
            with open(policy_path, 'r') as f:
                policy = json.load(f)
            
            policy_id = policy.get("policy_id")
            if not policy_id:
                logging.error(f"'{policy_file}' is missing 'policy_id'. Skipping.")
                continue

            requests.post(f"{ACP_SERVER_URL}/api/policy", json=policy, timeout=3)
            logging.info(f"Policy '{policy_id}' from '{policy_file}' registered.")
        
        except requests.exceptions.RequestException as e:
            logging.critical(f"FATAL: Could not connect to ACP at {ACP_SERVER_URL}.")
            return False
        except Exception as e:
            logging.error(f"FATAL: Error registering policy '{policy_file}': {e}")
            return False
            
    return True

def import_monitor_from_path(logic_path):
    """Helper function to dynamically import the monitor module."""
    global monitor_mape, monitor_drift
    try:
        monitor_path = os.path.join(logic_path, "mape_logic", "monitor.py")
        spec = importlib.util.spec_from_file_location("monitor", monitor_path)
        monitor_module = importlib.util.module_from_spec(spec)
        
        sys.path.insert(0, logic_path) # Add to path so monitor's internal imports work
        spec.loader.exec_module(monitor_module)
        sys.path.pop(0) # Clean up sys.path
        
        # We need to handle if a monitor doesn't exist (e.g. no drift)
        monitor_mape = getattr(monitor_module, "monitor_mape", None)
        monitor_drift = getattr(monitor_module, "monitor_drift", None)
        
        if not monitor_mape:
             logging.warning(f"Could not find 'monitor_mape' in {monitor_path}")
        if not monitor_drift:
             logging.warning(f"Could not find 'monitor_drift' in {monitor_path}")
             
    except Exception as e:
        logging.critical(f"FATAL: Could not import monitors from '{monitor_path}': {e}")
        exit(1)

# --- Main Execution Logic ---
if __name__ == '__main__':
    # 1. Read Master Configuration
    try:
        with open(APPROACH_CONFIG_FILE, 'r') as f:
            approach = f.read().strip().lower()
    except FileNotFoundError:
        logging.critical(f"FATAL: '{APPROACH_CONFIG_FILE}' not found. Please create it.")
        exit(1)

    # 2. Parse Configuration (e.g., "cv_harmone")
    if not '_' in approach:
        logging.critical(f"FATAL: Invalid approach '{approach}'. Must be format 'system_mode' (e.g., 'cv_harmone').")
        exit(1)
        
    system_type, run_mode = approach.split('_', 1)
    
    if system_type == "cv":
        LOGIC_PATH = "managed_system_cv"
    elif system_type == "reg":
        LOGIC_PATH = "managed_system_regression"
    else:
        logging.critical(f"FATAL: Unknown system_type '{system_type}'. Must be 'cv' or 'reg'.")
        exit(1)
    
    logging.info(f"--- Running System: '{system_type.upper()}' in Mode: '{run_mode.upper()}' ---")
    
    # 3. Dynamically import the correct logic
    import_monitor_from_path(LOGIC_PATH)

    # 4. Set up knowledge path and write the *local* config for manage.py
    KNOWLEDGE_PATH = os.path.join(LOGIC_PATH, "knowledge")
    LOCAL_APPROACH_CONFIG = os.path.join(LOGIC_PATH, "approach.conf")
    COMMAND_FILE_PATH = os.path.join(KNOWLEDGE_PATH, "command.txt") # Set global var

    os.makedirs(KNOWLEDGE_PATH, exist_ok=True)
    with open(LOCAL_APPROACH_CONFIG, "w") as f:
        # Pass the correct mode to the local manage.py
        f.write(f"{run_mode}_acp") # e.g., "harmone_acp", "switch_acp"
    
    # 5. Start Background Threads
    threading.Thread(target=push_telemetry, daemon=True).start()

    # 6. Start Subprocesses from the correct logic path
    subprocesses = []
    try:
        inference_cmd = ["python", "-u", "inference.py"]
        manage_cmd = ["python", "-u", "mape_logic/manage.py"]

        # Pass the logic path as a working directory so all file paths are correct
        subprocesses.append(subprocess.Popen(inference_cmd, cwd=LOGIC_PATH))
        logging.info(f"Inference engine '{inference_cmd[2]}' started in '{LOGIC_PATH}'.")
        
        if 'single' not in run_mode:
            subprocesses.append(subprocess.Popen(manage_cmd, cwd=LOGIC_PATH))
            logging.info(f"MAPE logic '{manage_cmd[2]}' started in '{LOGIC_PATH}'.")
        else:
            logging.info("Running in 'single' (monitor-only) mode. 'manage.py' will not be started.")

    except Exception as e:
        logging.critical(f"Failed to start subprocesses: {e}")
        exit(1)

    # 7. Handle ACP-specific setup
    policy_prefix = f"{system_type}_{run_mode}" # e.g., "cv_harmone"
    
    if not register_policies_with_acp(policy_prefix):
        for p in subprocesses: p.terminate()
        exit(1)
    
    if 'single' not in run_mode:
        threading.Thread(target=run_handler_api, daemon=True).start()
        logging.info(f"Adaptation Handler API listening on http://0.0.0.0:{HANDLER_PORT}...")

    # 8. Wait for processes to finish
    try:
        logging.info("Wrapper is running. (Press Ctrl+C to stop)")
        for p in subprocesses: p.wait()
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Terminating subprocesses...")
        for p in subprocesses: p.terminate()
    finally:
        logging.info("Wrapper script finished.")