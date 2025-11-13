import time
import requests
import threading
import logging
import subprocess
import os
import json
from flask import Flask, request, jsonify

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Wrapper] - %(levelname)s - %(message)s')

# --- Configuration ---
ACP_SERVER_URL = "http://localhost:5000" # Base URL for ACP server
APPROACH_CONFIG_FILE = "approach.conf"
MODEL_CONTROL_FILE_PATH = "knowledge/model.csv"
COMMAND_FILE_PATH = os.path.join("knowledge", "command.txt")
HANDLER_PORT = 8080
TELEMETRY_INTERVAL_SECONDS = 15 # How often to push telemetry

POLICY_DIR = "policies" 
# --- Import HarmonE's monitor ---
try:
    # We import the refactored monitor that now returns model_used
    from mape_logic.monitor import monitor_mape, monitor_drift
except ImportError:
    logging.critical("Could not import mape_logic.monitor. Ensure it exists and is correct.")
    exit(1)
except Exception as e:
    logging.critical(f"An error occurred importing monitor: {e}")
    exit(1)


# --- Adaptation Handler API (Listens for commands from ACP) ---
handler_app = Flask(__name__)

@handler_app.route('/adaptor/tactic', methods=['POST'])
def execute_tactic_from_acp():
    """Receives a command from the ACP and writes it to the command file for the local loop to execute."""
    data = request.json
    tactic_id = data.get("tactic_id")
    
    # --- ADDED FOR DEMO ---
    logging.info("\n" + "="*70)
    logging.info(f"   [ACP COMMAND RECEIVED] - TACTIC: {tactic_id}")
    logging.info("="*70 + "\n")
    # --- END DEMO ADD ---

    logging.info(f"[ACP_Handler] Command received from ACP: '{tactic_id}'")
    try:
        with open(COMMAND_FILE_PATH, "w") as f:
            f.write(tactic_id)
        logging.info(f"[ACP_Handler] Command '{tactic_id}' queued for local execution in {COMMAND_FILE_PATH}.")
        return jsonify({"message": "Command received and queued."}), 200
    except Exception as e:
        logging.error(f"[ACP_Handler] Failed to write command file: {e}")
        return jsonify({"error": "Failed to queue command"}), 500

def run_handler_api():
    """Runs the Flask app to listen for commands."""
    try:
        handler_app.run(host='0.0.0.0', port=HANDLER_PORT)
    except Exception as e:
        logging.critical(f"Could not start handler API: {e}")

# --- Telemetry & Policy Functions ---
def push_telemetry():
    """
    Uses HarmonE's monitor to calculate metrics and sends them to the ACP.
    This is the refactored function that sends the full payload.
    """
    logging.info(f"[Monitor] Telemetry thread started. Will push data every {TELEMETRY_INTERVAL_SECONDS}s.")
    time.sleep(10) # Wait for inference.py to generate some initial data
    
    while True:
        try:
            # 1. Get metrics from local monitor
            mape_metrics = monitor_mape()
            
            if mape_metrics:
                # 2. Add timestamp
                telemetry_payload = {
                    "timestamp": time.time(),
                    **mape_metrics  # <-- This now includes "model_used"
                }

                # 3. Add drift (if available)
                drift_metrics = monitor_drift()
                if drift_metrics:
                    telemetry_payload.update(drift_metrics)

                # 4. Push to ACP
                logging.info(f"[Monitor] Pushing telemetry: score={telemetry_payload.get('score', 'N/A'):.2f}, model={telemetry_payload.get('model_used', 'N/A')}")
                requests.post(f"{ACP_SERVER_URL}/api/telemetry", json=telemetry_payload, timeout=3)
            
            else:
                logging.info("[Monitor] monitor_mape() returned no new data.")

        except Exception as e:
            logging.error(f"[Monitor] Error in telemetry loop: {e}", exc_info=True)
        
        time.sleep(TELEMETRY_INTERVAL_SECONDS)

# def register_policy_with_acp():
#     """Loads the local policy.json and sends it to the ACP."""
#     try:
#         with open('policy.json', 'r') as f:
#             policy = json.load(f)
        
#         policy_id = policy.get("policy_id")
#         if not policy_id:
#             logging.error("policy.json is missing 'policy_id'.")
#             return False

#         requests.post(f"{ACP_SERVER_URL}/api/policy", json=policy, timeout=3)
#         logging.info(f"Policy '{policy_id}' successfully registered with ACP.")
#         return True
#     except FileNotFoundError:
#         logging.error("FATAL: 'policy.json' not found. Cannot register policy.")
#         return False
#     except requests.exceptions.RequestException as e:
#         logging.critical(f"FATAL: Could not connect to ACP at {ACP_SERVER_URL}. Is server running? Error: {e}")
#         return False
#     except Exception as e:
#         logging.error(f"FATAL: Error registering policy: {e}")
#         return False

def register_policies_with_acp(): # <-- Renamed function
    """Loads all local .json policies from the POLICY_DIR and sends them to the ACP."""
    policy_files = [f for f in os.listdir(POLICY_DIR) if f.endswith('.json')]
    
    if not policy_files:
        logging.error(f"FATAL: No .json policies found in '{POLICY_DIR}'. Cannot register policies.")
        return False
        
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
            logging.info(f"Policy '{policy_id}' from '{policy_file}' successfully registered with ACP.")
        
        except requests.exceptions.RequestException as e:
            logging.critical(f"FATAL: Could not connect to ACP at {ACP_SERVER_URL}. Is server running? Error: {e}")
            return False
        except Exception as e:
            logging.error(f"FATAL: Error registering policy '{policy_file}': {e}")
            return False
            
    return True

# --- Main Execution Logic ---
if __name__ == '__main__':
    # --- 1. Read Configuration ---
    try:
        with open(APPROACH_CONFIG_FILE, 'r') as f:
            approach = f.read().strip().lower()
    except FileNotFoundError:
        approach = "monitor_only"
        logging.warning(f"'{APPROACH_CONFIG_FILE}' not found. Defaulting to '{approach}'.")
    logging.info(f"--- Running in '{approach.upper()}' mode ---")

    # --- 2. Clean/Initialize Knowledge State ---
    with open(MODEL_CONTROL_FILE_PATH, "w") as f: f.write("lstm") # Always start with LSTM
    if os.path.exists(COMMAND_FILE_PATH): os.remove(COMMAND_FILE_PATH) # Clear any old commands
    # Note: We do NOT clear mape_info.json, as it holds valuable EMA scores
    
    # --- 3. Start Background Threads ---
    
    # Start the telemetry push thread (runs in all modes except 'none')
    if approach != "none":
        threading.Thread(target=push_telemetry, daemon=True).start()

    # --- 4. Start Subprocesses based on mode ---
    subprocesses = []
    
    # Start the inference engine (runs in all modes except 'none')
    if approach != "none":
        try:
            inference_process = subprocess.Popen(["python", "-u", "inference.py"])
            subprocesses.append(inference_process)
            logging.info("Inference engine (inference.py) started.")
        except Exception as e:
            logging.critical(f"Failed to start inference.py: {e}")
            exit(1)

    # Start the local MAPE logic (runs in local and acp modes)
    if approach in ["harmone_local", "harmone_acp"]:
        try:
            mape_process = subprocess.Popen(["python", "-u", "mape_logic/manage.py"])
            subprocesses.append(mape_process)
            logging.info("MAPE logic (manage.py) started.")
        except Exception as e:
            logging.critical(f"Failed to start manage.py: {e}")
            exit(1)

    # --- 5. Handle ACP-specific setup ---
    # if approach == "harmone_acp":
    #     # Register the policy with the server
    #     if not register_policy_with_acp():
    #         # If policy registration fails, we must stop.
    #         for p in subprocesses: p.terminate()
    #         exit(1)
        
    #     # Start the API handler to listen for commands
    #     threading.Thread(target=run_handler_api, daemon=True).start()
    #     logging.info(f"Adaptation Handler API listening on http://0.0.0.0:{HANDLER_PORT}...")
    if approach == "harmone_acp":
        # Create policies directory if it doesn't exist
        if not os.path.exists(POLICY_DIR):
            os.makedirs(POLICY_DIR)
            logging.warning(f"'{POLICY_DIR}' directory not found. Created it. Please add your policy .json files there.")

        # Register ALL policies with the server
        if not register_policies_with_acp(): # <-- Call the new function
            # If policy registration fails, we must stop.
            for p in subprocesses: p.terminate()
            exit(1)
        
        # Start the API handler to listen for commands
        threading.Thread(target=run_handler_api, daemon=True).start()
        logging.info(f"Adaptation Handler API listening on http://0.0.0.0:{HANDLER_PORT}...")

    # --- 6. Wait for processes to finish ---
    try:
        logging.info("Wrapper is running. Waiting for subprocesses. (Press Ctrl+C to stop)")
        for p in subprocesses: 
            p.wait() # Wait for each subprocess to complete
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Terminating subprocesses...")
        for p in subprocesses: 
            p.terminate()
    except Exception as e:
        logging.error(f"An unexpected error occurred in main loop: {e}")
    finally:
        logging.info("Wrapper script finished.")

