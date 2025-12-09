import time
import statistics
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import os
import subprocess
import psutil
import shutil

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ACP] - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# --- In-Memory Knowledge Base (K) ---
KNOWLEDGE_BASE = {
    "policies": {},
    "telemetry_data": {},
    "intervention_logs": {}
}

# ============================================================
# -----------------  HELPER FUNCTIONS  ------------------------
# ============================================================

def get_historical_average(policy_id, metric_key):
    """Calculates the historical average for a given metric from the Knowledge Base."""
    if policy_id not in KNOWLEDGE_BASE["telemetry_data"]:
        return None
    
    metric_values = [
        record[metric_key] 
        for record in KNOWLEDGE_BASE["telemetry_data"][policy_id] 
        if metric_key in record
    ]
    
    return statistics.mean(metric_values) if metric_values else None


def analyze_telemetry(policy, metric_value, metric_key):
    """
    Analyzes telemetry against the policy's primary adaptation boundary.
    Only handles the *primary* boundary here. Secondary boundaries are handled separately.
    """
    boundary = policy.get("adaptation_boundary", {})
    condition = boundary.get("condition")
    static_threshold = boundary.get("threshold")
    dynamic_logic_str = boundary.get("dynamic_logic")

    violation = False

    # ---- STATIC THRESHOLD CHECK ----
    if condition == "GREATER_THAN":
        if metric_value > static_threshold:
            logging.info(f"[ANALYZE] Static threshold VIOLATED: {metric_value} > {static_threshold}")
            violation = True
        else:
            logging.info(f"[ANALYZE] Static threshold NOT violated: {metric_value} <= {static_threshold}")
            return False

    elif condition == "LESS_THAN":
        if metric_value < static_threshold:
            logging.info(f"[ANALYZE] Static threshold VIOLATED: {metric_value} < {static_threshold}")
            violation = True
        else:
            logging.info(f"[ANALYZE] Static threshold NOT violated: {metric_value} >= {static_threshold}")
            return False

    else:
        logging.warning(f"[ANALYZE] Unknown condition '{condition}'. No action taken.")
        return False

    # ---- DYNAMIC LOGIC CHECK (OPTIONAL) ----
    if violation and dynamic_logic_str:
        logging.info(f"[ANALYZE] Checking dynamic logic: '{dynamic_logic_str}'")
        avg = get_historical_average(policy['policy_id'], metric_key)

        if avg is not None:
            try:
                # parse "historic_avg * factor"
                factor = float(dynamic_logic_str.split('*')[1].strip())
                dynamic_threshold = avg * factor

                if (condition == "GREATER_THAN" and metric_value > dynamic_threshold) or \
                   (condition == "LESS_THAN" and metric_value < dynamic_threshold):
                    logging.info(f"[ANALYZE] Dynamic condition MET: {metric_value} vs {dynamic_threshold}")
                    return True
                else:
                    logging.info(f"[ANALYZE] Dynamic condition NOT MET: {metric_value} vs {dynamic_threshold}")
                    return False

            except:
                logging.error("[ANALYZE] Error parsing dynamic logic string.")
                return False

        else:
            logging.warning("[ANALYZE] No historical data for dynamic logic. Using static threshold only.")
            return True

    return violation


def plan_and_execute(policy, trigger_value, trigger_metric=None):
    """Executes the given tactic from a policy."""
    tactics = sorted(policy.get("tactics", []), key=lambda t: t['priority'])
    if not tactics:
        logging.warning(f"[PLAN] No tactics found for policy '{policy['policy_id']}'")
        return

    selected_tactic = tactics[0]

    logging.info(f"[PLAN] Tactic selected: '{selected_tactic['tactic_id']}'")

    intervention_record = {
        "timestamp": time.time(),
        "policy_id": policy["policy_id"],
        "tactic_id": selected_tactic["tactic_id"],
        "trigger_value": trigger_value,
        "trigger_metric": trigger_metric or policy.get("quality_attribute", "unknown"),
        "status": "TRIGGERED"
    }

    KNOWLEDGE_BASE["intervention_logs"].setdefault(policy["policy_id"], []).append(intervention_record)

    payload = {
        "tactic_id": selected_tactic["tactic_id"],
        "trigger_value": trigger_value,
        "trigger_timestamp": intervention_record["timestamp"]
    }

    try:
        endpoint = selected_tactic["tactic_endpoint"]
        logging.info(f"[EXECUTE] Posting to {endpoint} with payload {payload}")
        response = requests.post(endpoint, json=payload, timeout=5)

        if response.status_code == 200:
            logging.info(f"[EXECUTE] SUCCESS: {response.json()}")
            intervention_record["status"] = "CONFIRMED_SUCCESS"
        else:
            logging.error(f"[EXECUTE] FAILED: {response.text}")
            intervention_record["status"] = "CONFIRMED_FAILED"

    except Exception as e:
        logging.error(f"[EXECUTE] ERROR during request: {e}")
        intervention_record["status"] = "REQUEST_FAILED"

# ============================================================
# ---- PERIODIC SECONDARY BOUNDARY CHECKER (KL DRIFT) --------
# ============================================================

def periodic_secondary_checks(interval=30):
    """
    Periodically evaluate secondary boundaries (like KL divergence)
    even when the primary metric (score) violates often.
    """
    logging.info(f"[SECONDARY] Starting periodic evaluator every {interval}s")

    while True:
        time.sleep(interval)

        for policy_id, policy in KNOWLEDGE_BASE["policies"].items():

            secondary = policy.get("secondary_boundaries", [])
            if not secondary:
                continue

            telemetry_history = KNOWLEDGE_BASE["telemetry_data"].get(policy_id, [])
            if not telemetry_history:
                continue

            latest = telemetry_history[-1]

            for sec in secondary:
                qa = sec["quality_attribute"]
                if qa not in latest:
                    continue

                value = latest[qa]
                condition = sec["condition"]
                threshold = sec["threshold"]
                tactic_id = sec["tactic_id"]

                violated = (
                    (condition == "GREATER_THAN" and value > threshold)
                    or
                    (condition == "LESS_THAN" and value < threshold)
                )

                if violated:
                    logging.info(f"[SECONDARY] KL DRIFT DETECTED: {qa}={value} {condition} {threshold}")

                    endpoint = policy["tactics"][0]["tactic_endpoint"]

                    drift_policy = {
                        "policy_id": policy_id,
                        "tactics": [
                            {
                                "tactic_id": tactic_id,
                                "priority": 1,
                                "tactic_endpoint": endpoint
                            }
                        ]
                    }

                    plan_and_execute(drift_policy, value, qa)

# ============================================================
# ---------------------- API ENDPOINTS ------------------------
# ============================================================

@app.route('/api/policy', methods=['POST'])
def add_policy():
    policy = request.json
    policy_id = policy["policy_id"]

    KNOWLEDGE_BASE["policies"][policy_id] = policy

    logging.info(f"[KNOWLEDGE] Policy '{policy_id}' added.")
    return jsonify({"message": "Policy added"}), 201


@app.route('/api/telemetry', methods=['POST'])
def receive_telemetry():
    telemetry = request.json
    logging.info(f"[MONITOR] Received telemetry: {telemetry}")

    policy_found = False

    for policy_id, policy in KNOWLEDGE_BASE["policies"].items():

        metric_key = policy.get("quality_attribute")
        if metric_key not in telemetry:
            continue

        policy_found = True

        KNOWLEDGE_BASE["telemetry_data"].setdefault(policy_id, []).append(telemetry)
        logging.info(f"[KNOWLEDGE] Stored telemetry under '{policy_id}'")

        value = telemetry[metric_key]

        # PRIMARY CHECK (score)
        primary_violation = analyze_telemetry(policy, value, metric_key)

        if primary_violation:
            plan_and_execute(policy, value, metric_key)
            continue

        logging.info(f"[ANALYZE] No primary violation for '{policy_id}'")

    if not policy_found:
        KNOWLEDGE_BASE["telemetry_data"].setdefault("unassigned", []).append(telemetry)

    return jsonify({"message": "Telemetry received"}), 200


@app.route('/api/knowledge/<policy_id>', methods=['GET'])
def get_knowledge(policy_id):

    if policy_id == "unassigned":
        return jsonify({
            "policy": {"policy_id": "unassigned"},
            "telemetry_history": KNOWLEDGE_BASE["telemetry_data"].get("unassigned", []),
            "intervention_logs": []
        })

    return jsonify({
        "policy": KNOWLEDGE_BASE["policies"].get(policy_id),
        "telemetry_history": KNOWLEDGE_BASE["telemetry_data"].get(policy_id, []),
        "intervention_logs": KNOWLEDGE_BASE["intervention_logs"].get(policy_id, [])
    })


@app.route('/api/write-approach', methods=['POST'])
def write_approach_config():
    data = request.json
    approach = data.get("approach")

    # First, stop any running managed system processes
    try:
        stop_managed_system()
        time.sleep(2)  # Give processes time to terminate
    except Exception as e:
        logging.warning(f"Error during cleanup: {e}")

    mapping = {
        'reg_harmone_score': 'reg_harmone',
        'reg_switch_r2': 'reg_switch', 
        'reg_single': 'reg_single',
        'cv_harmone_score': 'cv_harmone',
        'cv_switch_conf': 'cv_switch',
        'cv_single': 'cv_single'
    }

    config_val = mapping.get(approach, approach)

    with open("approach.conf", "w") as f:
        f.write(config_val)

    # Clear knowledge base when switching approaches
    global KNOWLEDGE_BASE
    KNOWLEDGE_BASE = {
        "policies": {},
        "telemetry_data": {}, 
        "intervention_logs": {}
    }

    logging.info(f"Approach updated: {config_val}, knowledge base cleared")
    return jsonify({"message": "Approach written and system cleaned"}), 200


@app.route('/api/save-policy', methods=['POST'])
def save_policy_file():
    policy = request.json
    pid = policy["policy_id"]

    os.makedirs("policies", exist_ok=True)
    with open(f"policies/{pid}.json", "w") as f:
        import json
        json.dump(policy, f, indent=2)

    return jsonify({"message": "Policy saved"}), 200


@app.route('/api/set-model', methods=['POST'])
def set_model():
    data = request.json
    model = data["model"]
    system = data["system"]

    if system == "regression":
        path = "managed_system_regression/knowledge/model.csv"
    else:
        path = "managed_system_cv/knowledge/model.csv"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        f.write(model)

    return jsonify({"message": "Model set"}), 200

# --- NEW: Reset Endpoint to flush data on approach switch ---
@app.route('/api/reset', methods=['POST'])
def reset_knowledge():
    """Clears the in-memory knowledge base to start fresh."""
    global KNOWLEDGE_BASE
    KNOWLEDGE_BASE = {
        "policies": {},
        "telemetry_data": {},
        "intervention_logs": {}
    }
    logging.info("[KNOWLEDGE] Knowledge base RESET requested by client.")
    return jsonify({"message": "Knowledge base reset."}), 200


@app.route('/')
def home():
    return "Welcome to ACP Server!", 200


@app.route('/favicon.ico')
def favicon():
    return "", 204

@app.route("/api/start-managed-system", methods=["POST"])
def start_managed_system():
    try:
        subprocess.Popen(["python3", "run_managed_system.py"])
        return jsonify({"status": "ok", "message": "Managed system started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/stop-managed-system", methods=["POST"])
def stop_managed_system():
    """Stop all managed system processes before switching approaches."""
    try:
        # Send shutdown signal to the managed system wrapper
        response = requests.post("http://localhost:8080/adaptor/shutdown", timeout=10)
        
        # Also kill any remaining processes using system commands
        current_pid = os.getpid()
        
        for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
            try:
                if proc.info['pid'] == current_pid:
                    continue
                    
                if proc.info['name'] in ['python', 'python3'] and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if ('run_managed_system.py' in cmdline or 
                        'inference.py' in cmdline or 
                        'manage.py' in cmdline or
                        'managed_system_cv' in cmdline or 
                        'managed_system_regression' in cmdline):
                        logging.info(f"Terminating process PID {proc.info['pid']}: {cmdline}")
                        proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        logging.info("[CLEANUP] Managed system processes terminated")
        return jsonify({"status": "ok", "message": "Managed system stopped"})
    except Exception as e:
        logging.error(f"[CLEANUP] Error stopping managed system: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/upload-custom-mape', methods=['POST'])
def upload_custom_mape():
    try:
        # 1. Get Base System (Regression or CV)
        base_system = request.form.get('base_system')
        if not base_system:
            return jsonify({"error": "Base system not specified"}), 400

        # 2. Define Paths
        if base_system == 'regression':
            source_dir = "managed_system_regression"
            # We treat the custom run as a variation of regression for the wrapper
            approach_conf_content = "custom_regression" 
        elif base_system == 'cv':
            source_dir = "managed_system_cv"
            approach_conf_content = "custom_cv"
        else:
            return jsonify({"error": "Invalid base system"}), 400

        target_dir = "managed_system_custom"

        # 3. Clean and Re-create Custom Directory
        # We copy the ENTIRE base directory first (inference.py, models, knowledge, etc.)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        shutil.copytree(source_dir, target_dir)
        logging.info(f"[CUSTOM] Copied base system '{source_dir}' to '{target_dir}'")

        # 4. Overwrite with Uploaded MAPE Files
        if 'files[]' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        uploaded_files = request.files.getlist('files[]')
        
        # Allowed files to overwrite in the logic folder
        allowed_files = ['monitor.py', 'analyse.py', 'plan.py', 'execute.py', 'manage.py']
        
        # Ensure mape_logic directory exists in target (it should from copytree)
        mape_logic_path = os.path.join(target_dir, "mape_logic")
        os.makedirs(mape_logic_path, exist_ok=True)

        count = 0
        for file in uploaded_files:
            if file.filename in allowed_files:
                # Save to the root of the custom folder or mape_logic depending on your structure
                # Based on your previous files, monitor/plan/etc seem to be in 'mape_logic' or root?
                # Looking at run_managed_system.py: monitor is imported from logic_path/mape_logic/monitor.py
                
                # So we save into managed_system_custom/mape_logic/
                save_path = os.path.join(mape_logic_path, file.filename)
                file.save(save_path)
                logging.info(f"[CUSTOM] Overwrote {file.filename}")
                count += 1
            else:
                logging.warning(f"[CUSTOM] Skipped unauthorized file: {file.filename}")

        # 5. Update approach.conf
        with open("approach.conf", "w") as f:
            f.write(approach_conf_content)

        # 6. Reset Knowledge Base
        global KNOWLEDGE_BASE
        KNOWLEDGE_BASE = {
            "policies": {},
            "telemetry_data": {},
            "intervention_logs": {}
        }

        return jsonify({"message": f"Custom system built with {count} uploaded files.", "approach": approach_conf_content}), 200

    except Exception as e:
        logging.error(f"[CUSTOM] Error building system: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================
# -------------------------- MAIN -----------------------------
# ============================================================

if __name__ == "__main__":
    # Start periodic drift monitor
    threading.Thread(
        target=periodic_secondary_checks,
        args=(30,),
        daemon=True
    ).start()

    app.run(host="0.0.0.0", port=5000)