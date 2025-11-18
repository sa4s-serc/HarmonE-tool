import time
import statistics
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- IMPORT THE NEW LIBRARY
import requests
import logging

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ACP] - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # <-- ENABLE CORS FOR YOUR ENTIRE APP

# --- In-Memory Knowledge Base (K) ---
KNOWLEDGE_BASE = {
    "policies": {},
    "telemetry_data": {},
    "intervention_logs": {}
}

# --- Helper Functions ---
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
    Analyzes incoming telemetry against the policy's adaptation boundary.
    Returns True if a violation is detected, False otherwise.
    """
    boundary = policy.get("adaptation_boundary", {})
    condition = boundary.get("condition")
    static_threshold = boundary.get("threshold")
    dynamic_logic_str = boundary.get("dynamic_logic")

    violation = False
    
    # 1. Static Threshold Check
    if condition == "GREATER_THAN":
        if metric_value > static_threshold:
            logging.info(f"[ANALYZE] Static threshold VIOLATED: {metric_value} > {static_threshold}")
            violation = True
        else:
            logging.info(f"[ANALYZE] Static threshold NOT violated: {metric_value} <= {static_threshold}")
            return False # No violation
    elif condition == "LESS_THAN":
        if metric_value < static_threshold:
            logging.info(f"[ANALYZE] Static threshold VIOLATED: {metric_value} < {static_threshold}")
            violation = True
        else:
            logging.info(f"[ANALYZE] Static threshold NOT violated: {metric_value} >= {static_threshold}")
            return False # No violation
    else:
        logging.warning(f"[ANALYZE] Unknown condition '{condition}'. No action taken.")
        return False

    # 2. Dynamic Logic Check (if static threshold was met)
    if violation and dynamic_logic_str:
        logging.info(f"[ANALYZE] Checking dynamic logic: '{dynamic_logic_str}'")
        avg = get_historical_average(policy['policy_id'], metric_key)
        if avg is not None:
            # A simple, safe evaluation of the dynamic logic string
            if "historic_avg" in dynamic_logic_str and "*" in dynamic_logic_str:
                try:
                    factor = float(dynamic_logic_str.split('*')[1].strip())
                    dynamic_threshold = avg * factor
                    if (condition == "GREATER_THAN" and metric_value > dynamic_threshold) or \
                       (condition == "LESS_THAN" and metric_value < dynamic_threshold):
                        logging.info(f"[ANALYZE] Dynamic logic condition MET: {metric_value} vs {dynamic_threshold:.2f} (avg: {avg:.2f})")
                        return True # Confirmed violation
                    else:
                        logging.info(f"[ANALYZE] Dynamic logic condition NOT MET: {metric_value} vs {dynamic_threshold:.2f} (avg: {avg:.2f})")
                        return False # Retract violation
                except (ValueError, IndexError) as e:
                    logging.error(f"Could not parse dynamic_logic: {e}")
                    return False
        else:
            logging.warning("[ANALYZE] Not enough historical data for dynamic check. Relying on static threshold.")
            return True # Cannot disprove violation, so it stands

    return violation

def plan_and_execute(policy, trigger_value):
    """Selects the highest priority tactic and sends the execution request."""
    # PLAN: Select the highest priority tactic
    tactics = sorted(policy.get("tactics", []), key=lambda t: t['priority'])
    if not tactics:
        logging.warning(f"[PLAN] No tactics found for policy '{policy['policy_id']}'")
        return

    selected_tactic = tactics[0]
    logging.info(f"[PLAN] Tactic selected: '{selected_tactic['tactic_id']}' with priority {selected_tactic['priority']}")

    # EXECUTE: Trigger the adaptation
    # 1. Log the planned intervention
    intervention_record = {
        "timestamp": time.time(),
        "policy_id": policy['policy_id'],
        "tactic_id": selected_tactic['tactic_id'],
        "trigger_value": trigger_value,
        "status": "TRIGGERED"
    }
    KNOWLEDGE_BASE["intervention_logs"].setdefault(policy['policy_id'], []).append(intervention_record)
    logging.info(f"[KNOWLEDGE] Logged planned intervention: {selected_tactic['tactic_id']}")

    # 2. Send the execution request to the managed system
    try:
        payload = {
            "tactic_id": selected_tactic['tactic_id'],
            "trigger_value": trigger_value,
            "trigger_timestamp": intervention_record['timestamp']
        }
        endpoint = selected_tactic['tactic_endpoint']
        
        logging.info(f"[EXECUTE] Sending POST request to endpoint: {endpoint} with payload: {payload}")
        response = requests.post(endpoint, json=payload, timeout=5)
        
        if response.status_code == 200:
            logging.info(f"[EXECUTE] Successfully executed tactic '{selected_tactic['tactic_id']}'. Response: {response.json()}")
            intervention_record["status"] = "CONFIRMED_SUCCESS"
        else:
            logging.error(f"[EXECUTE] Failed to execute tactic. Status: {response.status_code}, Response: {response.text}")
            intervention_record["status"] = "CONFIRMED_FAILED"

    except requests.exceptions.RequestException as e:
        logging.error(f"[EXECUTE] Error during tactic execution request: {e}")
        intervention_record["status"] = "REQUEST_FAILED"

# --- API Endpoints ---
@app.route('/api/policy', methods=['POST'])
def add_policy():
    """Endpoint to upload a new adaptation policy to the Knowledge Base."""
    policy = request.json
    if not policy or "policy_id" not in policy:
        return jsonify({"error": "Invalid policy format"}), 400
    
    policy_id = policy["policy_id"]
    KNOWLEDGE_BASE["policies"][policy_id] = policy
    logging.info(f"[KNOWLEDGE] New policy added/updated: '{policy_id}'")
    return jsonify({"message": f"Policy '{policy_id}' added successfully"}), 201

@app.route('/api/telemetry', methods=['POST'])
def receive_telemetry():
    """
    Main endpoint to receive telemetry data and trigger the MAPE-K loop.
    """
    telemetry = request.json
    logging.info(f"[MONITOR] Received telemetry: {telemetry}")
    
    # KNOWLEDGE: Store incoming telemetry
    # We find the right policy to associate it with
    policy_found = False
    for policy_id, policy in KNOWLEDGE_BASE["policies"].items():
        metric_key = policy.get("quality_attribute")
        if metric_key in telemetry:
            policy_found = True
            
            # Store data
            KNOWLEDGE_BASE["telemetry_data"].setdefault(policy_id, []).append(telemetry)
            logging.info(f"[KNOWLEDGE] Stored telemetry under policy '{policy_id}'")
            
            # ANALYZE: Check if the adaptation boundary is violated
            metric_value = telemetry[metric_key]
            is_violation = analyze_telemetry(policy, metric_value, metric_key)
            
            if is_violation:
                # PLAN & EXECUTE
                plan_and_execute(policy, metric_value)
            else:
                logging.info(f"[ANALYZE] No violation detected for '{policy_id}'. No action needed.")
    
    if not policy_found:
        logging.warning(f"Received telemetry but no matching policy found. Data: {telemetry}")

    return jsonify({"message": "Telemetry received"}), 200

@app.route('/api/knowledge/<policy_id>', methods=['GET'])
def get_knowledge(policy_id):
    """A debug endpoint to view all knowledge associated with a policy."""
    if policy_id not in KNOWLEDGE_BASE["policies"]:
        return jsonify({"error": "Policy not found"}), 404
        
    return jsonify({
        "policy": KNOWLEDGE_BASE["policies"].get(policy_id),
        "telemetry_history": KNOWLEDGE_BASE["telemetry_data"].get(policy_id, []),
        "intervention_history": KNOWLEDGE_BASE["intervention_logs"].get(policy_id, [])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

