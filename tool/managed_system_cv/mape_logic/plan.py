# research/sustainable-mlops/HarmonEXT/mape/plan.py
import json
import random
import logging # <-- Good to add logging
import os
from analyse import analyse_mape, analyse_drift

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")
MODEL_FILE = os.path.join(KNOWLEDGE_DIR, "model.csv")

thresholds_file = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")

MODELS = ["yolo_n", "yolo_s", "yolo_m"]
# --- ADD THIS HELPER FUNCTION ---
def get_current_model():
    """Fetch the currently active model from knowledge."""
    try:
        with open(MODEL_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "yolo_s" # Default to a CV model
    
def load_mape_info():
    """Load stored MAPE info including model-specific EMA scores."""
    try:
        with open(mape_info_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Provide a default structure if the file is missing or corrupt
        return {"ema_scores": {m: 0.5 for m in MODELS}}

# --- MODIFIED FUNCTION ---
def plan_mape(trigger="local"):
    """
    Decide on a model switch based on performance analysis, energy constraints,
    and an exploratory strategy.
    
    If trigger == 'acp', it bypasses local analysis.
    """
    # 1. Exploratory Action (Alpha-based random switching)
    thresholds = json.load(open(thresholds_file))
    alpha = thresholds.get("alpha", 0.1)
    if random.random() < alpha:
        chosen = random.choice(MODELS)
        print(f"[MAPE-PLAN] Random switch triggered by alpha. Chosen: {chosen.upper()}")
        return chosen

    # --- THIS IS THE KEY CHANGE ---
    threshold_violated = None
    if trigger == "local":
        # 2. Analyze current performance
        print("[MAPE-PLAN] Running in 'local' mode. Calling local analyse_mape()...")
        analysis = analyse_mape()
        if not analysis or not analysis["switch_needed"]:
            print("[MAPE-PLAN] No switch needed based on analysis (thresholds not violated).")
            return None
        threshold_violated = analysis["threshold_violated"]
    else: # trigger == "acp"
        print("[MAPE-PLAN] Running in 'acp' mode. Bypassing local analysis.")
        # When triggered by ACP, we assume a violation occurred.
        # We'll default to "score" logic (switch to best model)
        # as the ACP doesn't tell us *why* it triggered.
        threshold_violated = "score"
    # --- END OF CHANGE ---

    # 3. Plan action based on the reason for the switch
    mape_info = load_mape_info()
    ema_scores = mape_info["ema_scores"]

    try:
        with open(model_file, "r") as f:
            current_model = f.read().strip()
    except FileNotFoundError:
        current_model = None

    chosen_model = None
    if threshold_violated == "energy":
        # If energy is the issue, find the best-scoring model that is NOT the current one
        best_alternatives = sorted(ema_scores.items(), key=lambda item: item[1], reverse=True)
        chosen_model = next((model for model, score in best_alternatives if model != current_model), None)
        if chosen_model:
            print(f"[MAPE-PLAN] Energy threshold violated. Switching to best alternative: {chosen_model.upper()}")
        else:
            print(f"[MAPE-PLAN] Energy threshold violated, but no alternative models available.")
            return None
    else: # "score" or default "acp" trigger
        # For score violations, switch to the model with the absolute highest score
        chosen_model = max(ema_scores, key=ema_scores.get)
        print(f"[MAPE-PLAN] Score/ACP trigger. Switching to best overall model: {chosen_model.upper()}")

    # 4. Final check to prevent redundant switching
    if chosen_model == current_model:
        print(f"[MAPE-PLAN] Chosen model ({chosen_model.upper()}) is already active. No switch needed.")
        return None

    print(f"[MAPE-PLAN] Planning to switch to: {chosen_model.upper()}")
    return chosen_model

# --- MODIFIED FUNCTION ---
def plan_drift(trigger="local"):
    """
    Decide if retraining is needed or if an existing version can be used.
    
    If trigger == 'acp', it bypasses local analysis and assumes drift.
    """
    drift_analysis = None
    
    # --- THIS IS THE KEY CHANGE ---
    if trigger == "local":
        print("[DRIFT-PLAN] Running in 'local' mode. Calling local analyse_drift()...")
        drift_analysis = analyse_drift()
        if not drift_analysis or not drift_analysis["drift_detected"]:
            print("[DRIFT-PLAN] No drift detected. No action required.")
            return None
    else: # trigger == "acp"
        print("[DRIFT-PLAN] Running in 'acp' mode. Bypassing local analysis.")
        # When triggered by ACP, we must assume drift was detected
        # and run the full analysis to find the best version or retrain.
        # So we call analyse_drift() anyway, but skip the first check.
        drift_analysis = analyse_drift()
        if not drift_analysis:
            print("[DRIFT-PLAN] ACP triggered drift, but analysis failed. No action.")
            return None
    # --- END OF CHANGE ---

    # The analysis already determined the best action (switch_version or retrain)
    if drift_analysis.get("action") == "switch_version":
        version_path = drift_analysis["best_version"]
        print(f"[DRIFT-PLAN] Planning to switch to a better-suited previous version: {version_path}")
        return {"action": "switch_version", "version_path": version_path}
    else: # action == "retrain" or default
        print("[DRIFT-PLAN] Drift detected and no suitable version found. Planning to retrain.")
        return {"action": "retrain"}

def plan_simple_switch():
    """
    A simple baseline plan: just switch to a different model.
    Picks one of the *other* available models at random.
    """
    logging.info("PLAN (Simple Switch): Triggered by R² baseline.")
    current_model = get_current_model()
    available_models = ["yolo_n", "yolo_s", "yolo_m"]

    # Remove the current model from the list
    available_models.remove(current_model)

    # Randomly pick from the remaining two
    chosen_model = random.choice(available_models)

    logging.info(f"PLAN (Simple Switch): Switching from '{current_model.upper()}' to '{chosen_model.upper()}'.")
    return chosen_model