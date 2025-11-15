import json
import random
import logging
import os
import sys
from analyse import analyse_mape, analyse_drift # analyse_mape is ONLY for local mode

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [plan.py] - %(levelname)-8s - %(message)s',
    stream=sys.stdout
)

# File paths
KNOWLEDGE_DIR = "knowledge"
THRESHOLDS_FILE = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
MODEL_FILE = os.path.join(KNOWLEDGE_DIR, "model.csv")
MAPE_INFO_FILE = os.path.join(KNOWLEDGE_DIR, "mape_info.json")

# --- Helper Functions ---
def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {}

def get_current_model():
    try:
        with open(MODEL_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "lstm" # Default

# --- Main Planning Logic ---
def plan_mape(trigger="local"):
    """
    Decides on the best model to use.
    - trigger='local': Runs full analysis first (original HarmonE).
    - trigger='acp': Skips analysis and proceeds to planning (ACP-driven).
    """
    logging.info(f"PLAN (MAPE) triggered by: {trigger.upper()}")

    # 1. ANALYZE (Only for local mode)
    if trigger == "local":
        logging.info("Running in 'local' mode, performing local analysis...")
        analysis = analyse_mape()
        if not analysis or not analysis["switch_needed"]:
            logging.info("Local analysis: No switch needed.")
            return None
        logging.info("Local analysis: Violation detected, proceeding to plan.")
    
    elif trigger == "acp":
        logging.info("Running in 'acp' mode. ACP detected violation. Proceeding to plan.")
        # We skip the local 'analyse_mape()' because the ACP has already made the decision
    
    else:
        logging.warning(f"Unknown trigger '{trigger}'. Aborting plan.")
        return None

    # 2. PLAN (This logic is now shared by both modes)
    
    # Load local knowledge (thresholds for alpha, ema_scores for planning)
    thresholds = load_json(THRESHOLDS_FILE)
    mape_info = load_json(MAPE_INFO_FILE)
    ema_scores = mape_info.get("ema_scores", {})
    alpha = thresholds.get("alpha", 0.1) # Exploration probability
    
    current_model = get_current_model()

    # Tactic 1: Exploration (using alpha)
    if random.random() < alpha:
        available_models = [m for m in ema_scores.keys() if m != current_model]
        if not available_models:
             logging.warning("PLAN: Exploration tactic: No alternative models to explore.")
             return None
        chosen_model = random.choice(available_models)
        logging.info(f"🎲 PLAN: Exploratory tactic! Randomly selecting '{chosen_model.upper()}'.")
    
    # Tactic 2: Exploitation (choose best alternative)
    else:
        best_alternative = sorted(ema_scores.items(), key=lambda x: x[1], reverse=True)
        # Find the best model that is NOT the current one
        chosen_model = next((m for m, score in best_alternative if m != current_model), None)
        
        if not chosen_model:
            # This happens if all other models have a score of 0 or are not listed
            logging.warning("PLAN: Exploitation tactic: No valid alternatives found. Sticking with current model.")
            return None
            
        logging.info(f"🏆 PLAN: Exploitation tactic: Best alternative to '{current_model.upper()}' is '{chosen_model.upper()}' (Score: {ema_scores.get(chosen_model, 'N/A'):.2f}).")

    # Final check: Don't switch if we're already on the best model
    if chosen_model == current_model:
        logging.info(f"PLAN: Already using the chosen model ('{chosen_model.upper()}'). No switch needed.")
        return None

    return chosen_model

def plan_drift(trigger="local"):
    """Decides if retraining or model replacement is needed."""
    # (This logic can be expanded similarly to plan_mape if needed)
    logging.info(f"PLAN (Drift) triggered by: {trigger.upper()}")
    
    drift = analyse_drift()
    if not drift or not drift["drift_detected"]:
        logging.info("PLAN (Drift): No drift detected. No action required.")
        return None

    if drift["best_version"]:
        logging.info(f"PLAN (Drift): Switching to lower KL divergence model: {drift['best_version']}")
        return {"action": "replace", "version": drift["best_version"]}
    
    logging.info("PLAN (Drift): Drift detected! No previous version available. Retraining required.")
    return {"action": "retrain"}

def plan_simple_switch():
    """
    A simple baseline plan: just switch to a different model.
    Picks one of the *other* available models at random.
    """
    logging.info("PLAN (Simple Switch): Triggered by R² baseline.")
    current_model = get_current_model()
    available_models = ["lstm", "linear", "svm"]
    
    # Remove the current model from the list
    available_models.remove(current_model) 
    
    # Randomly pick from the remaining two
    chosen_model = random.choice(available_models)
    
    logging.info(f"PLAN (Simple Switch): Switching from '{current_model.upper()}' to '{chosen_model.upper()}'.")
    return chosen_model