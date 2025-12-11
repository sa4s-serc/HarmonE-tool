import pandas as pd
import os
import json

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")
PREDICTIONS_FILE = os.path.join(KNOWLEDGE_DIR, "predictions.csv")
MODEL_FILE = os.path.join(KNOWLEDGE_DIR, "model.csv")

def get_current_model():
    try:
        with open(MODEL_FILE, "r") as f:
            return f.read().strip()
    except:
        return "unknown"

def monitor_mape():
    """
    Reads the last few predictions to generate telemetry.
    """
    current_model = get_current_model()
    
    # Default values if no data exists yet
    metrics = {
        "score": 0.8,
        "energy": round(0.5, 2),
        "normalized_energy": 0.3,
        "model_used": current_model,
        "r2_score": 0.9,
        "confidence": 0.85
    }

    if os.path.exists(PREDICTIONS_FILE):
        try:
            # Read last 10 rows to calculate current performance
            df = pd.read_csv(PREDICTIONS_FILE)
            if not df.empty:
                df = df.tail(10)
                # Simple mock logic: if model is 'svm' or 'yolo_n', score is lower
                if "svm" in current_model or "yolo_n" in current_model:
                    metrics["score"] = 0.65 # trigger adaptation
                else:
                    metrics["score"] = 0.95
        except Exception as e:
            print(f"[CustomMonitor] Error reading predictions: {e}")

    print(f"[CustomMonitor] Reporting: {metrics}")
    return metrics

def monitor_drift():
    # Placeholder for drift
    return {"kl_div": 0.05}