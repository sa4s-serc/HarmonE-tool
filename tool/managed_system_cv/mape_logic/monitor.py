import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.drift_utils import kl_divergence

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
thresholds_file = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")


def load_mape_info():
    with open(mape_info_file, "r") as f:
        return json.load(f)

def save_mape_info(data):
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def get_current_model():
    try:
        with open(model_file, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def monitor_mape():
    info = load_mape_info()
    last_line = info["last_line"]
    current_model = get_current_model()
    if current_model is None:
        print("[MAPE] No current model found.")
        return None

    try:
        df = pd.read_csv(predictions_file, skiprows=range(1, last_line+1))
        if df.empty:
            print("[MAPE] No new predictions to monitor.")
            # Return cached values with event counters when no new data
            event_counters = info.get("event_counters", {
                "model_switches": 0,
                "retrains": 0,
                "vmr_events": 0,
                "mape_k_energy_uJ": 0.0
            })
            
            # Include simple switch counters
            simple_switch_counters = info.get("simple_switch_counters", {
                "simple_switches": 0
            })
            
            # Use cached EMA score
            final_score = info["ema_scores"].get(current_model, 0.5)
            
            print(f"📊 Event Counters - Switches: {event_counters['model_switches']}, Retrains: {event_counters['retrains']}, VMR: {event_counters['vmr_events']}, MAPE-K Energy: {event_counters['mape_k_energy_uJ']:.2f} µJ")
            
            return {
                "confidence": 0.5,  # Default value
                "energy": 0.0,  # Default actual energy value for display
                "normalized_energy": 0.5,  # Default normalized value for calculations
                "score": final_score,
                "model_used": current_model,
                "model_switches": event_counters["model_switches"],
                "retrains": event_counters["retrains"],
                "vmr_events": event_counters["vmr_events"],
                "mape_k_energy_uJ": round(event_counters["mape_k_energy_uJ"], 2),
                "simple_switches": simple_switch_counters["simple_switches"]
            }
    except FileNotFoundError:
        print("[MAPE] Predictions file not found.")
        return None

    thresholds = json.load(open(thresholds_file))
    # 1. FETCH ENERGY MIN/MAX BY KEY
    energy_min = thresholds.get("E_m", 0)
    energy_max = thresholds.get("E_M", 10000000)

    avg_conf = df["confidence"].mean()
    avg_energy = df["energy_uJ"].mean()

    # Avoid division by zero if energy_max equals energy_min
    if (energy_max - energy_min) > 0:
        energy_norm = (avg_energy - energy_min) / (energy_max - energy_min)
    else:
        energy_norm = 0.0
    energy_norm = np.clip(energy_norm, 0, 1)

    beta = thresholds.get("beta", 0.95)
    score = beta * avg_conf + (1 - beta) * (1 - energy_norm)

    gamma = thresholds.get("gamma", 0.8)
    prev_score = info["ema_scores"].get(current_model, 0.5)
    final_score = gamma * score + (1 - gamma) * prev_score

    print(f"[MAPE] Monitoring values for model {current_model}:")
    print(f"  avg_confidence={avg_conf:.4f}, avg_energy={avg_energy:.2f}, normalized_energy={energy_norm:.4f}")
    print(f"  score={score:.4f}, final_score={final_score:.4f}")

    info["ema_scores"][current_model] = final_score
    info["last_line"] += len(df)
    
    # Ensure event counters exist
    if "event_counters" not in info:
        info["event_counters"] = {
            "model_switches": 0,
            "retrains": 0,
            "vmr_events": 0,
            "mape_k_energy_uJ": 0.0
        }
    
    save_mape_info(info)

    # Include event counters in telemetry
    event_counters = info["event_counters"]
    
    # Include simple switch counters
    simple_switch_counters = info.get("simple_switch_counters", {
        "simple_switches": 0
    })
    
    print(f"📊 Event Counters - Switches: {event_counters['model_switches']}, Retrains: {event_counters['retrains']}, VMR: {event_counters['vmr_events']}, MAPE-K Energy: {event_counters['mape_k_energy_uJ']:.2f} µJ, Simple Switches: {simple_switch_counters['simple_switches']}")

    return {
        "confidence": avg_conf,
        "energy": round(avg_energy, 2),  # Return actual energy for display
        "normalized_energy": energy_norm,  # Keep normalized energy for internal calculations
        "score": final_score,
        "model_used": current_model,
        "model_switches": event_counters["model_switches"],
        "retrains": event_counters["retrains"],
        "vmr_events": event_counters["vmr_events"],
        "mape_k_energy_uJ": round(event_counters["mape_k_energy_uJ"], 2),
        "simple_switches": simple_switch_counters["simple_switches"]
    }

def monitor_drift():
    try:
        df = pd.read_csv(predictions_file)
        # 2. USE LUMINANCE HISTOGRAMS FOR KL DIVERGENCE
        # We need two windows of 1000, so at least 2000 data points.
        if len(df) < 2000:
            print("[DRIFT] Not enough data for drift monitoring (need 2000 entries).")
            return None

        if "histogram" not in df.columns:
            print("[DRIFT] 'histogram' column not found in predictions.csv. Cannot monitor drift.")
            print("[DRIFT] Please update inference.py to save histograms.")
            return None

        # Reference window: images from -2000 to -1000
        ref_hists_str = df["histogram"].iloc[-2000:-1000]
        # Current window: images from -1000 to present
        cur_hists_str = df["histogram"].iloc[-1000:]

        # Convert string histograms to numpy arrays
        ref_hists = np.array([np.fromstring(h, sep=' ') for h in ref_hists_str if h])
        cur_hists = np.array([np.fromstring(h, sep=' ') for h in cur_hists_str if h])

        if ref_hists.size == 0 or cur_hists.size == 0:
            print("[DRIFT] Could not parse histograms from predictions.csv.")
            return None

        # Average the histograms for each window to get a single distribution
        ref_dist = np.mean(ref_hists, axis=0)
        cur_dist = np.mean(cur_hists, axis=0)

        # Calculate KL divergence using the imported utility function
        kl = kl_divergence(cur_dist, ref_dist)

        print(f"[DRIFT] KL divergence computed on luminance histograms: {kl:.4f}")
        return {"kl_div": kl}

    except FileNotFoundError:
        print("[DRIFT] Predictions file not found for drift monitoring.")
        return None
    except Exception as e:
        print(f"[DRIFT] An error occurred during drift monitoring: {e}")
        return None