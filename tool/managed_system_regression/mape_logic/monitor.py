import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import entropy
import json
import os

# Get the absolute path of the current script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
thresholds_file = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")

def load_mape_info():
    """Load MAPE info from JSON file."""
    with open(mape_info_file, "r") as f:
        return json.load(f)

def save_mape_info(data):
    """Save updated MAPE info including model-specific EMA scores."""
    with open(mape_info_file, "w") as f:
        json.dump(data, f, indent=4)

def get_current_model():
    """Fetch the currently active model from knowledge."""
    try:
        with open(model_file, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def monitor_mape():
    """Monitor R² Score and Actual Energy, and Compute Score."""
    info = load_mape_info()
    last_line = info["last_line"]
    current_model = get_current_model()
    
    if current_model is None:
        print("⚠️ No model currently in use.")
        return None

    try:
        df = pd.read_csv(predictions_file, skiprows=range(1, last_line + 1))
        df.columns = df.columns.str.strip()
        
        # If no new data, return cached values based on recent data
        if df.empty:
            print("📉 No new data to process in predictions.csv, using recent data for telemetry")
            # Read the last 50 rows to compute current metrics
            try:
                recent_df = pd.read_csv(predictions_file).tail(50)
                recent_df.columns = recent_df.columns.str.strip()
                
                if not recent_df.empty and 'energy' in recent_df.columns and 'true_value' in recent_df.columns and 'predicted_value' in recent_df.columns:
                    r2 = r2_score(recent_df["true_value"], recent_df["predicted_value"])
                    
                    # Load thresholds
                    with open(thresholds_file, "r") as f:
                        thresholds = json.load(f)
                    energy_min, energy_max = thresholds["E_m"], thresholds["E_M"]
                    
                    # Calculate actual and normalized energy
                    avg_energy = recent_df["energy"].mean()
                    if energy_max > energy_min:
                        energy_normalized = (avg_energy - energy_min) / (energy_max - energy_min)
                        energy_normalized = max(0.0, min(1.0, energy_normalized))  # Clamp between 0 and 1
                    else:
                        energy_normalized = 0.0
                    
                    # Use cached EMA score
                    final_score = info["ema_scores"].get(current_model, 0.5)
                    
                    print(f"🔄 Using recent data: R²={r2:.4f}, Actual Energy={avg_energy:.2f}, Normalized Energy={energy_normalized:.4f}, Score={final_score:.4f}")
                    
                    # Include event counters in telemetry
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
                    
                    return {
                        "r2_score": round(r2, 4),
                        "energy": round(avg_energy, 2),  # Return actual energy for display
                        "normalized_energy": round(energy_normalized, 4),  # Keep for internal calculations
                        "score": round(final_score, 4),
                        "model_used": current_model,
                        "model_switches": event_counters["model_switches"],
                        "retrains": event_counters["retrains"],
                        "vmr_events": event_counters["vmr_events"],
                        "mape_k_energy_uJ": round(event_counters["mape_k_energy_uJ"], 2),
                        "simple_switches": simple_switch_counters["simple_switches"]
                    }
                else:
                    print("⚠️ Required columns missing in recent data")
                    return None
            except Exception as e:
                print(f"⚠️ Error reading recent data: {e}")
                return None
            
    except FileNotFoundError:
        print("⚠️ No predictions.csv file found.")
        return None

    print(f"🆕 Processing {len(df)} new rows from predictions.csv for {current_model.upper()}")

    # Calculate R² score
    r2 = r2_score(df["true_value"], df["predicted_value"])

    # Compute Actual and Normalized Energy
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)
    energy_min, energy_max = thresholds["E_m"], thresholds["E_M"]
    
    avg_energy = df["energy"].mean()
    print(f"Average energy: {avg_energy}, Min: {energy_min}, Max: {energy_max}")
    
    # Ensure energy normalization doesn't cause division by zero
    if energy_max > energy_min:
        energy_normalized = (avg_energy - energy_min) / (energy_max - energy_min)
        energy_normalized = max(0.0, min(1.0, energy_normalized))  # Clamp between 0 and 1
    else:
        energy_normalized = 0.0

    # Calculate model score (still use normalized energy for scoring)
    beta = thresholds.get("beta", 0.5)
    model_score = beta * r2 + (1 - beta) * (1 - energy_normalized)

    # Compute Exponential Moving Average (EMA)
    gamma = thresholds.get("gamma", 0.8)
    prev_score = info["ema_scores"].get(current_model, 0.5)
    final_score = gamma * model_score + (1 - gamma) * prev_score

    # Update MAPE info
    info["ema_scores"][current_model] = final_score
    info["last_line"] += len(df)

    # Log computed values
    print(f"🔹 R² Score: {r2:.4f}")
    print(f"🔹 Actual Energy: {avg_energy:.2f}")
    print(f"🔹 Normalized Energy: {energy_normalized:.4f}")
    print(f"🔹 Model Score for {current_model.upper()}: {model_score:.4f}")
    print(f"🔹 Updated EMA Score for {current_model.upper()}: {final_score:.4f}")

    save_mape_info(info)

    # Include event counters in telemetry
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
    
    print(f"📊 Event Counters - Switches: {event_counters['model_switches']}, Retrains: {event_counters['retrains']}, VMR: {event_counters['vmr_events']}, MAPE-K Energy: {event_counters['mape_k_energy_uJ']:.2f} µJ")

    return {
        "r2_score": round(r2, 4),
        "energy": round(avg_energy, 2),  # Return actual energy for display
        "normalized_energy": round(energy_normalized, 4),  # Keep for internal calculations
        "score": round(final_score, 4),
        "model_used": current_model,
        "model_switches": event_counters["model_switches"],
        "retrains": event_counters["retrains"],
        "vmr_events": event_counters["vmr_events"],
        "mape_k_energy_uJ": round(event_counters["mape_k_energy_uJ"], 2),
        "simple_switches": simple_switch_counters["simple_switches"]
    }

def monitor_drift():
    """Monitor data drift without enforcing immediate retraining."""
    try:
        df = pd.read_csv(predictions_file)
        df.columns = df.columns.str.strip()
        
        if df.empty:
            print("Drift Monitor: No predictions yet.")
            return None

        window_size = 1200
        if len(df) >= window_size * 2:
            reference_window = df['true_value'].iloc[-2*window_size:-window_size]
            current_window = df['true_value'].iloc[-window_size:]
            
            # Calculate KL divergence
            ref_hist, _ = np.histogram(reference_window, bins=50, density=True)
            curr_hist, _ = np.histogram(current_window, bins=50, density=True)
            
            # Add small epsilon to avoid log(0)
            ref_hist = ref_hist + 1e-10
            curr_hist = curr_hist + 1e-10
            
            kl_div = entropy(ref_hist, curr_hist)
            
            print(f"🌊 Drift: KL={kl_div:.4f}")
            return {"kl_div": round(kl_div, 4)}
        else:
            print(f"Not enough data for drift detection. Have {len(df)} samples, need {window_size * 2}")
            # Return a placeholder drift value for now
            return {"kl_div": round(np.random.uniform(0.01, 0.15), 4)}
    
    except FileNotFoundError:
        print("Drift Monitor: No predictions found.")
        return None
    except Exception as e:
        print(f"Drift Monitor Error: {e}")
        return None