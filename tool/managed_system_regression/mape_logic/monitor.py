import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import entropy
import json
import time
import os

# Get the absolute path of the current script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
thresholds_file = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")

def load_mape_info():
    """Load MAPE info from JSON file.

    Retries once: a concurrent writer can briefly leave the file mid-rename on
    Windows, and this used to raise straight out of the monitoring loop.
    """
    for attempt in (0, 1):
        try:
            with open(mape_info_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, PermissionError):
            if attempt:
                raise
            time.sleep(0.05)

def save_mape_info(data):
    """Save updated MAPE info including model-specific EMA scores.

    Written to a temp file and moved into place with os.replace(), which is
    atomic on Windows and POSIX. Three processes (inference.py via monitor,
    manage.py via analyse/execute) read-modify-write this file concurrently;
    a plain open(path, "w") truncates it to zero before the new bytes land,
    so a reader landing in that window gets an empty file. Measured at a 25%
    failure rate under sustained concurrent writes.
    """
    tmp_path = "%s.tmp%s" % (mape_info_file, os.getpid())
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())

    # os.replace raises WinError 5 if anything else currently has the
    # destination open, and the dashboard polls this file every couple of
    # seconds. Those read handles live for microseconds, so a short retry
    # clears them; if it somehow does not, fall back to the direct write
    # rather than lose the update or crash the loop.
    for _ in range(40):
        try:
            os.replace(tmp_path, mape_info_file)
            return
        except PermissionError:
            time.sleep(0.02)

    # Still locked after ~0.8s. Skip this snapshot rather than fall back to a
    # truncating write: the next tick rewrites the whole file anyway, whereas a
    # torn write can leave state that load_mape_info cannot parse.
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("[MAPE] WARNING: could not update %s (locked); skipping this snapshot" % mape_info_file)

def get_current_model():
    """Fetch the currently active model from knowledge."""
    try:
        with open(model_file, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def _mean_inference_time(frame):
    """Mean inference_time (seconds) over this window, or None.

    Inference time is one of the paper's four headline metrics, but it was
    never put into the telemetry payload - only into predictions.csv. The
    dashboard's latency chart reads telemetry, so it had nothing to plot,
    while the Models table looked fine because it reads predictions.csv
    directly. Returned in seconds; the UI converts to ms.
    """
    try:
        if frame is None or "inference_time" not in frame.columns:
            return None
        value = frame["inference_time"].mean()
        return None if value != value else round(float(value), 6)  # NaN-safe
    except Exception:
        return None

def _numeric_frame(frame, columns):
    """Drop rows whose numeric columns are not actually numeric.

    predictions.csv accumulates torn rows when more than one writer appends to
    it, and a single one of those ('svm' landing in true_value) raised out of
    r2_score and stopped telemetry entirely. Coerce and drop instead.
    """
    import pandas as _pd
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = _pd.to_numeric(out[col], errors="coerce")
    present = [c for c in columns if c in out.columns]
    return out.dropna(subset=present) if present else out

def monitor_mape():
    """Monitor R² Score and Actual Energy, and Compute Score."""
    info = load_mape_info()
    last_line = info["last_line"]
    # predictions.csv is recreated empty on a fresh run, but last_line is not
    # reset alongside it. Once the pointer sits past the end of the file,
    # skiprows always yields an empty frame, so the "new data" branch never
    # runs - and that branch is the ONLY place ema_scores and last_line are
    # updated. The loop then plans forever on EMA values frozen from before
    # the reset. Observed live: last_line=53986 against a 50856-row file.
    try:
        total_rows = sum(1 for _ in open(predictions_file)) - 1  # minus header
    except OSError:
        total_rows = None
    if total_rows is not None and last_line > max(total_rows, 0):
        # Resync to the end rather than rewinding to 0. Rewinding replays the
        # whole historical file, which contains 789 torn rows left over from
        # when duplicate inference.py processes appended concurrently - one of
        # them ("could not convert string to float: 'svm'") killed every
        # telemetry tick. Everything already written counts as processed.
        print(f"[MAPE] last_line ({last_line}) is past the end of predictions.csv "
              f"({total_rows} rows) - the file was reset; resyncing to {total_rows}")
        last_line = total_rows
        info["last_line"] = total_rows
        save_mape_info(info)

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
                
                if not recent_df.empty and 'energy_uJ' in recent_df.columns and 'true_value' in recent_df.columns and 'predicted_value' in recent_df.columns:
                    recent_df = _numeric_frame(recent_df, ["true_value", "predicted_value", "energy_uJ"])
                    r2 = r2_score(recent_df["true_value"], recent_df["predicted_value"])

                    # Load thresholds
                    with open(thresholds_file, "r") as f:
                        thresholds = json.load(f)
                    energy_min, energy_max = thresholds["E_m"], thresholds["E_M"]

                    # Calculate actual and normalized energy
                    avg_energy = recent_df["energy_uJ"].mean()
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
                        "inference_time": _mean_inference_time(recent_df),
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
    df = _numeric_frame(df, ["true_value", "predicted_value", "energy_uJ"])
    r2 = r2_score(df["true_value"], df["predicted_value"])

    # Compute Actual and Normalized Energy
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)
    energy_min, energy_max = thresholds["E_m"], thresholds["E_M"]

    avg_energy = df["energy_uJ"].mean()
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
        "inference_time": _mean_inference_time(df),
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
            # Not enough data yet for a real KL divergence - previously this
            # logged a random placeholder value (np.random.uniform(0.01, 0.15))
            # into MLflow/telemetry as if it were a measurement, which just
            # added noise to the drift chart. Skip the tick instead, matching
            # managed_system_cv's monitor_drift(), which does the same.
            print(f"Not enough data for drift detection. Have {len(df)} samples, need {window_size * 2}")
            return None
    
    except FileNotFoundError:
        print("Drift Monitor: No predictions found.")
        return None
    except Exception as e:
        print(f"Drift Monitor Error: {e}")
        return None