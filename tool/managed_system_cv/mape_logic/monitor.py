import pandas as pd
import numpy as np
import json
import time
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
    """Save MAPE info atomically.

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
                "inference_time": None,
            }
    except FileNotFoundError:
        print("[MAPE] Predictions file not found.")
        return None

    thresholds = json.load(open(thresholds_file))
    # 1. FETCH ENERGY MIN/MAX BY KEY
    energy_min = thresholds.get("E_m", 0)
    energy_max = thresholds.get("E_M", 10000000)

    df = _numeric_frame(df, ["confidence", "energy_uJ"])
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

    print(f"📊 Event Counters - Switches: {event_counters['model_switches']}, Retrains: {event_counters['retrains']}, VMR: {event_counters['vmr_events']}, MAPE-K Energy: {event_counters['mape_k_energy_uJ']:.2f} µJ")

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
        "inference_time": _mean_inference_time(df),
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