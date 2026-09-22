# research/sustainable-mlops/HarmonEXT/mape/analyse.py
import os
import json
import time
import re
import numpy as np
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utility.drift_utils import kl_divergence
from monitor import monitor_mape, monitor_drift
from session_utils import session_versioned_dir

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

thresholds_file = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
current_model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")
drift_kl_file = os.path.join(KNOWLEDGE_DIR, "drift_kl.json")
# Only this session's versions are reuse candidates - see tool/session_utils.py.
versioned_dir = session_versioned_dir(
    os.path.join(BASE_DIR, "..", "versionedMR"), KNOWLEDGE_DIR
)

ALL_MODELS = ["yolo_n", "yolo_s", "yolo_m"]
DRIFT_THRESHOLD = 0.07

def load_mape_info():
    """Load stored MAPE info including energy threshold and recovery cycles."""
    try:
        with open(mape_info_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "last_line": 0,
            "current_energy_threshold": json.load(open(thresholds_file))["max_energy"],
            "ema_scores": {m: 0.5 for m in ALL_MODELS},
            "recovery_cycles": 0
        }

def save_mape_info(data):
    """Save updated MAPE info.

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

def analyse_mape():
    """Analyze performance and decide if switching is needed, using dynamic energy thresholds and recovery cycles."""
    data = monitor_mape()
    if not data:
        print("[MAPE] No monitoring data available for analysis.")
        return None

    thresholds = json.load(open(thresholds_file))
    min_score = thresholds["min_score"]
    original_energy_threshold = thresholds["max_energy"]

    mape_info = load_mape_info()
    current_energy_threshold = mape_info.get("current_energy_threshold", original_energy_threshold)
    recovery_cycles = mape_info.get("recovery_cycles", 0)

    used_energy_norm = data["normalized_energy"]
    new_energy_threshold = current_energy_threshold + 0.4 * (original_energy_threshold - used_energy_norm)
    mape_info["current_energy_threshold"] = new_energy_threshold

    switch_needed = False
    threshold_violated = None

    if recovery_cycles > 0:
        recovery_cycles -= 1
        print(f"[MAPE] Recovery mode active: {recovery_cycles} cycles remaining. No switching allowed.")
    else:
        if data["score"] < min_score:
            switch_needed = True
            threshold_violated = "score"
            print(f"[MAPE] Threshold violated: score ({data['score']:.4f} < {min_score})")

        if used_energy_norm > current_energy_threshold:
            switch_needed = True
            threshold_violated = "energy"
            recovery_cycles = 3
            print(f"[MAPE] Threshold violated: energy ({used_energy_norm:.4f} > {current_energy_threshold:.4f}). Entering recovery mode.")

    mape_info["recovery_cycles"] = recovery_cycles
    # Persist what this check actually found, so execute.py can log the real
    # trigger context (score/threshold) against any adaptation it leads to.
    mape_info["last_score_check"] = {
        "score": data["score"],
        "min_score_threshold": min_score,
        "energy_normalized": used_energy_norm,
        "energy_threshold": current_energy_threshold,
        "switch_needed": switch_needed,
        "threshold_violated": threshold_violated,
    }
    save_mape_info(mape_info)
    print(f"[MAPE] Updated Energy Threshold: {new_energy_threshold:.4f}")

    return {"switch_needed": switch_needed, "threshold_violated": threshold_violated, "score": data["score"]}


def get_best_version_for_model(model_name, current_drift_dist):
    """
    Finds the best previous version for a SINGLE model type.
    Returns the best version path and its KL divergence.
    """
    pattern = re.compile(f"({model_name}_v(\\d+))\\.pt")
    versions = [m.group(1) for f in os.listdir(versioned_dir) if (m := pattern.match(f))]
    if len(versions) <= 1:
        return None, float('inf')

    min_kl_div = float("inf")
    best_version_path = None

    for version_base_name in versions:
        hist_path = os.path.join(versioned_dir, f"{version_base_name}_hist.json")
        if not os.path.exists(hist_path):
            continue

        try:
            with open(hist_path, "r") as f:
                version_dist = np.array(json.load(f)["average_histogram"])
            kl_div = kl_divergence(current_drift_dist, version_dist)

            if kl_div < min_kl_div:
                min_kl_div = kl_div
                best_version_path = os.path.join(versioned_dir, f"{version_base_name}.pt")
        except Exception:
            continue

    return best_version_path, min_kl_div


def analyse_drift():
    """
    Analyze data drift. If detected, search across ALL model types (n, s, m)
    for the best existing version before triggering a retrain.
    """
    drift = monitor_drift()
    if not drift:
        print("[DRIFT] No drift monitoring data available for analysis.")
        return None

    kl_div = float(drift["kl_div"])

    # Persist what this check actually found (see analyse_mape's last_score_check
    # for the same rationale) so the real KL divergence/threshold is available
    # to log against whatever adaptation this decision leads to. Cast to native
    # float/bool since numpy.bool_ (unlike numpy.float64) isn't JSON-serializable
    # by the stdlib json module.
    mape_info = load_mape_info()
    mape_info["last_drift_check"] = {
        "kl_div": kl_div,
        "drift_threshold": DRIFT_THRESHOLD,
        "drift_detected": bool(kl_div > DRIFT_THRESHOLD),
    }
    save_mape_info(mape_info)

    if kl_div <= DRIFT_THRESHOLD:
        print(f"[DRIFT] No significant drift detected. KL ({kl_div:.4f}) <= {DRIFT_THRESHOLD}")
        return {"drift_detected": False}

    print(f"[DRIFT] Drift detected! KL ({kl_div:.4f}) > {DRIFT_THRESHOLD}")

    # --- Drift is detected, now find the best possible version across ALL models ---
    try:
        df = pd.read_csv(predictions_file)
        if len(df) < 1000:
            print("[DRIFT] Not enough data to compare versions. Planning retrain.")
            return {"drift_detected": True, "best_version": None, "action": "retrain"}
        
        drift_hists_str = df["histogram"].iloc[-1000:]
        drift_hists = np.array([np.fromstring(h, sep=' ') for h in drift_hists_str if h])
        if drift_hists.size == 0:
             return {"drift_detected": True, "best_version": None, "action": "retrain"}
        current_drift_dist = np.mean(drift_hists, axis=0)
    except Exception as e:
        print(f"[DRIFT] Error processing current data for version comparison: {e}. Planning retrain.")
        return {"drift_detected": True, "best_version": None, "action": "retrain"}

    overall_best_version_path = None
    overall_min_kl_div = float('inf')
    all_kl_results = {}

    print("[DRIFT] Searching all model types for the best version to handle drift...")
    for model_name in ALL_MODELS:
        best_path, min_kl = get_best_version_for_model(model_name, current_drift_dist)
        all_kl_results[model_name] = min_kl
        print(f"[DRIFT] Best version for {model_name.upper()}: KL={min_kl:.4f}")
        if min_kl < overall_min_kl_div:
            overall_min_kl_div = min_kl
            overall_best_version_path = best_path

    # Store comprehensive KL results for debugging
    with open(drift_kl_file, "w") as f:
        json.dump({
            "best_overall_version": overall_best_version_path,
            "min_overall_kl": overall_min_kl_div,
            "kl_per_model": all_kl_results
        }, f, indent=4)

    # Decide on the final action
    if overall_min_kl_div < DRIFT_THRESHOLD:
        print(f"[DRIFT] Found suitable version across all models: {overall_best_version_path} (KL={overall_min_kl_div:.4f})")
        return {"drift_detected": True, "best_version": overall_best_version_path, "action": "switch_version"}
    else:
        print("[DRIFT] No suitable previous version found across any model type. A full retrain is needed.")
        return {"drift_detected": True, "best_version": None, "action": "retrain"}