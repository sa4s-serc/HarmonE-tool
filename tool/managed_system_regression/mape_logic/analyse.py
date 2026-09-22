import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import entropy
from monitor import monitor_mape, monitor_drift

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..", "..")))
from session_utils import session_versioned_dir

# Only this session's versions are reuse candidates: HarmonE's VMR reuse is
# "drift realigned with a distribution THIS run already trained on", not
# "some model from an unrelated earlier run happens to match". See
# tool/session_utils.py.
BASE_VERSION_DIR = session_versioned_dir(
    os.path.join(BASE_DIR, "..", "versionedMR"), KNOWLEDGE_DIR
)

thresholds_file = os.path.join(KNOWLEDGE_DIR, "thresholds.json")
mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
current_model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
drift_kl_file = os.path.join(KNOWLEDGE_DIR, "drift_kl.json")
drift_data_file = os.path.join(KNOWLEDGE_DIR, "drift.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")

def load_mape_info():
    """Load stored MAPE info including energy debt and recovery cycles.

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
    """Save updated MAPE info including energy debt and recovery cycles.

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
    """Analyze performance and decide if switching is needed, using dynamic energy thresholds."""
    mape_data = monitor_mape()
    if not mape_data:
        print("⚠️ No MAPE data available for analysis.")
        return None

    # Load thresholds
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)

    min_score = thresholds["min_score"]
    original_energy_threshold = thresholds["max_energy"]

    # Load current MAPE info
    mape_info = load_mape_info()
    current_energy_threshold = mape_info.get("current_energy_threshold", original_energy_threshold)
    recovery_cycles = mape_info["recovery_cycles"]

    # Update energy threshold dynamically
    used_energy = mape_data["normalized_energy"]
    new_energy_threshold = current_energy_threshold + 0.95 * (original_energy_threshold - used_energy)
    mape_info["current_energy_threshold"] = new_energy_threshold

    # Check if switching is needed
    switch_needed = False
    threshold_violated = None

    if recovery_cycles > 0:
        recovery_cycles -= 1
        print(f"⏳ Recovery mode active: {recovery_cycles} cycles remaining. No switching allowed.")
    else:
        if mape_data["score"] < min_score:
            print("⚠️ Model score too low! Model switch required.")
            switch_needed = True
            threshold_violated = "score"

        if used_energy > current_energy_threshold:
            print(f"⚠️ Energy threshold exceeded! Used: {used_energy:.4f}, Limit: {current_energy_threshold:.4f}")
            switch_needed = True
            threshold_violated = "energy"
            recovery_cycles = 3

    # Save updated info
    mape_info["recovery_cycles"] = recovery_cycles
    # Persist what this check actually found, so execute.py can log the real
    # trigger context (score/threshold, not just "a switch happened") against
    # any adaptation this decision leads to.
    mape_info["last_score_check"] = {
        "score": mape_data["score"],
        "min_score_threshold": min_score,
        "energy_normalized": used_energy,
        "energy_threshold": current_energy_threshold,
        "switch_needed": switch_needed,
        "threshold_violated": threshold_violated,
    }
    save_mape_info(mape_info)

    print(f"📊 Updated Energy Threshold: {new_energy_threshold:.4f}")

    return {
        "switch_needed": switch_needed,
        "score": mape_data["score"],
        "threshold_violated": threshold_violated
    }


def get_model_versions(model_name):
    """Returns available versions for a given model."""
    model_dir = os.path.join(BASE_VERSION_DIR, model_name)
    if not os.path.exists(model_dir):
        print("debug: Hey! this path is incorrect")
        return []
    return sorted([d for d in os.listdir(model_dir) if d.startswith("version_")], key=lambda x: int(x.split("_")[-1]))

def get_best_version(model_name):
    """Finds the version with the lowest KL divergence from past data."""
    versions = get_model_versions(model_name)
    if len(versions) <= 1:
        return None  # No previous versions exist

    if not os.path.exists(drift_data_file):
        print("⚠️ No drift.csv found. Cannot compare versions.")
        return None

    # Load current drift data
    try:
        drift_data = pd.read_csv(drift_data_file)["true_value"].values
        drift_hist, _ = np.histogram(drift_data, bins=50, density=True)
        drift_hist += 1e-10  
    except Exception as e:
        print(f"❌ Error reading drift.csv: {e}")
        return None

    min_kl_div = float("inf")
    best_version = None

    for version in versions:
        version_data_path = os.path.join(BASE_VERSION_DIR, model_name, version, "data.csv")
        if not os.path.exists(version_data_path):
            continue

        # Load versioned model's training data
        try:
            version_data = pd.read_csv(version_data_path)["train_data"].values
            version_hist, _ = np.histogram(version_data, bins=50, density=True)
            version_hist += 1e-10  # ✅ Prevent zero probabilities
        except Exception as e:
            print(f"❌ Error reading {version_data_path}: {e}")
            continue

        # Compute KL divergence
        kl_div = entropy(drift_hist, version_hist)
        kl_div = np.clip(kl_div, 0, 10) 
        print(f"🔎 KL divergence for {version}: {kl_div:.4f}")

        if kl_div < min_kl_div:
            min_kl_div = kl_div
            best_version = version_data_path  # Return the best version data path

    # Store KL divergences for debugging
    with open(drift_kl_file, "w") as f:
        json.dump({"best_version": best_version, "min_kl_div": min_kl_div}, f, indent=4)

    return best_version if min_kl_div < 0.75 else None  # Use version if KL is below threshold

def analyse_drift():
    """Analyze drift & decide if retraining is needed or if an existing version can be used."""
    drift_data = monitor_drift()
    if not drift_data:
        return None

    kl_div = float(drift_data["kl_div"])
    drift_threshold = 0.5
    drift_detected = bool(kl_div > drift_threshold)

    # Persist what this check actually found (see analyse_mape's last_score_check
    # for the same rationale) so the real KL divergence/threshold is available
    # to log against whatever adaptation this decision leads to. kl_div/comparison
    # results here are numpy scalars (from scipy/numpy ops in monitor.py); cast
    # to native float/bool since numpy.bool_ (unlike numpy.float64) isn't
    # JSON-serializable by the stdlib json module.
    mape_info = load_mape_info()
    mape_info["last_drift_check"] = {
        "kl_div": kl_div,
        "drift_threshold": drift_threshold,
        "drift_detected": drift_detected,
    }
    save_mape_info(mape_info)

    if drift_detected:
        print(f"🚨 Drift detected! KL divergence = {kl_div:.4f}")
        try:
            df = pd.read_csv(predictions_file)
            df.columns = df.columns.str.strip()
            df.tail(1200).to_csv(drift_data_file, index=False)
        except FileNotFoundError:
            print("No predictions file found to store drift data.")

        # Get the currently used model
        with open(current_model_file, "r") as f:
            current_model = f.read().strip()

        best_version = get_best_version(current_model)

        if best_version:
            print(f"✔ Best version found with lower KL divergence: {best_version}")
            return {"drift_detected": True, "best_version": best_version}

        # No suitable previous version found → Retrain needed
        return {"drift_detected": True, "best_version": None}

    return {"drift_detected": False}