import os
import sys
import shutil
import time
import re
import json
import csv
import logging
import mlflow
from mlflow.tracking import MlflowClient

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", ".."))
import energy_utils as pyRAPL
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")
SESSION_RUN_ID_FILE = os.path.join(KNOWLEDGE_DIR, "mlflow_session_run_id.txt")

# mlflow.projects.run() below pre-creates its run against whatever tracking URI
# *this* process is using, then hands the run ID to the retrain.py subprocess via
# env var. We must set it here (the caller), not just inside retrain.py, or the
# child looks for that run ID in the wrong store and crashes.
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, '..', '..', 'mlflow.db')}")

# --- Shared adaptation timeline ---
# See managed_system_regression/mape_logic/execute.py for the full rationale:
# run_managed_system.py (a different OS process) owns the session run and logs
# continuous telemetry onto it; we append our adaptation events to that SAME
# run_id via MlflowClient, so both land on one native, chartable timeline.
# One experiment per managed system - see run_managed_system.py.
ADAPTATIONS_EXPERIMENT = "harmonica-cv"
_mlflow_client = MlflowClient()
_session_run_id = None

def _get_local_approach():
    try:
        with open(os.path.join(BASE_DIR, "..", "approach.conf")) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "unknown"

def _active_experiment_id(name):
    """Experiment id for `name`, created or restored so it is always ACTIVE.

    get_experiment_by_name() returns SOFT-DELETED experiments too, so taking
    its id blindly makes create_run() fail with "must be in the 'active'
    state". That is exactly how session runs silently stopped being created:
    'harmone-adaptations' had been deleted, the id still resolved, create_run
    raised, the surrounding except swallowed it, and the whole adaptation
    timeline vanished with no visible error.
    """
    exp = _mlflow_client.get_experiment_by_name(name)
    if exp is None:
        return _mlflow_client.create_experiment(name)
    if exp.lifecycle_stage == "deleted":
        _mlflow_client.restore_experiment(exp.experiment_id)
        logging.warning("[MLFLOW] Restored soft-deleted experiment '%s'", name)
    return exp.experiment_id


def _get_session_run_id():
    global _session_run_id
    if _session_run_id is None:
        try:
            with open(SESSION_RUN_ID_FILE) as f:
                _session_run_id = f.read().strip()
        except FileNotFoundError:
            # Fallback for standalone use without run_managed_system.py.
            exp_id = _active_experiment_id(ADAPTATIONS_EXPERIMENT)
            run = _mlflow_client.create_run(experiment_id=exp_id, run_name=f"session-{int(time.time())}")
            _session_run_id = run.info.run_id
            _mlflow_client.set_tag(_session_run_id, "approach", _get_local_approach())
    return _session_run_id

model_file = os.path.join(KNOWLEDGE_DIR, "model.csv")
mape_info_file = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
event_log_file = os.path.join(KNOWLEDGE_DIR, "event_log.csv")
predictions_file = os.path.join(KNOWLEDGE_DIR, "predictions.csv")

# --- IMPORT ALL THREE planners ---
from plan import plan_mape, plan_drift, plan_simple_switch

models_dir = "models"

# Initialize PyRAPL for MAPE-K energy monitoring
pyRAPL.setup()

def _clear_sibling_active_aliases(keep_name, prefix):
    """Remove 'active' from every OTHER registered model sharing this prefix.

    MLflow scopes aliases per registered model, so pointing 'active' at the
    incoming model does nothing to the one that was serving before. Setting it
    without this sweep left @active on every model that had ever served -
    linear, lstm and svm all carried it simultaneously - which breaks the thing
    the alias exists to answer: HarmonE's Current Model Repository holds exactly
    one live model, so exactly one registered model should be @active.
    """
    try:
        page = _mlflow_client.search_registered_models(max_results=100)
        while True:
            for rm in page:
                if rm.name == keep_name or not rm.name.startswith(prefix):
                    continue
                if not (rm.aliases or {}).get("active"):
                    continue
                try:
                    _mlflow_client.delete_registered_model_alias(rm.name, "active")
                    logging.info(f"[MLFLOW] cleared stale 'active' alias from '{rm.name}'")
                except Exception as e:
                    logging.warning(f"[MLFLOW] Could not clear 'active' on '{rm.name}': {e}")
            token = getattr(page, "token", None)
            if not token:
                break
            page = _mlflow_client.search_registered_models(max_results=100, page_token=token)
    except Exception as e:
        logging.warning(f"[MLFLOW] Could not sweep stale 'active' aliases: {e}")

def set_active_model_alias(model_name):
    """Maps model.csv (the live model) onto the registry's 'active' alias -
    see managed_system_regression/mape_logic/execute.py for the full rationale."""
    registered_name = f"harmone-cv-{model_name}"
    try:
        versions = _mlflow_client.search_model_versions(f"name='{registered_name}'")
        if not versions:
            return
        latest = max(versions, key=lambda v: int(v.version))
        _mlflow_client.set_registered_model_alias(registered_name, "active", latest.version)
        logging.info(f"[MLFLOW] '{registered_name}' alias 'active' -> v{latest.version}")
        _clear_sibling_active_aliases(registered_name, "harmone-cv-")
    except Exception as e:
        logging.warning(f"[MLFLOW] Failed to set active alias for '{registered_name}': {e}")

def load_mape_info():
    """Loads the mape_info JSON file with event counters and energy tracking."""
    try:
        with open(mape_info_file, "r") as f:
            info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        info = {
            "last_line": 0,
            "current_energy_threshold": 0.6,
            "ema_scores": {"yolo_n": 0.5, "yolo_s": 0.5, "yolo_m": 0.5},
            "recovery_cycles": 0
        }
    
    # Ensure event counters exist
    if "event_counters" not in info:
        info["event_counters"] = {
            "model_switches": 0,
            "retrains": 0,
            "vmr_events": 0,
            "mape_k_energy_uJ": 0.0
        }
    
    # Ensure simple switch counters exist (separate from MAPE counters)
    if "simple_switch_counters" not in info:
        info["simple_switch_counters"] = {
            "simple_switches": 0
        }
    
    return info

def save_mape_info(data):
    """Saves data to the mape_info JSON file.

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

def get_last_prediction_line():
    try:
        with open(predictions_file, "r") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0

def log_event(event_type, model=None, version=None, details=None):
    last_line = get_last_prediction_line()
    log_entry = {
        "event_type": event_type,
        "last_line": last_line,
        "model": model or "",
        "version": version or "",
        "details": details or ""
    }
    file_exists = os.path.isfile(event_log_file)
    with open(event_log_file, "a", newline="") as csvfile:
        fieldnames = ["event_type", "last_line", "model", "version", "details"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

def record_event(event_type, energy_consumed=0.0, details=None):
    """Record an event and update counters."""
    info = load_mape_info()
    
    # Update counters
    if event_type == "switch":
        info["event_counters"]["model_switches"] += 1
        logging.info(f"📊 Event recorded: Model switch #{info['event_counters']['model_switches']}")
    elif event_type == "retrain":
        info["event_counters"]["retrains"] += 1
        logging.info(f"📊 Event recorded: Retrain #{info['event_counters']['retrains']}")
    elif event_type == "vmr":
        info["event_counters"]["vmr_events"] += 1
        logging.info(f"📊 Event recorded: VMR event #{info['event_counters']['vmr_events']}")
    
    # Add MAPE-K energy consumption
    info["event_counters"]["mape_k_energy_uJ"] += energy_consumed
    
    if details:
        logging.info(f"📊 Event details: {details}")
    if energy_consumed > 0:
        logging.info(f"⚡ MAPE-K energy consumed: {energy_consumed:.2f} µJ (Total: {info['event_counters']['mape_k_energy_uJ']:.2f} µJ)")

    save_mape_info(info)
    return info

def log_adaptation_event(event_type, energy_uJ, info, details=None, extra_tags=None):
    """
    Logs a single adaptation decision (switch / vmr / retrain) as a step on this
    process's shared 'harmone-adaptations' session run - see
    managed_system_regression/mape_logic/execute.py for the full rationale
    (this mirrors that implementation for the CV pipeline).
    """
    try:
        run_id = _get_session_run_id()
        # Wall-clock MILLISECONDS as `step`: this is what makes the two
        # processes' metrics (continuous telemetry from run_managed_system.py,
        # discrete events from here) land on one shared, comparable axis. Must
        # match the unit run_managed_system.py uses, or the two series end up on
        # different scales and the overlay stops lining up.
        step = int(time.time() * 1000)

        last_score = info.get("last_score_check") or {}
        last_drift = info.get("last_drift_check") or {}

        cumulative_energy = info["event_counters"]["mape_k_energy_uJ"]
        metrics = {
            "adaptation_energy_uJ": energy_uJ,
            "cumulative_switches": info["event_counters"]["model_switches"],
            "cumulative_retrains": info["event_counters"]["retrains"],
            "cumulative_vmr_events": info["event_counters"]["vmr_events"],
            "cumulative_mape_k_energy_uJ": cumulative_energy,
            # See managed_system_regression/mape_logic/execute.py: logged at the
            # same value as cumulative_mape_k_energy_uJ so the event marker sits
            # directly on the energy curve when both are overlaid on one chart.
            f"event_{event_type}": cumulative_energy,
        }
        if last_score:
            metrics["triggering_score"] = last_score.get("score", 0.0)
            metrics["score_threshold"] = last_score.get("min_score_threshold", 0.0)
            metrics["energy_normalized"] = last_score.get("energy_normalized", 0.0)
        if last_drift:
            metrics["kl_divergence"] = last_drift.get("kl_div", 0.0)
            metrics["kl_threshold"] = last_drift.get("drift_threshold", 0.0)

        for key, value in metrics.items():
            _mlflow_client.log_metric(run_id, key, value, step=step)

        tags = {f"event_{step}_type": event_type}
        if extra_tags:
            tags.update({f"event_{step}_{k}": str(v) for k, v in extra_tags.items()})
        if details:
            tags[f"event_{step}_details"] = str(details)[:250]
        for key, value in tags.items():
            _mlflow_client.set_tag(run_id, key, value)
    except Exception as e:
        logging.warning(f"[MLFLOW] Failed to log adaptation event '{event_type}': {e}")

def record_simple_switch():
    """Record a simple switch event (no energy tracking, just count)."""
    info = load_mape_info()
    
    # Update simple switch counter
    info["simple_switch_counters"]["simple_switches"] += 1
    
    logging.info(f"📊 Simple switch recorded: #{info['simple_switch_counters']['simple_switches']}")
    
    save_mape_info(info)

def execute_mape(trigger="local"):
    """Execute a model switch based on the MAPE plan."""
    # Start energy monitoring for MAPE-K loop
    energy_meter = pyRAPL.Measurement("mape_k_cv_execution")
    energy_meter.begin()
    
    print("[MAPE-EXEC] Planning model switch...")
    
    # Pass the trigger down to the planner
    decision = plan_mape(trigger=trigger)
    
    if not decision:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("switch", energy_consumed, "No switch needed - planning returned no decision")
        print("[MAPE-EXEC] No model switch needed.")
        return

    print(f"[MAPE-EXEC] Executing switch to model: {decision.upper()}")
    
    try:
        # Get current model for logging
        try:
            with open(model_file, "r") as f:
                old_model = f.read().strip()
        except FileNotFoundError:
            old_model = "unknown"
        
        with open(model_file, "w") as f:
            f.write(decision)
        
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        
        # Record both old CSV log and new event counter
        log_event("switch", model=decision)
        info = record_event("switch", energy_consumed, f"Model switched from {old_model} to {decision}")
        log_adaptation_event("switch", energy_consumed, info, details=f"{old_model} -> {decision}",
                              extra_tags={"from_model": old_model, "to_model": decision, "trigger": trigger})
        set_active_model_alias(decision)

        print(f"⚡ Switched active model to {decision.upper()}")
        
    except Exception as e:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("switch", energy_consumed, f"Failed to write model file: {e}")
        print(f"[MAPE-EXEC] Error switching model: {e}")

def execute_drift(trigger="local"):
    """Execute the drift response: switch to a previous version or trigger retraining."""
    # Start energy monitoring for MAPE-K loop
    energy_meter = pyRAPL.Measurement("mape_k_cv_drift_execution")
    energy_meter.begin()
    
    print("[DRIFT-EXEC] Planning drift response...")
    
    # Pass the trigger down to the planner
    decision = plan_drift(trigger=trigger)
    
    if not decision:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("vmr", energy_consumed, "No drift action needed")
        print("[DRIFT-EXEC] No drift action needed.")
        return

    action = decision.get("action")
    print(f"[DRIFT-EXEC] Drift action planned: {action}")

    if action == "switch_version":
        version_path = decision["version_path"]
        if not os.path.exists(version_path):
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("vmr", energy_consumed, f"Version path does not exist: {version_path}")
            print(f"[DRIFT-EXEC] Error: Version path '{version_path}' does not exist. Cannot switch.")
            return

        base_name_match = re.search(r'(yolo_[nsm])', os.path.basename(version_path))
        if not base_name_match:
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("vmr", energy_consumed, f"Could not determine base model name from {version_path}")
            print(f"[DRIFT-EXEC] Error: Could not determine base model name from '{version_path}'.")
            return
        
        base_name = base_name_match.group(1)
        destination_path = os.path.join(models_dir, f"{base_name}.pt")
        
        try:
            shutil.copy(version_path, destination_path)
            print(f"[DRIFT-EXEC] Copied '{version_path}' to '{destination_path}'.")
            
            with open(model_file, "w") as f:
                f.write(base_name)
            
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            
            # Record both old CSV log and new event counter
            log_event("vmr", model=base_name, version=os.path.basename(version_path),
                     details=f"Switched to versioned model at {version_path}")
            info = record_event("vmr", energy_consumed, f"VMR: Switched to version {version_path}")
            log_adaptation_event("vmr", energy_consumed, info, details=f"Reused version {version_path}",
                                  extra_tags={"model_name": base_name, "version_path": version_path, "trigger": trigger})
            set_active_model_alias(base_name)

            print(f"⚡ Switched active model to version: {os.path.basename(version_path)}")
            
            # Inflate EMA score for stability
            print(f"[DRIFT-EXEC] Inflating EMA score for {base_name.upper()} to ensure stability...")
            mape_info = load_mape_info()
            current_score = mape_info["ema_scores"].get(base_name, 0.5)
            new_score = min(1.0, current_score + 0.1)
            mape_info["ema_scores"][base_name] = new_score
            save_mape_info(mape_info)
            print(f"[DRIFT-EXEC] EMA score for {base_name.upper()} updated from {current_score:.4f} to {new_score:.4f}.")
            time.sleep(20)  # Reduced simulation time
            
        except Exception as e:
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("vmr", energy_consumed, f"Failed to copy versioned model: {e}")
            print(f"[DRIFT-EXEC] Error copying versioned model: {e}")

    elif action == "retrain":
        print("[DRIFT-EXEC] Triggering retraining via MLflow Project...")
        try:
            submitted_run = mlflow.projects.run(
                uri=os.path.join(BASE_DIR, ".."),
                entry_point="retrain",
                experiment_name="harmonica-cv",
                env_manager="local",
                synchronous=True,
            )

            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0

            # Look up exactly which registered model+version this retrain
            # produced, so it's visible directly on the session run's tags.
            registered_model_info = "unknown"
            try:
                versions = _mlflow_client.search_model_versions(f"run_id='{submitted_run.run_id}'")
                if versions:
                    registered_model_info = f"{versions[0].name} v{versions[0].version}"
            except Exception as e:
                logging.warning(f"[MLFLOW] Could not resolve registered model for run {submitted_run.run_id}: {e}")

            # Record both old CSV log and new event counter, cross-referencing the MLflow run
            log_event("retrain", details=f"Retraining triggered by drift detection (mlflow run_id={submitted_run.run_id}).")
            info = record_event("retrain", energy_consumed, f"Model retrained due to drift (mlflow run_id={submitted_run.run_id})")
            log_adaptation_event("retrain", energy_consumed, info, details="Retrained due to drift",
                                  extra_tags={"training_run_id": submitted_run.run_id, "trigger": trigger,
                                              "registered_model": registered_model_info})
            try:
                with open(model_file, "r") as f:
                    retrained_model_name = f.read().strip().split('_v')[0]
                set_active_model_alias(retrained_model_name)
            except FileNotFoundError:
                pass

            time.sleep(20) # Reduced simulation time

        except Exception as e:
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("retrain", energy_consumed, f"Retraining failed: {e}")
            print(f"[DRIFT-EXEC] Error during retraining: {e}")

def execute_simple_switch(trigger="local"):
    """Executes a simple model switch based on the confidence baseline plan."""
    logging.info("Executing Simple Switch (Confidence Baseline)...")
    
    # Call the simple planner
    decision = plan_simple_switch() 
    
    if not decision:
        logging.info("EXECUTE (Simple Switch): No action needed.")
        return

    logging.info(f"⚡ EXECUTE (Simple Switch): Switching model to {decision.upper()} in {model_file}")
    
    try:
        # Get current model for logging
        try:
            with open(model_file, "r") as f:
                old_model = f.read().strip()
        except FileNotFoundError:
            old_model = "unknown"
        
        with open(model_file, "w") as f:
            f.write(decision)
        
        # Record both old CSV log and new simple switch counter
        log_event("switch", model=decision, details="confidence_baseline_switch")
        record_simple_switch()
        
        logging.info(f"📊 Simple switch: {old_model} → {decision}")
        logging.info("EXECUTE (Simple Switch): Model switch successful.")
        
    except Exception as e:
        logging.error(f"EXECUTE (Simple Switch): Failed to write to {model_file}: {e}")