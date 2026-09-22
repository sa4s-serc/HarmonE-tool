import os
import shutil
import logging
import sys
import json
import time
import mlflow
from mlflow.tracking import MlflowClient
from plan import plan_mape, plan_drift, plan_simple_switch

# Define the base directory dynamically based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", ".."))
import energy_utils as pyRAPL
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")

MODEL_FILE = os.path.join(KNOWLEDGE_DIR, "model.csv")
MAPE_INFO_FILE = os.path.join(KNOWLEDGE_DIR, "mape_info.json")
SESSION_RUN_ID_FILE = os.path.join(KNOWLEDGE_DIR, "mlflow_session_run_id.txt")

# mlflow.projects.run() below pre-creates its run against whatever tracking URI
# *this* process is using, then hands the run ID to the retrain.py subprocess via
# env var. We must set it here (the caller), not just inside retrain.py, or the
# child looks for that run ID in the wrong store and crashes.
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, '..', '..', 'mlflow.db')}")

# --- Shared adaptation timeline ---
# run_managed_system.py (a *different* OS process) creates one MLflow run per
# managed-system session and writes its run_id to SESSION_RUN_ID_FILE; it logs
# continuous telemetry (score/energy) onto that run as it polls monitor_mape().
# We append our own adaptation events (switch/vmr/retrain) to that SAME run_id
# via MlflowClient, so both land on one native, chartable MLflow timeline
# instead of two disconnected ones. MlflowClient (not mlflow.start_run) is used
# deliberately: two independent processes are appending to one run concurrently.
# One experiment per managed system - see run_managed_system.py.
ADAPTATIONS_EXPERIMENT = "harmonica-regression"
_mlflow_client = MlflowClient()
_session_run_id = None

def _get_local_approach():
    """Reads the local approach.conf (e.g. 'harmone_acp') for context on which
    adaptation strategy/policy was active during this session."""
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
            # Fallback for standalone use (e.g. tests driving execute.py directly
            # without run_managed_system.py): still track events, just without
            # the unified continuous-telemetry timeline.
            exp_id = _active_experiment_id(ADAPTATIONS_EXPERIMENT)
            run = _mlflow_client.create_run(experiment_id=exp_id, run_name=f"session-{int(time.time())}")
            _session_run_id = run.info.run_id
            _mlflow_client.set_tag(_session_run_id, "approach", _get_local_approach())
    return _session_run_id

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [execute.py] - %(levelname)-8s - %(message)s',
    stream=sys.stdout
)

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
    """
    Maps HarmonE's Current Model Repository concept (whatever model.csv names
    is the live model) onto MLflow's Model Registry: points the 'active' alias
    at that model's latest registered version, so 'what's deployed right now'
    is answerable from the registry itself, not just a local file.
    Best-effort: a model that has never been retrained/trained through MLflow
    (e.g. a repo-provided baseline .pkl) has no registry entry yet - skip
    quietly rather than treat that as an error.
    """
    registered_name = f"harmone-regression-{model_name}"
    try:
        versions = _mlflow_client.search_model_versions(f"name='{registered_name}'")
        if not versions:
            return
        latest = max(versions, key=lambda v: int(v.version))
        _mlflow_client.set_registered_model_alias(registered_name, "active", latest.version)
        logging.info(f"[MLFLOW] '{registered_name}' alias 'active' -> v{latest.version}")
        _clear_sibling_active_aliases(registered_name, "harmone-regression-")
    except Exception as e:
        logging.warning(f"[MLFLOW] Failed to set active alias for '{registered_name}': {e}")

def load_mape_info():
    """Load MAPE info with event counters and energy tracking."""
    try:
        with open(MAPE_INFO_FILE, "r") as f:
            info = json.load(f)
    except FileNotFoundError:
        info = {}
    
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

def save_mape_info(info):
    """Save updated MAPE info including event counters.

    Written to a temp file and moved into place with os.replace(), which is
    atomic on Windows and POSIX. Three processes (inference.py via monitor,
    manage.py via analyse/execute) read-modify-write this file concurrently;
    a plain open(path, "w") truncates it to zero before the new bytes land,
    so a reader landing in that window gets an empty file. Measured at a 25%
    failure rate under sustained concurrent writes.
    """
    tmp_path = "%s.tmp%s" % (MAPE_INFO_FILE, os.getpid())
    with open(tmp_path, "w") as f:
        json.dump(info, f, indent=4)
        f.flush()
        os.fsync(f.fileno())

    # os.replace raises WinError 5 if anything else currently has the
    # destination open, and the dashboard polls this file every couple of
    # seconds. Those read handles live for microseconds, so a short retry
    # clears them; if it somehow does not, fall back to the direct write
    # rather than lose the update or crash the loop.
    for _ in range(40):
        try:
            os.replace(tmp_path, MAPE_INFO_FILE)
            return
        except PermissionError:
            time.sleep(0.02)

    # Still locked after ~0.8s. Skip this snapshot rather than fall back to a
    # truncating write: the next tick rewrites the whole file anyway, whereas a
    # torn write can leave state that load_mape_info cannot parse.
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("[MAPE] WARNING: could not update %s (locked); skipping this snapshot" % MAPE_INFO_FILE)

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
    process's shared 'harmone-adaptations' session run.

    This is deliberately separate from the detailed per-retrain training run
    (logged by retrain.py under 'harmone-regression'): that run captures what a
    retrain produced, this one captures the *adaptation timeline itself* - which
    tactic fired, when, at what decision cost, and against what actual trigger
    (the score/threshold or KL divergence that analyse.py computed, persisted
    into mape_info.json as last_score_check/last_drift_check) - so the sequence
    of switches, retrains, and version reuse, and the policy context behind each,
    can be queried and plotted directly from MLflow as a real timeline, mirroring
    HarmonE's own evaluation of adaptation frequency/efficiency rather than just
    model quality.
    A logging failure here must never break the actual adaptation - it's
    observability, not control flow.
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
            # Logged at the SAME value as cumulative_mape_k_energy_uJ (not a flat
            # 1) so that overlaying both on one MLflow chart places this event's
            # marker directly on the energy curve, at the point it happened - a
            # flat 1 would be invisible next to a metric climbing into the
            # thousands. This is what makes a combined "Add chart" view actually
            # show R/S/V events on the energy line, like the paper's Figure 4.
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

        # Tags aren't step-indexed, so each event gets its own uniquely-named
        # tag key - this keeps the full per-event text context (which model,
        # which version path) inspectable without overwriting prior events'.
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

# --- Tactic Execution ---
def execute_mape(trigger="local"):
    """Switch to the best model based on planning."""
    # Start energy monitoring for MAPE-K loop
    energy_meter = pyRAPL.Measurement("mape_k_execution")
    energy_meter.begin()
    
    logging.info("Executing MAPE (model switch)...")
    
    # Call plan_mape with the trigger
    decision = plan_mape(trigger=trigger)
    
    if not decision:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("switch", energy_consumed, "No switch needed - planning returned no decision")
        logging.info("EXECUTE: No action needed (plan was empty).")
        return

    logging.info(f"⚡ EXECUTE: Switching model to {decision.upper()} in {MODEL_FILE}")
    try:
        # Get current model for logging
        try:
            with open(MODEL_FILE, "r") as f:
                old_model = f.read().strip()
        except FileNotFoundError:
            old_model = "unknown"
        
        with open(MODEL_FILE, "w") as f:
            f.write(decision)
        
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        
        # Record the switch event
        info = record_event("switch", energy_consumed, f"Model switched from {old_model} to {decision}")
        log_adaptation_event("switch", energy_consumed, info, details=f"{old_model} -> {decision}",
                              extra_tags={"from_model": old_model, "to_model": decision, "trigger": trigger})
        set_active_model_alias(decision)

        logging.info("EXECUTE: Model switch successful.")
    except Exception as e:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("switch", energy_consumed, f"Failed to write model file: {e}")
        logging.error(f"EXECUTE: Failed to write to {MODEL_FILE}: {e}")

def execute_drift(trigger="local"):
    """Replaces model with best version or retrains if necessary."""
    # Start energy monitoring for MAPE-K loop
    energy_meter = pyRAPL.Measurement("mape_k_drift_execution")
    energy_meter.begin()
    
    logging.info("Executing Drift handling...")
    
    decision = plan_drift(trigger=trigger)
    
    if not decision:
        energy_meter.end()
        energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
        record_event("vmr", energy_consumed, "No drift action needed")
        logging.info("EXECUTE (Drift): No action needed.")
        return

    if decision["action"] == "replace":
        best_version_path = decision["version"]
        # Basic validation
        if not best_version_path or "version" not in best_version_path:
             energy_meter.end()
             energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
             record_event("vmr", energy_consumed, f"Invalid version path: {best_version_path}")
             logging.warning(f"EXECUTE (Drift): Invalid version path provided: {best_version_path}")
             return

        model_name = os.path.basename(os.path.dirname(best_version_path))
        model_extension = ".pkl" if model_name in ["linear", "svm"] else ".pth"
        model_target_path = os.path.join(BASE_DIR, "..", "models", f"{model_name}{model_extension}")
        
        try:
            shutil.copy(best_version_path, model_target_path)
            
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            
            # Record VMR event
            info = record_event("vmr", energy_consumed, f"VMR: Switched to version {best_version_path}")
            log_adaptation_event("vmr", energy_consumed, info, details=f"Reused version {best_version_path}",
                                  extra_tags={"model_name": model_name, "version_path": best_version_path, "trigger": trigger})
            set_active_model_alias(model_name)

            logging.info(f"✔ EXECUTE (Drift): Switched to lower KL divergence model: {best_version_path}")
        except Exception as e:
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("vmr", energy_consumed, f"Failed to copy model: {e}")
            logging.error(f"EXECUTE (Drift): Failed to copy model: {e}")

    elif decision["action"] == "retrain":
        logging.info("🚀 EXECUTE (Drift): Triggering retraining via MLflow Project...")
        try:
            # Ensure retrain.py exists
            if os.path.exists("retrain.py"):
                submitted_run = mlflow.projects.run(
                    uri=os.path.join(BASE_DIR, ".."),
                    entry_point="retrain",
                    experiment_name="harmonica-regression",
                    env_manager="local",
                    synchronous=True,
                )

                energy_meter.end()
                energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0

                # Look up exactly which registered model+version this retrain
                # produced, so it's visible directly on the session run's tags
                # rather than requiring a manual hunt through the Model Registry.
                registered_model_info = "unknown"
                try:
                    versions = _mlflow_client.search_model_versions(f"run_id='{submitted_run.run_id}'")
                    if versions:
                        registered_model_info = f"{versions[0].name} v{versions[0].version}"
                except Exception as e:
                    logging.warning(f"[MLFLOW] Could not resolve registered model for run {submitted_run.run_id}: {e}")

                # Record retrain event, cross-referencing the MLflow run
                info = record_event("retrain", energy_consumed, f"Model retrained due to drift (mlflow run_id={submitted_run.run_id})")
                log_adaptation_event("retrain", energy_consumed, info, details="Retrained due to drift",
                                      extra_tags={"training_run_id": submitted_run.run_id, "trigger": trigger,
                                                  "registered_model": registered_model_info})
                # Retraining refreshes the active model's weights in place (it
                # doesn't change WHICH model is active), so point 'active' at
                # the version this retrain just registered.
                try:
                    with open(MODEL_FILE, "r") as f:
                        retrained_model_name = f.read().strip()
                    set_active_model_alias(retrained_model_name)
                except FileNotFoundError:
                    pass

                logging.info(f"EXECUTE (Drift): Retraining finished. MLflow run_id={submitted_run.run_id}")
            else:
                energy_meter.end()
                energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
                record_event("retrain", energy_consumed, "Retrain.py not found")
                logging.warning("EXECUTE (Drift): 'retrain.py' not found. Skipping.")
        except Exception as e:
            energy_meter.end()
            energy_consumed = energy_meter.result.pkg[0] if energy_meter.result.pkg else 0.0
            record_event("retrain", energy_consumed, f"Retraining failed: {e}")
            logging.error(f"EXECUTE (Drift): Retraining failed: {e}")

def execute_simple_switch(trigger="local"):
    """Switches model based on the simple_switch plan."""
    logging.info("Executing Simple Switch (R² Baseline)...")
    
    decision = plan_simple_switch(trigger=trigger) # Call the new planner
    
    if not decision:
        logging.info("EXECUTE (Simple Switch): No action needed.")
        return

    logging.info(f"⚡ EXECUTE (Simple Switch): Switching model to {decision.upper()} in {MODEL_FILE}")
    try:
        # Get current model for logging
        try:
            with open(MODEL_FILE, "r") as f:
                old_model = f.read().strip()
        except FileNotFoundError:
            old_model = "unknown"
        
        with open(MODEL_FILE, "w") as f:
            f.write(decision)
        
        # Record the simple switch event (no energy tracking)
        record_simple_switch()
        
        logging.info(f"📊 Simple switch: {old_model} → {decision}")
        logging.info("EXECUTE (Simple Switch): Model switch successful.")
    except Exception as e:
        logging.error(f"EXECUTE (Simple Switch): Failed to write to {MODEL_FILE}: {e}")