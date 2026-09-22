import time
import csv
import json
import statistics
import threading
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import os
import sys
import subprocess
import psutil
import shutil
import zipfile
import mlflow
from mlflow.tracking import MlflowClient

# Must run before anything is spawned: run_managed_system.py and, through it,
# inference.py / manage.py inherit this process's environment.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from console_utils import force_utf8_console
force_utf8_console()

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ACP] - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# --- MLflow tracking setup (absolute path so it matches retrain.py/execute.py
# regardless of which directory each process was actually launched from) ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_TRACKING_DB_PATH = os.path.join(APP_DIR, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_TRACKING_DB_PATH}")
MLFLOW_UI_URL = "http://localhost:5001"

# --- In-Memory Knowledge Base (K) ---
KNOWLEDGE_BASE = {
    "policies": {},
    "telemetry_data": {},
    "intervention_logs": {}
}

# --- Latest decision as computed by THIS server -----------------------------
# In ACP mode - the path the dashboard drives - plan_mape(trigger="acp")
# explicitly SKIPS analyse_mape(), and analyse_mape() is the only writer of
# mape_info.json's last_score_check. So that field is a fossil from the last
# harmone_local run and never moves while the dashboard is driving, which is
# why the decision panel looked frozen. The server is the analyser here, so it
# records what it actually decided.
LAST_ACP_DECISION = {"at": None, "policy_id": None, "score_check": None, "drift_check": None}

# --- Per-run baseline into predictions.csv ----------------------------------
# predictions.csv accumulates across runs, so sampling its tail showed share,
# latency and energy for every model before a run had even been started. The
# row count at start-of-run is recorded here so those stats describe THIS run.
# None means "no run started in this server session" - the UI then shows no
# per-run rates rather than last run's.
RUN_BASELINE = {"regression": None, "cv": None}


def _prediction_row_count(system):
    try:
        with open(_knowledge_path(system, "predictions.csv")) as f:
            return max(sum(1 for _ in f) - 1, 0)  # minus header
    except OSError:
        return 0

# --- Manual retrain-trigger state (per managed system) ---
RETRAIN_STATE = {
    "regression": {"status": "idle", "run_id": None, "error": None},
    "cv": {"status": "idle", "run_id": None, "error": None},
}

# ============================================================
# -----------------  HELPER FUNCTIONS  ------------------------
# ============================================================

def get_historical_average(policy_id, metric_key):
    """Calculates the historical average for a given metric from the Knowledge Base."""
    if policy_id not in KNOWLEDGE_BASE["telemetry_data"]:
        return None
    
    metric_values = [
        record[metric_key] 
        for record in KNOWLEDGE_BASE["telemetry_data"][policy_id] 
        if metric_key in record
    ]
    
    return statistics.mean(metric_values) if metric_values else None


def analyze_telemetry(policy, metric_value, metric_key):
    """
    Analyzes telemetry against the policy's primary adaptation boundary.
    Only handles the *primary* boundary here. Secondary boundaries are handled separately.
    """
    boundary = policy.get("adaptation_boundary", {})
    condition = boundary.get("condition")
    static_threshold = boundary.get("threshold")
    dynamic_logic_str = boundary.get("dynamic_logic")

    violation = False

    # ---- STATIC THRESHOLD CHECK ----
    if condition == "GREATER_THAN":
        if metric_value > static_threshold:
            logging.info(f"[ANALYZE] Static threshold VIOLATED: {metric_value} > {static_threshold}")
            violation = True
        else:
            logging.info(f"[ANALYZE] Static threshold NOT violated: {metric_value} <= {static_threshold}")
            return False

    elif condition == "LESS_THAN":
        if metric_value < static_threshold:
            logging.info(f"[ANALYZE] Static threshold VIOLATED: {metric_value} < {static_threshold}")
            violation = True
        else:
            logging.info(f"[ANALYZE] Static threshold NOT violated: {metric_value} >= {static_threshold}")
            return False

    else:
        logging.warning(f"[ANALYZE] Unknown condition '{condition}'. No action taken.")
        return False

    # ---- DYNAMIC LOGIC CHECK (OPTIONAL) ----
    if violation and dynamic_logic_str:
        logging.info(f"[ANALYZE] Checking dynamic logic: '{dynamic_logic_str}'")
        avg = get_historical_average(policy['policy_id'], metric_key)

        if avg is not None:
            try:
                # parse "historic_avg * factor"
                factor = float(dynamic_logic_str.split('*')[1].strip())
                dynamic_threshold = avg * factor

                if (condition == "GREATER_THAN" and metric_value > dynamic_threshold) or \
                   (condition == "LESS_THAN" and metric_value < dynamic_threshold):
                    logging.info(f"[ANALYZE] Dynamic condition MET: {metric_value} vs {dynamic_threshold}")
                    return True
                else:
                    logging.info(f"[ANALYZE] Dynamic condition NOT MET: {metric_value} vs {dynamic_threshold}")
                    return False

            except:
                logging.error("[ANALYZE] Error parsing dynamic logic string.")
                return False

        else:
            logging.warning("[ANALYZE] No historical data for dynamic logic. Using static threshold only.")
            return True

    return violation


def plan_and_execute(policy, trigger_value, trigger_metric=None):
    """Executes the given tactic from a policy."""
    tactics = sorted(policy.get("tactics", []), key=lambda t: t['priority'])
    if not tactics:
        logging.warning(f"[PLAN] No tactics found for policy '{policy['policy_id']}'")
        return

    selected_tactic = tactics[0]

    logging.info(f"[PLAN] Tactic selected: '{selected_tactic['tactic_id']}'")

    intervention_record = {
        "timestamp": time.time(),
        "policy_id": policy["policy_id"],
        "tactic_id": selected_tactic["tactic_id"],
        "trigger_value": trigger_value,
        "trigger_metric": trigger_metric or policy.get("quality_attribute", "unknown"),
        "status": "TRIGGERED"
    }

    KNOWLEDGE_BASE["intervention_logs"].setdefault(policy["policy_id"], []).append(intervention_record)

    payload = {
        "tactic_id": selected_tactic["tactic_id"],
        "trigger_value": trigger_value,
        "trigger_timestamp": intervention_record["timestamp"]
    }

    try:
        endpoint = selected_tactic["tactic_endpoint"]
        logging.info(f"[EXECUTE] Posting to {endpoint} with payload {payload}")
        response = requests.post(endpoint, json=payload, timeout=5)

        if response.status_code == 200:
            logging.info(f"[EXECUTE] SUCCESS: {response.json()}")
            intervention_record["status"] = "CONFIRMED_SUCCESS"
        else:
            logging.error(f"[EXECUTE] FAILED: {response.text}")
            intervention_record["status"] = "CONFIRMED_FAILED"

    except Exception as e:
        logging.error(f"[EXECUTE] ERROR during request: {e}")
        intervention_record["status"] = "REQUEST_FAILED"

# ============================================================
# ---- PERIODIC SECONDARY BOUNDARY CHECKER (KL DRIFT) --------
# ============================================================

def periodic_secondary_checks(interval=30):
    """
    Periodically evaluate secondary boundaries (like KL divergence)
    even when the primary metric (score) violates often.
    """
    logging.info(f"[SECONDARY] Starting periodic evaluator every {interval}s")

    while True:
        time.sleep(interval)

        for policy_id, policy in KNOWLEDGE_BASE["policies"].items():

            secondary = policy.get("secondary_boundaries", [])
            if not secondary:
                continue

            telemetry_history = KNOWLEDGE_BASE["telemetry_data"].get(policy_id, [])
            if not telemetry_history:
                continue

            latest = telemetry_history[-1]

            for sec in secondary:
                qa = sec["quality_attribute"]
                if qa not in latest:
                    continue

                value = latest[qa]
                condition = sec["condition"]
                threshold = sec["threshold"]
                tactic_id = sec["tactic_id"]

                violated = (
                    (condition == "GREATER_THAN" and value > threshold)
                    or
                    (condition == "LESS_THAN" and value < threshold)
                )

                if violated:
                    logging.info(f"[SECONDARY] KL DRIFT DETECTED: {qa}={value} {condition} {threshold}")

                    endpoint = policy["tactics"][0]["tactic_endpoint"]

                    drift_policy = {
                        "policy_id": policy_id,
                        "tactics": [
                            {
                                "tactic_id": tactic_id,
                                "priority": 1,
                                "tactic_endpoint": endpoint
                            }
                        ]
                    }

                    plan_and_execute(drift_policy, value, qa)

# ============================================================
# ---------------------- API ENDPOINTS ------------------------
# ============================================================

@app.route('/api/policy', methods=['POST'])
def add_policy():
    policy = request.json
    policy_id = policy["policy_id"]

    KNOWLEDGE_BASE["policies"][policy_id] = policy

    logging.info(f"[KNOWLEDGE] Policy '{policy_id}' added.")
    return jsonify({"message": "Policy added"}), 201


def _reset_last_decision():
    """A decision describes one run, so it must not outlive it.

    LAST_ACP_DECISION is module state. Without clearing it, a freshly opened
    dashboard showed the PREVIOUS run's reasoning as though it were current -
    the panel was never blank on a fresh start.
    """
    LAST_ACP_DECISION.update({"at": None, "policy_id": None,
                              "score_check": None, "drift_check": None})


def _record_acp_decision(policy_id, policy, telemetry, metric_key, value, violation):
    """Snapshot the analysis this server just performed, in the same shape the
    dashboard already reads from analyse.py, so the decision panel shows live
    reasoning in ACP mode instead of a stale local-mode record."""
    boundary = policy.get("adaptation_boundary", {}) or {}

    score_check = {
        "score": value,
        "min_score_threshold": boundary.get("threshold"),
        "condition": boundary.get("condition"),
        "energy_normalized": telemetry.get("normalized_energy", telemetry.get("energy_normalized")),
        "energy_threshold": telemetry.get("energy_threshold"),
        "switch_needed": bool(violation),
        "threshold_violated": metric_key if violation else None,
    }

    # Secondary boundaries are what carry drift (kl_div) in the presets.
    drift_check = None
    for sec in policy.get("secondary_boundaries", []) or []:
        qa = sec.get("quality_attribute")
        if qa not in telemetry or telemetry[qa] is None:
            continue
        v, thr = telemetry[qa], sec.get("threshold")
        breached = (v > thr) if sec.get("condition") == "GREATER_THAN" else (v < thr)
        drift_check = {"kl_div": v, "drift_threshold": thr, "drift_detected": bool(breached)}
        break

    LAST_ACP_DECISION.update({
        "at": time.time(),
        "policy_id": policy_id,
        "score_check": score_check,
        "drift_check": drift_check or LAST_ACP_DECISION.get("drift_check"),
    })


@app.route('/api/telemetry', methods=['POST'])
def receive_telemetry():
    telemetry = request.json
    logging.info(f"[MONITOR] Received telemetry: {telemetry}")

    policy_found = False

    for policy_id, policy in KNOWLEDGE_BASE["policies"].items():

        metric_key = policy.get("quality_attribute")
        if metric_key not in telemetry:
            continue

        policy_found = True

        KNOWLEDGE_BASE["telemetry_data"].setdefault(policy_id, []).append(telemetry)
        logging.info(f"[KNOWLEDGE] Stored telemetry under '{policy_id}'")

        value = telemetry[metric_key]

        # PRIMARY CHECK (score)
        primary_violation = analyze_telemetry(policy, value, metric_key)
        _record_acp_decision(policy_id, policy, telemetry, metric_key, value, primary_violation)

        if primary_violation:
            plan_and_execute(policy, value, metric_key)
            continue

        logging.info(f"[ANALYZE] No primary violation for '{policy_id}'")

    if not policy_found:
        KNOWLEDGE_BASE["telemetry_data"].setdefault("unassigned", []).append(telemetry)

    return jsonify({"message": "Telemetry received"}), 200


@app.route('/api/knowledge/<policy_id>', methods=['GET'])
def get_knowledge(policy_id):

    if policy_id == "unassigned":
        return jsonify({
            "policy": {"policy_id": "unassigned"},
            "telemetry_history": KNOWLEDGE_BASE["telemetry_data"].get("unassigned", []),
            "intervention_logs": []
        })

    return jsonify({
        "policy": KNOWLEDGE_BASE["policies"].get(policy_id),
        "telemetry_history": KNOWLEDGE_BASE["telemetry_data"].get(policy_id, []),
        "intervention_logs": KNOWLEDGE_BASE["intervention_logs"].get(policy_id, [])
    })


@app.route('/api/write-approach', methods=['POST'])
def write_approach_config():
    data = request.json
    approach = data.get("approach")

    # First, stop any running managed system processes
    try:
        stop_managed_system()
        time.sleep(2)  # Give processes time to terminate
    except Exception as e:
        logging.warning(f"Error during cleanup: {e}")

    mapping = {
        'reg_harmone_score': 'reg_harmone',
        'reg_switch_r2': 'reg_switch', 
        'reg_single': 'reg_single',
        'cv_harmone_score': 'cv_harmone',
        'cv_switch_conf': 'cv_switch',
        'cv_single': 'cv_single'
    }

    config_val = mapping.get(approach, approach)

    with open("approach.conf", "w") as f:
        f.write(config_val)

    # Clear knowledge base when switching approaches
    global KNOWLEDGE_BASE
    KNOWLEDGE_BASE = {
        "policies": {},
        "telemetry_data": {}, 
        "intervention_logs": {}
    }

    logging.info(f"Approach updated: {config_val}, knowledge base cleared")
    return jsonify({"message": "Approach written and system cleaned"}), 200


@app.route('/api/save-policy', methods=['POST'])
def save_policy_file():
    policy = request.json
    pid = policy["policy_id"]

    os.makedirs("policies", exist_ok=True)
    with open(f"policies/{pid}.json", "w") as f:
        import json
        json.dump(policy, f, indent=2)

    return jsonify({"message": "Policy saved"}), 200


@app.route('/api/set-model', methods=['POST'])
def set_model():
    data = request.json
    model = data["model"]
    system = data["system"]

    if system == "regression":
        path = "managed_system_regression/knowledge/model.csv"
    else:
        path = "managed_system_cv/knowledge/model.csv"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        f.write(model)

    return jsonify({"message": "Model set"}), 200

def _run_retrain_project(system, project_dir):
    """Runs the managed system's MLflow Project `retrain` entry point in the
    background and updates RETRAIN_STATE as it progresses."""
    try:
        submitted_run = mlflow.projects.run(
            uri=project_dir,
            entry_point="retrain",
            parameters={},
            experiment_name=f"harmonica-{system}",
            env_manager="local",
            synchronous=False,
        )
        RETRAIN_STATE[system]["run_id"] = submitted_run.run_id
        submitted_run.wait()
        RETRAIN_STATE[system]["status"] = "finished"
        logging.info(f"[MLFLOW] Manual retrain finished for '{system}' (run_id={submitted_run.run_id})")
    except Exception as e:
        RETRAIN_STATE[system]["status"] = "failed"
        RETRAIN_STATE[system]["error"] = str(e)
        logging.error(f"[MLFLOW] Manual retrain failed for '{system}': {e}")


@app.route('/api/trigger-retrain', methods=['POST'])
def trigger_retrain():
    """Manually launches a retrain run (via MLflow Projects) for the given system,
    independent of drift detection. Mirrors /api/set-model's `system` convention."""
    data = request.json or {}
    system = data.get("system")

    if system not in ("regression", "cv"):
        return jsonify({"error": "system must be 'regression' or 'cv'"}), 400

    if RETRAIN_STATE[system]["status"] == "running":
        return jsonify({"status": "already_running", "run_id": RETRAIN_STATE[system]["run_id"]}), 409

    project_dir = "managed_system_regression" if system == "regression" else "managed_system_cv"
    RETRAIN_STATE[system] = {"status": "running", "run_id": None, "error": None}

    threading.Thread(target=_run_retrain_project, args=(system, project_dir), daemon=True).start()

    logging.info(f"[MLFLOW] Manual retrain triggered for '{system}'")
    return jsonify({"status": "ok", "message": f"Retrain started for {system}"}), 202


@app.route('/api/retrain-status/<system>', methods=['GET'])
def retrain_status(system):
    if system not in RETRAIN_STATE:
        return jsonify({"error": "system must be 'regression' or 'cv'"}), 400

    return jsonify({**RETRAIN_STATE[system], "mlflow_ui_url": MLFLOW_UI_URL})


@app.route('/api/mlflow-info/<system>', methods=['GET'])
def mlflow_info(system):
    """
    Gives the dashboard everything it needs to link directly into MLflow: the
    live adaptation-timeline session run (if the managed system is running),
    and the currently active model's registry entry (from model.csv) - so
    navigating from the HarmonE UI into MLflow is a click, not a URL to
    copy-paste and a manual search.
    """
    if system not in ("regression", "cv"):
        return jsonify({"error": "system must be 'regression' or 'cv'"}), 400

    project_dir = "managed_system_regression" if system == "regression" else "managed_system_cv"
    result = {"mlflow_ui_url": MLFLOW_UI_URL, "session_run_id": None, "experiment_id": None,
              "training_experiment_id": None, "active_model": None, "registered_model_name": None}

    # One experiment per managed system holds both the session runs and the
    # training runs nested under them, so these two ids are the same lookup.
    # They stay separate keys because the dashboard links to them for different
    # reasons (session timeline vs training history).
    try:
        exp = mlflow.get_experiment_by_name(f"harmonica-{system}")
        if exp:
            result["experiment_id"] = exp.experiment_id
            result["training_experiment_id"] = exp.experiment_id
    except Exception as e:
        logging.warning(f"[MLFLOW] Failed to look up harmonica-{system} experiment: {e}")

    session_file = os.path.join(project_dir, "knowledge", "mlflow_session_run_id.txt")
    if os.path.exists(session_file):
        with open(session_file) as f:
            result["session_run_id"] = f.read().strip()

    model_file = os.path.join(project_dir, "knowledge", "model.csv")
    if os.path.exists(model_file):
        with open(model_file) as f:
            active_model = f.read().strip().split('_v')[0]  # cv's model.csv may hold e.g. 'yolo_s_v3'
        result["active_model"] = active_model
        result["registered_model_name"] = f"harmone-{system}-{active_model}"

        # Resolve the 'active' alias to a concrete version so the dashboard can
        # deep-link to THE deployed model rather than the registered model's
        # page, which lists every version ever trained across every session.
        # execute.py repoints this alias on each switch/vmr/retrain.
        try:
            mv = MlflowClient().get_model_version_by_alias(result["registered_model_name"], "active")
            result["active_version"] = mv.version
        except Exception as e:
            # No alias yet (e.g. a repo-provided baseline model that has never
            # been retrained through MLflow) - the dashboard falls back to the
            # registered model page.
            logging.info(f"[MLFLOW] No 'active' alias for {result['registered_model_name']}: {e}")

    return jsonify(result)


# =============================================================================
# Knowledge Base introspection
#
# The MAPE-K loop already writes all of this to disk; none of it was reachable
# over HTTP, so the dashboard could only ever show the telemetry stream. These
# endpoints are read-only views over that state.
# =============================================================================

MODEL_CATALOGUE = {
    "regression": [
        {"name": "lstm",   "label": "LSTM",   "file": "lstm.pth",
         "blurb": "Sequence model. Most accurate, most expensive to run."},
        {"name": "svm",    "label": "SVM",    "file": "svm.pkl",
         "blurb": "Kernel regressor. Middle of the accuracy/energy tradeoff."},
        {"name": "linear", "label": "Linear", "file": "linear.pkl",
         "blurb": "Cheapest to run, degrades fastest under drift."},
    ],
    "cv": [
        {"name": "yolo_n", "label": "YOLO Nano",   "file": "yolo_n.pt",
         "blurb": "Smallest. Fastest inference, lowest accuracy."},
        {"name": "yolo_s", "label": "YOLO Small",  "file": "yolo_s.pt",
         "blurb": "Balanced accuracy against energy."},
        {"name": "yolo_m", "label": "YOLO Medium", "file": "yolo_m.pt",
         "blurb": "Largest. Most accurate, most energy per inference."},
    ],
}


def _project_dir(system):
    return "managed_system_regression" if system == "regression" else "managed_system_cv"


def _knowledge_path(system, *parts):
    return os.path.join(APP_DIR, _project_dir(system), "knowledge", *parts)


class _TransientRead(Exception):
    """A knowledge file could not be read this instant, but exists."""


def _read_json_file(path, default=None):
    """
    Read a knowledge file, tolerating a concurrent writer.

    save_mape_info() now writes via a temp file and os.replace(). That is
    atomic, but on Windows the destination is briefly locked during the
    rename, so an unlucky reader gets PermissionError. Returning {} on that
    would blank the sustainability goals, decision panel and EMA scores for a
    tick - which is exactly the intermittent emptiness this fixes. Retry
    instead, and if it still fails let the caller answer 503 so the dashboard
    keeps showing the last good values rather than wiping them.
    """
    for attempt in range(4):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return default if default is not None else {}
        except (PermissionError, OSError, json.JSONDecodeError):
            if attempt == 3:
                raise _TransientRead(path)
            time.sleep(0.03)


def _active_model(system):
    """model.csv is the Current Model Repository. CV may store 'yolo_s_v3'."""
    try:
        with open(_knowledge_path(system, "model.csv")) as f:
            return f.read().strip().split("_v")[0] or None
    except Exception:
        return None


def _tail_predictions(system, max_rows=1500, since_row=None):
    """
    Last N rows of predictions.csv as dicts.

    Read from the tail rather than the whole file: this is polled, and the
    regression log runs to tens of thousands of rows within a single session.
    The stats it feeds are therefore 'recent', not lifetime - the UI says so.
    """
    path = _knowledge_path(system, "predictions.csv")
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []
            if since_row:
                for _ in range(since_row):
                    if next(reader, None) is None:
                        break
            rows = deque(reader, maxlen=max_rows)
        return [dict(zip(header, r)) for r in rows if len(r) == len(header)]
    except Exception as e:
        logging.warning(f"[KB] Could not read predictions.csv for {system}: {e}")
        return []


def _as_float(value):
    try:
        f = float(value)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def _per_model_stats(system, since_row=None):
    """Recent inference counts, mean latency and mean energy, grouped by model."""
    rows = _tail_predictions(system, since_row=since_row)
    # CV logs 'energy_uJ'; regression logs 'energy'. Same meaning, different header.
    energy_key = "energy_uJ" if system == "cv" else "energy"

    stats = {}
    for row in rows:
        model = (row.get("model_used") or "").strip()
        if not model:
            continue
        entry = stats.setdefault(model, {"count": 0, "_t": [], "_e": []})
        entry["count"] += 1
        t = _as_float(row.get("inference_time"))
        if t is not None:
            entry["_t"].append(t)
        e = _as_float(row.get(energy_key))
        if e is not None:
            entry["_e"].append(e)

    total = sum(v["count"] for v in stats.values()) or 1
    for model, v in stats.items():
        v["share"] = round(v["count"] / total, 4)
        v["mean_inference_ms"] = round(statistics.fmean(v["_t"]) * 1000, 3) if v["_t"] else None
        v["mean_energy_uj"] = round(statistics.fmean(v["_e"]), 2) if v["_e"] else None
        del v["_t"], v["_e"]

    return stats, len(rows)


def _count_rows(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, newline="") as f:
            return max(sum(1 for _ in f) - 1, 0)  # minus header
    except Exception:
        return None


def _registry_models(system):
    """Registered models for this system, keyed by bare model name.

    The registry is the source of truth for which models EXIST. It is not
    sufficient on its own: managed_system_cv currently has no registered
    models at all while its loop happily serves yolo_n/s/m, so a
    registry-only list would render an empty Models page. Callers union this
    with what the loop can actually select.
    """
    prefix = f"harmone-{system}-"
    out = {}
    try:
        client = MlflowClient()
        page = client.search_registered_models(max_results=100)
        while True:
            for rm in page:
                if not rm.name.startswith(prefix):
                    continue
                short = rm.name[len(prefix):]
                aliases = dict(rm.aliases) if rm.aliases else {}
                try:
                    versions = client.search_model_versions(f"name='{rm.name}'")
                except Exception:
                    versions = []
                out[short] = {
                    "registered_model_name": rm.name,
                    "registry_versions": len(versions),
                    "active_alias_version": aliases.get("active"),
                }
            token = getattr(page, "token", None)
            if not token:
                break
            page = client.search_registered_models(max_results=100, page_token=token)
    except Exception as e:
        logging.warning(f"[KB] Registry listing failed for {system}: {e}")
    return out


def _model_file(system, name, spec):
    """Path to the model's weights, by catalogue spec or by matching stem."""
    models_dir = os.path.join(APP_DIR, _project_dir(system), "models")
    if spec and spec.get("file"):
        p = os.path.join(models_dir, spec["file"])
        if os.path.exists(p):
            return p
    try:
        for f in os.listdir(models_dir):
            if os.path.splitext(f)[0] == name:
                return os.path.join(models_dir, f)
    except OSError:
        pass
    return None


def _disk_versions(system, model):
    """
    The paper's Versioned Model Repository - model+data pairs kept for reuse.

    The two managed systems lay it out differently and both are supported:
      regression  versionedMR/<model>/version_N/{data.csv, <model file>}
      cv          versionedMR/<model>_vN.pt  +  <model>_vN_hist.json
    (CV pairs its model with a drift histogram rather than a training CSV.)
    """
    vmr_root = os.path.join(APP_DIR, _project_dir(system), "versionedMR")
    if not os.path.isdir(vmr_root):
        return []

    out = []

    nested = os.path.join(vmr_root, model)
    if os.path.isdir(nested):
        for entry in os.listdir(nested):
            vdir = os.path.join(nested, entry)
            if not entry.startswith("version_") or not os.path.isdir(vdir):
                continue
            try:
                number = int(entry.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            data_csv = os.path.join(vdir, "data.csv")
            out.append({
                "version": number,
                "created_at": os.path.getmtime(vdir),
                "has_training_data": os.path.exists(data_csv),
                "training_rows": _count_rows(data_csv),
            })

    if not out:
        prefix = f"{model}_v"
        for entry in os.listdir(vmr_root):
            path = os.path.join(vmr_root, entry)
            if not entry.startswith(prefix) or not os.path.isfile(path):
                continue
            stem = os.path.splitext(entry)[0]
            if stem.endswith("_hist"):
                continue  # the companion histogram, not a version of its own
            try:
                number = int(stem[len(prefix):])
            except ValueError:
                continue
            out.append({
                "version": number,
                "created_at": os.path.getmtime(path),
                "has_training_data": os.path.exists(os.path.join(vmr_root, f"{stem}_hist.json")),
                "training_rows": None,
            })

    return sorted(out, key=lambda v: v["version"], reverse=True)


@app.route('/api/approach', methods=['GET'])
def get_approach_config():
    """
    The approach currently written to approach.conf, and the system it implies.

    The dashboard otherwise only learns the active system as a side effect of
    the user clicking an approach in that browser tab, so a plain reload left
    it blank - which silently disabled the Models page, the decision inspector
    and every MLflow link. This lets the page recover that state from the
    server instead of guessing.
    """
    try:
        with open(os.path.join(APP_DIR, "approach.conf")) as f:
            approach = f.read().strip()
    except FileNotFoundError:
        return jsonify({"approach": None, "system": None})

    system = "cv" if approach.startswith("cv") else ("regression" if approach.startswith("reg") else None)
    return jsonify({"approach": approach, "system": system})


def _live_decision(key):
    """The server's own decision, if it has made one this session."""
    return LAST_ACP_DECISION.get(key) if LAST_ACP_DECISION.get("at") else None


@app.route('/api/knowledge-state/<system>', methods=['GET'])
def knowledge_state(system):
    """
    mape_info.json + thresholds.json: the loop's own working state - why the
    last decision went the way it did, and the sustainability goals it is
    being held to.
    """
    if system not in ("regression", "cv"):
        return jsonify({"error": "system must be 'regression' or 'cv'"}), 400

    try:
        info = _read_json_file(_knowledge_path(system, "mape_info.json"))
        thresholds = _read_json_file(_knowledge_path(system, "thresholds.json"))
        drift_kl = _read_json_file(_knowledge_path(system, "drift_kl.json"))
    except _TransientRead as e:
        # A writer holds the file right now. Say so instead of reporting empty
        # state, so the dashboard keeps its last good render.
        return jsonify({"error": "knowledge file busy", "path": str(e)}), 503

    best_version = drift_kl.get("best_version")
    if best_version:
        # Absolute path on the server; the UI only needs the tail of it.
        best_version = os.path.normpath(best_version).replace("\\", "/").split("versionedMR/")[-1]

    score_check = _live_decision("score_check") or info.get("last_score_check")

    # Drift is a special case. push_telemetry() never sends kl_div, so the server
    # cannot evaluate the drift boundary in ACP mode at all - periodic_secondary_
    # checks() skips it with `if qa not in latest: continue`. The only kl_div on
    # disk is whatever analyse_drift() last wrote in harmone_local mode, so
    # serving it unqualified made the dashboard read "Drift detected" forever
    # off a frozen 0.6678. Mark it, so the UI can say "not monitored" instead of
    # presenting a fossil as the current state.
    live_drift = _live_decision("drift_check")
    drift_check = live_drift or info.get("last_drift_check")
    if drift_check and not live_drift:
        drift_check = dict(drift_check, stale=True)
    # Telemetry does not carry the energy budget, so fill it from the goals file
    # rather than rendering "0.722 / -" in the decision panel.
    if score_check and score_check.get("energy_threshold") is None:
        score_check = dict(score_check, energy_threshold=thresholds.get("max_energy"))
    if score_check and not _live_decision("score_check"):
        score_check = dict(score_check, stale=True)

    return jsonify({
        "system": system,
        "active_model": _active_model(system),
        "thresholds": thresholds,
        "event_counters": info.get("event_counters", {}),
        "ema_scores": info.get("ema_scores", {}),
        "last_score_check": score_check,
        "last_drift_check": drift_check,
        "decision_at": LAST_ACP_DECISION["at"],
        "decision_source": "acp" if LAST_ACP_DECISION["at"] else "local",
        "current_energy_threshold": info.get("current_energy_threshold"),
        "recovery_cycles": info.get("recovery_cycles"),
        "vmr_best_match": {"path": best_version, "kl_div": drift_kl.get("min_kl_div")} if best_version else None,
    })


@app.route('/api/models/<system>', methods=['GET'])
def list_models(system):
    """Every model this system knows about, and how each is doing.

    The list is built from the MLflow Model Registry, unioned with the models
    the loop can actually select (mape_info.json's ema_scores) and the local
    catalogue. Registry alone is not enough - CV has no registered models yet
    - and ema_scores alone would hide registry entries that exist but are
    never chosen. Each row says which of those it came from, so a registered
    but unselectable entry is visible rather than silently dropped.
    """
    if system not in ("regression", "cv"):
        return jsonify({"error": "system must be 'regression' or 'cv'"}), 400

    try:
        info = _read_json_file(_knowledge_path(system, "mape_info.json"))
    except _TransientRead as e:
        return jsonify({"error": "knowledge file busy", "path": str(e)}), 503

    ema = info.get("ema_scores", {})
    registry = _registry_models(system)
    catalogue = {m["name"]: m for m in MODEL_CATALOGUE[system]}
    baseline = RUN_BASELINE.get(system)
    # No run started in this server session means there are no rates to report
    # for "this run" - showing the accumulated history instead is what made the
    # page look populated before anything had been started.
    if baseline is None:
        stats, sampled = {}, 0
    else:
        stats, sampled = _per_model_stats(system, since_row=baseline)
    active = _active_model(system)

    # Catalogue order first so the familiar models keep a stable, labelled
    # position; anything the registry or the loop knows about follows.
    names = [m["name"] for m in MODEL_CATALOGUE[system]]
    for extra in list(registry) + list(ema):
        if extra not in names:
            names.append(extra)

    models = []
    for name in names:
        spec = catalogue.get(name, {})
        reg = registry.get(name)
        st = stats.get(name, {})
        path = _model_file(system, name, spec)

        models.append({
            "name": name,
            "label": spec.get("label") or name,
            "blurb": spec.get("blurb") or "",
            "is_active": name == active,
            # Provenance, so the page can explain itself.
            "registered": bool(reg),
            "registered_model_name": reg["registered_model_name"] if reg else None,
            "registry_versions": reg["registry_versions"] if reg else 0,
            "active_alias_version": reg["active_alias_version"] if reg else None,
            "selectable": name in ema,          # what plan.py can switch to
            "available": path is not None,       # weights present on disk
            "size_bytes": os.path.getsize(path) if path else None,
            "version": info.get(f"{name}_version"),
            "ema_score": ema.get(name),
            "recent_inferences": st.get("count", 0),
            "share": st.get("share", 0),
            "mean_inference_ms": st.get("mean_inference_ms"),
            "mean_energy_uj": st.get("mean_energy_uj"),
            "disk_versions": len(_disk_versions(system, name)),
        })

    return jsonify({"system": system, "active_model": active,
                    "sampled_inferences": sampled,
                    "run_active": baseline is not None,
                    "registry_count": len(registry),
                    "models": models})


@app.route('/api/versions/<system>/<model>', methods=['GET'])
def list_versions(system, model):
    """
    Version history from BOTH sources, deliberately not merged: the MLflow
    registry is the product-facing record, versionedMR/ is what the loop
    actually reuses at runtime. They can legitimately disagree.
    """
    if system not in ("regression", "cv"):
        return jsonify({"error": "system must be 'regression' or 'cv'"}), 400
    if model not in [m["name"] for m in MODEL_CATALOGUE[system]]:
        return jsonify({"error": f"unknown model '{model}' for system '{system}'"}), 400

    registered_name = f"harmone-{system}-{model}"
    registry = []
    active_alias_version = None

    try:
        client = MlflowClient()
        try:
            rm = client.get_registered_model(registered_name)
            active_alias_version = (rm.aliases or {}).get("active")
        except Exception:
            rm = None

        if rm is not None:
            for v in client.search_model_versions(f"name='{registered_name}'"):
                registry.append({
                    "version": int(v.version),
                    "run_id": v.run_id,
                    "created_at": v.creation_timestamp / 1000 if v.creation_timestamp else None,
                    "is_active_alias": str(v.version) == str(active_alias_version),
                })
            registry.sort(key=lambda v: v["version"], reverse=True)
    except Exception as e:
        logging.warning(f"[KB] Registry lookup failed for {registered_name}: {e}")

    return jsonify({
        "system": system,
        "model": model,
        "registered_model_name": registered_name,
        "mlflow_ui_url": MLFLOW_UI_URL,
        "registry_versions": registry,
        "disk_versions": _disk_versions(system, model),
    })


# --- NEW: Reset Endpoint to flush data on approach switch ---
@app.route('/api/reset', methods=['POST'])
def reset_knowledge():
    """Clears the in-memory knowledge base to start fresh."""
    global KNOWLEDGE_BASE
    _reset_last_decision()
    KNOWLEDGE_BASE = {
        "policies": {},
        "telemetry_data": {},
        "intervention_logs": {}
    }
    logging.info("[KNOWLEDGE] Knowledge base RESET requested by client.")
    return jsonify({"message": "Knowledge base reset."}), 200


@app.route('/')
def home():
    return "Welcome to ACP Server!", 200


@app.route('/favicon.ico')
def favicon():
    return "", 204

@app.route("/api/start-managed-system", methods=["POST"])
def start_managed_system():
    try:
        # sys.executable, not "python3": on Windows that name resolves to the
        # Microsoft Store alias stub, which exits 0 without running anything -
        # Popen then "succeeds", this endpoint reports ok, and the dashboard
        # waits forever for telemetry that no process is producing. Using the
        # running interpreter also keeps the managed system in this venv.
        _reset_last_decision()
        for sysname in RUN_BASELINE:
            RUN_BASELINE[sysname] = _prediction_row_count(sysname)
        subprocess.Popen([sys.executable, "run_managed_system.py"])
        return jsonify({"status": "ok", "message": "Managed system started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/stop-managed-system", methods=["POST"])
def stop_managed_system():
    """Stop all managed system processes before switching approaches."""
    try:
        # Send shutdown signal to the managed system wrapper. Best-effort: if
        # the wrapper already died or is unresponsive, this raises - but the
        # psutil sweep below is what actually guarantees a clean stop, so it
        # must still run rather than being skipped via this exception.
        try:
            requests.post("http://localhost:8080/adaptor/shutdown", timeout=10)
        except requests.exceptions.RequestException as e:
            logging.warning(f"[CLEANUP] Graceful shutdown request failed ({e}); falling back to process sweep")

        # Also kill any remaining processes using system commands
        current_pid = os.getpid()
        
        for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
            try:
                if proc.info['pid'] == current_pid:
                    continue
                    
                if proc.info['name'] in ['python', 'python3'] and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if ('run_managed_system.py' in cmdline or 
                        'inference.py' in cmdline or 
                        'manage.py' in cmdline or
                        'managed_system_cv' in cmdline or 
                        'managed_system_regression' in cmdline):
                        logging.info(f"Terminating process PID {proc.info['pid']}: {cmdline}")
                        proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        logging.info("[CLEANUP] Managed system processes terminated")
        return jsonify({"status": "ok", "message": "Managed system stopped"})
    except Exception as e:
        logging.error(f"[CLEANUP] Error stopping managed system: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/upload-custom-mape', methods=['POST'])
def upload_custom_mape():
    try:
        # 1. Get Base System
        base_system = request.form.get('base_system')
        if not base_system:
            return jsonify({"error": "Base system not specified"}), 400

        # 2. Define Paths
        if base_system == 'regression':
            source_dir = "managed_system_regression"
            approach_conf_content = "custom_regression" 
        elif base_system == 'cv':
            source_dir = "managed_system_cv"
            approach_conf_content = "custom_cv"
        else:
            return jsonify({"error": "Invalid base system"}), 400

        target_dir = "managed_system_custom"

        # 3. Clean and Re-create Custom Directory
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        shutil.copytree(source_dir, target_dir)
        logging.info(f"[CUSTOM] Copied base system '{source_dir}' to '{target_dir}'")

        # 4. Overwrite with Uploaded MAPE Files
        if 'files[]' not in request.files:
            return jsonify({"error": "No MAPE files uploaded"}), 400

        uploaded_files = request.files.getlist('files[]')
        allowed_files = ['monitor.py', 'analyse.py', 'plan.py', 'execute.py', 'manage.py']
        mape_logic_path = os.path.join(target_dir, "mape_logic")
        os.makedirs(mape_logic_path, exist_ok=True)

        count = 0
        for file in uploaded_files:
            if file.filename in allowed_files:
                save_path = os.path.join(mape_logic_path, file.filename)
                file.save(save_path)
                logging.info(f"[CUSTOM] Overwrote {file.filename}")
                count += 1

        # 5. Update approach.conf
        with open("approach.conf", "w") as f:
            f.write(approach_conf_content)

        # 6. Reset Knowledge Base
        global KNOWLEDGE_BASE
        KNOWLEDGE_BASE = {
            "policies": {},
            "telemetry_data": {},
            "intervention_logs": {}
        }

        # --- NEW LOGIC STARTS HERE ---
        # 7. Handle Dataset Upload
        dataset_file = request.files.get('dataset')
        
        if dataset_file:
            logging.info(f"[CUSTOM] Processing dataset upload for {base_system}...")
            
            if base_system == 'regression':
                # Path: managed_system_custom/knowledge/dataset.csv
                knowledge_dir = os.path.join(target_dir, "knowledge")
                os.makedirs(knowledge_dir, exist_ok=True)
                dataset_path = os.path.join(knowledge_dir, "dataset.csv")
                
                dataset_file.save(dataset_path)
                logging.info(f"[CUSTOM] Regression dataset saved to {dataset_path}")

            elif base_system == 'cv':
                # Path: managed_system_custom/data/bdd100k/images/test/
                # We expect a ZIP file for CV
                test_images_dir = os.path.join(target_dir, "data", "bdd100k", "images", "test")
                
                # Clear existing test images so we only run on uploaded ones
                if os.path.exists(test_images_dir):
                    shutil.rmtree(test_images_dir)
                os.makedirs(test_images_dir, exist_ok=True)

                # Save zip temporarily
                zip_path = os.path.join(target_dir, "temp_dataset.zip")
                dataset_file.save(zip_path)

                # Extract
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(test_images_dir)
                    logging.info(f"[CUSTOM] CV dataset extracted to {test_images_dir}")
                except zipfile.BadZipFile:
                    return jsonify({"error": "Uploaded CV dataset is not a valid zip file"}), 400
                finally:
                    # Clean up zip
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
        else:
            logging.info("[CUSTOM] No dataset uploaded, using default base system data.")
        # --- NEW LOGIC ENDS HERE ---

        return jsonify({"message": f"Custom system built with {count} MAPE files.", "approach": approach_conf_content}), 200

    except Exception as e:
        logging.error(f"[CUSTOM] Error building system: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# -------------------------- MAIN -----------------------------
# ============================================================

if __name__ == "__main__":
    
    # Start periodic drift monitor
    threading.Thread(
        target=periodic_secondary_checks,
        args=(30,),
        daemon=True
    ).start()

    app.run(host="0.0.0.0", port=5000)