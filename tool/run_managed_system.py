import time
import requests
import threading
import logging
import subprocess
import os
import json
import importlib.util
import sys
import shutil
import signal
import psutil
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric
from flask import Flask, request, jsonify

# Must run before anything prints. Also covers inference.py / manage.py, which
# inherit this process's environment when spawned below.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from console_utils import force_utf8_console
force_utf8_console()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MasterWrapper] - %(levelname)s - %(message)s')

# --- Configuration ---
ACP_SERVER_URL = "http://localhost:5000"
APPROACH_CONFIG_FILE = "approach.conf"
POLICY_DIR = "policies"

# How often the telemetry thread samples the monitors and pushes to the ACP
# server + MLflow. Lower = finer-grained live charts, at the cost of re-reading
# predictions.csv more often (monitor_mape parses the whole file each tick), so
# going far below ~1s mostly buys resolution the monitors can't actually supply.
TELEMETRY_INTERVAL_SECONDS = float(os.environ.get("HARMONE_TELEMETRY_INTERVAL_SECONDS", "2"))
HANDLER_PORT = 8080

# --- MLflow: this process owns the shared 'adaptation timeline' session run.
# execute.py (running in the separate manage.py process) appends its own
# adaptation events to the SAME run_id (read from a file this process writes),
# via MlflowClient rather than the fluent start_run API, since two independent
# OS processes need to append to one run concurrently.
# ONE experiment per managed system: the session run and the training runs
# that nest under it belong together. The old split ('harmone-adaptations' for
# sessions, 'harmone-<system>' for training) left every session disconnected
# from the models it produced.
def mlflow_experiment_for(system):
    return f"harmonica-{system}"
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlflow.db')}")
mlflow_client = MlflowClient()


def active_experiment_id(name):
    """Experiment id for `name`, created or restored so it is always ACTIVE.

    get_experiment_by_name() returns SOFT-DELETED experiments too, so taking
    its id blindly makes create_run() fail with "must be in the 'active'
    state". That is exactly how session runs silently stopped being created:
    'harmone-adaptations' had been deleted, the id still resolved, create_run
    raised, the surrounding except swallowed it, and the whole adaptation
    timeline vanished with no visible error.
    """
    exp = mlflow_client.get_experiment_by_name(name)
    if exp is None:
        return mlflow_client.create_experiment(name)
    if exp.lifecycle_stage == "deleted":
        mlflow_client.restore_experiment(exp.experiment_id)
        logging.warning("[MLFLOW] Restored soft-deleted experiment '%s'", name)
    return exp.experiment_id
SESSION_RUN_ID = None

# --- Dynamically set logic paths ---
LOGIC_PATH = ""
COMMAND_FILE_PATH = ""
monitor_mape = None
monitor_drift = None

# --- Global process tracking ---
subprocesses = []
should_shutdown = False

def get_python_command():
    """The interpreter to spawn child processes (inference.py, manage.py) with.

    Deliberately sys.executable rather than a PATH lookup for "python3": on
    Windows, `shutil.which("python3")` matches the Microsoft Store's app
    execution alias, a stub that prints "Python was not found" and exits 0
    without running anything - so the children silently never start and the
    dashboard just sits there with no telemetry. sys.executable is the
    interpreter already running this script, so it also keeps children in the
    same virtualenv without depending on PATH ordering."""
    return sys.executable

def build_and_log_timeline_chart(run_id):
    """
    Renders one interactive HTML chart combining the continuous telemetry
    (score/energy/drift, logged by push_telemetry) with the discrete adaptation
    events (switch/vmr/retrain, logged by execute.py in the manage.py process)
    that both live on this same session run, then attaches it back to the run
    as an artifact.

    This exists because MLflow's own 'Model metrics' tab auto-generates one
    disconnected mini-chart per metric - it doesn't overlay them for you. A
    manually-built 'Add chart' view in the UI can do this, but scripting it here
    means every session gets a ready-made timeline without that manual step,
    and it's what actually reproduces the paper's Figure 4 (adaptation events
    marked directly on a cumulative-energy curve) rather than 12 separate tiles.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        import tempfile

        def hist_df(key):
            hist = mlflow_client.get_metric_history(run_id, key)
            if not hist:
                return None
            return pd.DataFrame({
                "time": pd.to_datetime([m.timestamp for m in hist], unit="ms"),
                "value": [m.value for m in hist],
            }).sort_values("time")

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=False,
            specs=[[{}], [{}], [{}], [{"type": "domain"}]],
            row_heights=[0.28, 0.24, 0.24, 0.24],
            subplot_titles=(
                "Adaptation timeline - cumulative MAPE-K energy, with switch/vmr/retrain marked on it",
                "Performance vs. adaptation - continuous score, with the value that triggered each switch",
                "Drift vs. adaptation - continuous KL divergence, with the value that triggered each retrain",
                "Model distribution - share of time each model was active this session",
            ),
            vertical_spacing=0.1,
        )

        energy_df = hist_df("cumulative_mape_k_energy_uJ")
        if energy_df is not None:
            fig.add_trace(go.Scatter(x=energy_df["time"], y=energy_df["value"], mode="lines+markers",
                                      name="cumulative MAPE-K energy (uJ)", line=dict(color="gray")), row=1, col=1)
        for key, label, color in [("event_switch", "switch", "orange"), ("event_vmr", "vmr", "green"), ("event_retrain", "retrain", "red")]:
            df = hist_df(key)
            if df is not None:
                fig.add_trace(go.Scatter(x=df["time"], y=df["value"], mode="markers", name=label,
                                          marker=dict(size=14, color=color, symbol="star")), row=1, col=1)

        score_df = hist_df("score")
        if score_df is not None:
            fig.add_trace(go.Scatter(x=score_df["time"], y=score_df["value"], mode="lines+markers",
                                      name="score (continuous)", line=dict(color="steelblue")), row=2, col=1)
        trig_df = hist_df("triggering_score")
        if trig_df is not None:
            fig.add_trace(go.Scatter(x=trig_df["time"], y=trig_df["value"], mode="markers", name="score @ switch decision",
                                      marker=dict(size=12, color="orange", symbol="x")), row=2, col=1)

        kl_df = hist_df("kl_div")
        if kl_df is not None:
            fig.add_trace(go.Scatter(x=kl_df["time"], y=kl_df["value"], mode="lines+markers",
                                      name="KL divergence (continuous)", line=dict(color="mediumpurple")), row=3, col=1)
        kld_df = hist_df("kl_divergence")
        if kld_df is not None:
            fig.add_trace(go.Scatter(x=kld_df["time"], y=kld_df["value"], mode="markers", name="KL @ retrain decision",
                                      marker=dict(size=12, color="red", symbol="x")), row=3, col=1)

        # Model distribution: push_telemetry() logs a `using_<model>` metric
        # (value 1) each tick a given model is active, one series per model name
        # actually seen - so the set of matching keys, and each one's point count,
        # is discovered from the run itself rather than a hardcoded model list
        # (this file is shared by both the regression and CV managed systems).
        run_metrics = mlflow_client.get_run(run_id).data.metrics
        usage_keys = [k for k in run_metrics if k.startswith("using_")]
        if usage_keys:
            labels = [k[len("using_"):] for k in usage_keys]
            counts = [len(mlflow_client.get_metric_history(run_id, k)) for k in usage_keys]
            fig.add_trace(go.Pie(labels=labels, values=counts, name="model distribution"), row=4, col=1)

        fig.update_layout(height=1400, title_text="HarmonE Adaptation Timeline", template="plotly_dark")

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "adaptation_timeline.html")
            fig.write_html(html_path)
            mlflow_client.log_artifact(run_id, html_path)
        logging.info(f"[MLFLOW] Adaptation timeline chart logged to run {run_id}")
    except Exception as e:
        logging.warning(f"[MLFLOW] Failed to build timeline chart: {e}")

def cleanup_processes():
    """Clean up all subprocesses and related processes."""
    global subprocesses, should_shutdown
    should_shutdown = True

    logging.info("Starting process cleanup...")

    if SESSION_RUN_ID:
        try:
            build_and_log_timeline_chart(SESSION_RUN_ID)
            mlflow_client.set_terminated(SESSION_RUN_ID, "FINISHED")
        except Exception as e:
            logging.warning(f"[MLFLOW] Failed to terminate session run: {e}")
    
    # Terminate direct subprocesses
    for p in subprocesses:
        try:
            if p.poll() is None:  # Process is still running
                logging.info(f"Terminating subprocess PID {p.pid}")
                p.terminate()
                
                # Wait for graceful termination, then force kill if needed
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logging.warning(f"Process PID {p.pid} didn't terminate gracefully, force killing...")
                    p.kill()
        except Exception as e:
            logging.error(f"Error terminating subprocess PID {p.pid}: {e}")
    
    # Clean up any orphaned Python processes related to our systems
    try:
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
            try:
                # Skip our own process
                if proc.info['pid'] == current_pid:
                    continue
                    
                # Look for Python processes that might be our inference/manage processes
                if proc.info['name'] in ['python', 'python3'] and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if ('inference.py' in cmdline or 'manage.py' in cmdline or 
                        'managed_system_cv' in cmdline or 'managed_system_regression' in cmdline):
                        logging.info(f"Killing orphaned process PID {proc.info['pid']}: {cmdline}")
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logging.error(f"Error during orphaned process cleanup: {e}")
    
    subprocesses.clear()
    logging.info("Process cleanup completed")

# --- Adaptation Handler API (Listens for commands from ACP) ---
handler_app = Flask(__name__)

@handler_app.route('/adaptor/tactic', methods=['POST'])
def execute_tactic_from_acp():
    """Receives a command from the ACP and writes it to the correct command file."""
    global should_shutdown
    if should_shutdown:
        return jsonify({"error": "System is shutting down"}), 503
        
    data = request.json
    tactic_id = data.get("tactic_id")
    logging.info(f"[ACP_Handler] Command received: '{tactic_id}'")
    try:
        # COMMAND_FILE_PATH is set dynamically in __main__
        with open(COMMAND_FILE_PATH, "w") as f:
            f.write(tactic_id)
        logging.info(f"[ACP_Handler] Command '{tactic_id}' queued in {COMMAND_FILE_PATH}.")
        return jsonify({"message": "Command queued."}), 200
    except Exception as e:
        logging.error(f"[ACP_Handler] Failed to write command file: {e}")
        return jsonify({"error": "Failed to queue command"}), 500

@handler_app.route('/adaptor/shutdown', methods=['POST'])
def shutdown_system():
    """Endpoint to shutdown all managed system processes."""
    global should_shutdown
    logging.info("[ACP_Handler] Shutdown command received")
    should_shutdown = True
    
    # Run cleanup (including chart-building) SYNCHRONOUSLY, before responding.
    # app.py's /api/stop-managed-system posts here, then - as soon as THIS call
    # returns - separately sweeps and psutil-kills this same process. If cleanup
    # ran in a background thread instead, that external kill could land mid
    # chart-build (a plotly render + artifact upload) and the timeline artifact
    # would silently never get saved, no matter how generous a same-process
    # timer here was. Blocking makes the HTTP response slower, but that's a
    # small price for the chart actually existing afterward.
    cleanup_processes()

    def delayed_exit():
        # Cleanup already fully finished above; this delay is purely to let
        # Werkzeug actually flush the HTTP response bytes before the hard exit -
        # 0.5s measurably wasn't enough margin in practice (caused a client-side
        # ConnectionResetError), so match the original, empirically-fine value.
        time.sleep(2)
        os._exit(0)

    threading.Thread(target=delayed_exit, daemon=True).start()
    
    return jsonify({"message": "System shutdown initiated"}), 200

@handler_app.route('/adaptor/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global should_shutdown
    if should_shutdown:
        return jsonify({"status": "shutting_down"}), 503
    return jsonify({"status": "running", "processes": len(subprocesses)}), 200

def run_handler_api():
    handler_app.run(host='0.0.0.0', port=HANDLER_PORT)

# --- Telemetry & Policy Functions ---
def push_telemetry():
    """Dynamically pushes telemetry from the correct monitor."""
    global monitor_mape, monitor_drift, should_shutdown
    if not monitor_mape and not monitor_drift:
        logging.critical("Monitors not imported. Exiting telemetry thread.")
        return

    logging.info(f"[Monitor] Telemetry thread started.")
    time.sleep(2) # Initial delay
    
    while not should_shutdown:
        try:
            telemetry_payload = {"timestamp": time.time()}
            
            # Dynamically call the correct monitor
            if monitor_mape:
                mape_metrics = monitor_mape()
                if mape_metrics:
                    telemetry_payload.update(mape_metrics)

            if monitor_drift:
                drift_metrics = monitor_drift()
                if drift_metrics:
                    telemetry_payload.update(drift_metrics)

            if len(telemetry_payload) > 1: # More than just timestamp
                logging.info(f"[Monitor] Pushing telemetry: {telemetry_payload}")
                requests.post(f"{ACP_SERVER_URL}/api/telemetry", json=telemetry_payload, timeout=3)

                # Also log the continuous telemetry onto the shared adaptation-
                # timeline run, so it renders on the SAME native MLflow chart as
                # the discrete switch/vmr/retrain events logged by execute.py (in
                # the separate manage.py process) - wall-clock MILLISECONDS as
                # `step` gives both processes a shared, comparable axis. It must
                # stay milliseconds in all three places (here and both
                # mape_logic/execute.py files) or the two series land on
                # different scales; seconds would also collide into one step
                # whenever TELEMETRY_INTERVAL_SECONDS drops below 1.
                #
                # Every numeric field monitor_mape()/monitor_drift() returns gets
                # logged generically (score, r2_score, energy, normalized_energy,
                # kl_div, and the running model_switches/retrains/vmr_events/
                # mape_k_energy_uJ counts) rather than a hand-picked subset, so a
                # new metric added to monitor.py shows up in MLflow automatically.
                # (simple_switches was dropped from monitor_mape()'s return value -
                # it's dead/always-zero in ACP mode, the only wired execution path.)
                if SESSION_RUN_ID:
                    now_ms = int(time.time() * 1000)
                    step = now_ms
                    metrics = []
                    for key, value in telemetry_payload.items():
                        if key == "timestamp":
                            continue
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            metrics.append(Metric(key, float(value), now_ms, step))

                    # Model distribution: a one-hot indicator for whichever model
                    # is active this tick, so MLflow can compute/plot the time
                    # distribution across models - mirrors the dashboard's own
                    # 'Model Distribution' pie chart, but durable and queryable
                    # (the dashboard's version resets when the browser tab closes).
                    model_used = telemetry_payload.get("model_used")
                    if model_used:
                        metrics.append(Metric(f"using_{model_used}", 1.0, now_ms, step))

                    # One batched write rather than one call (and one SQLite
                    # transaction) per metric - at a short TELEMETRY_INTERVAL_SECONDS
                    # the per-call overhead is what would otherwise start to dominate.
                    if metrics:
                        try:
                            mlflow_client.log_batch(SESSION_RUN_ID, metrics=metrics)
                        except Exception as e:
                            logging.warning(f"[MLFLOW] Failed to log telemetry batch: {e}")
            else:
                logging.info("[Monitor] No new data from monitors.")

        except Exception as e:
            if not should_shutdown:
                logging.error(f"[Monitor] Error in telemetry loop: {e}", exc_info=True)
        
        time.sleep(TELEMETRY_INTERVAL_SECONDS)

    logging.info("[Monitor] Telemetry thread shutting down")

def register_policies_with_acp(policy_prefix):
    """
    Registers policies from the POLICY_DIR that match the prefix
    (e.g., 'cv_harmone' prefix will load 'cv_harmone_score.json').
    """
    if 'single' in policy_prefix:
        logging.info("Running in 'single' mode. No policies will be registered.")
        return True

    try:
        policy_files = [
            f for f in os.listdir(POLICY_DIR) 
            if f.endswith('.json') and f.startswith(policy_prefix)
        ]
    except FileNotFoundError:
        logging.error(f"FATAL: Policy directory '{POLICY_DIR}' not found.")
        return False
        
    if not policy_files:
        logging.error(f"FATAL: No policies found in '{POLICY_DIR}' with prefix '{policy_prefix}'. Required for non-single modes.")
        logging.error(f"Expected policy file like: '{policy_prefix}_score.json' or similar in '{POLICY_DIR}' directory.")
        return False # Make this fatal for non-single modes

    logging.info(f"Found {len(policy_files)} policies to register: {policy_files}")
    
    for policy_file in policy_files:
        try:
            policy_path = os.path.join(POLICY_DIR, policy_file)
            with open(policy_path, 'r') as f:
                policy = json.load(f)
            
            policy_id = policy.get("policy_id")
            if not policy_id:
                logging.error(f"'{policy_file}' is missing 'policy_id'. Skipping.")
                continue

            requests.post(f"{ACP_SERVER_URL}/api/policy", json=policy, timeout=3)
            logging.info(f"Policy '{policy_id}' from '{policy_file}' registered.")
        
        except requests.exceptions.RequestException as e:
            logging.critical(f"FATAL: Could not connect to ACP at {ACP_SERVER_URL}.")
            return False
        except Exception as e:
            logging.error(f"FATAL: Error registering policy '{policy_file}': {e}")
            return False
            
    return True

def import_monitor_from_path(logic_path):
    """Helper function to dynamically import the monitor module."""
    global monitor_mape, monitor_drift
    try:
        monitor_path = os.path.join(logic_path, "mape_logic", "monitor.py")
        spec = importlib.util.spec_from_file_location("monitor", monitor_path)
        monitor_module = importlib.util.module_from_spec(spec)
        
        sys.path.insert(0, logic_path) # Add to path so monitor's internal imports work
        spec.loader.exec_module(monitor_module)
        sys.path.pop(0) # Clean up sys.path
        
        # We need to handle if a monitor doesn't exist (e.g. no drift)
        monitor_mape = getattr(monitor_module, "monitor_mape", None)
        monitor_drift = getattr(monitor_module, "monitor_drift", None)
        
        if not monitor_mape:
             logging.warning(f"Could not find 'monitor_mape' in {monitor_path}")
        if not monitor_drift:
             logging.warning(f"Could not find 'monitor_drift' in {monitor_path}")
             
    except Exception as e:
        logging.critical(f"FATAL: Could not import monitors from '{monitor_path}': {e}")
        exit(1)

# --- Main Execution Logic ---
if __name__ == '__main__':
    # 1. Read Master Configuration
    try:
        with open(APPROACH_CONFIG_FILE, 'r') as f:
            approach = f.read().strip().lower()
    except FileNotFoundError:
        logging.critical(f"FATAL: '{APPROACH_CONFIG_FILE}' not found. Please create it.")
        exit(1)
        
    # --- MODIFIED BLOCK STARTS HERE ---
    if approach.startswith("custom_"):
        # Handle Custom Mode
        system_type = "custom" # This helps us distinguish in logs
        run_mode = approach.split('_')[1] # 'regression' or 'cv' effectively
        LOGIC_PATH = "managed_system_custom" # <--- The key change: Point to the new dir
        logging.info(f"--- Running CUSTOM System based on: '{run_mode.upper()}' ---")
    
    # Standard Modes
    elif '_' in approach:
        system_type, run_mode = approach.split('_', 1)
        
        if system_type == "cv":
            LOGIC_PATH = "managed_system_cv"
        elif system_type == "reg":
            LOGIC_PATH = "managed_system_regression"
        else:
            logging.critical(f"FATAL: Unknown system_type '{system_type}'.")
            exit(1)
            
        logging.info(f"--- Running System: '{system_type.upper()}' in Mode: '{run_mode.upper()}' ---")
    else:
        logging.critical(f"FATAL: Invalid approach format '{approach}'.")
        exit(1)
    # --- MODIFIED BLOCK ENDS HERE ---

    # 2. Parse Configuration (e.g., "cv_harmone")
    # if not '_' in approach:
    #     logging.critical(f"FATAL: Invalid approach '{approach}'. Must be format 'system_mode' (e.g., 'cv_harmone').")
    #     exit(1)
        
    # system_type, run_mode = approach.split('_', 1)
    
    # if system_type == "cv":
    #     LOGIC_PATH = "managed_system_cv"
    # elif system_type == "reg":
    #     LOGIC_PATH = "managed_system_regression"
    # else:
    #     logging.critical(f"FATAL: Unknown system_type '{system_type}'. Must be 'cv' or 'reg'.")
    #     exit(1)
    
    # logging.info(f"--- Running System: '{system_type.upper()}' in Mode: '{run_mode.upper()}' ---")
    
    # 3. Dynamically import the correct logic
    import_monitor_from_path(LOGIC_PATH)

    # 4. Set up knowledge path and write the *local* config for manage.py
    KNOWLEDGE_PATH = os.path.join(LOGIC_PATH, "knowledge")
    LOCAL_APPROACH_CONFIG = os.path.join(LOGIC_PATH, "approach.conf")
    COMMAND_FILE_PATH = os.path.join(KNOWLEDGE_PATH, "command.txt") # Set global var

    os.makedirs(KNOWLEDGE_PATH, exist_ok=True)
    with open(LOCAL_APPROACH_CONFIG, "w") as f:
        # Pass the correct mode to the local manage.py
        f.write(f"{run_mode}_acp") # e.g., "harmone_acp", "switch_acp"

    # 4b. Create the shared MLflow session run for this managed-system run, and
    # hand its run_id to the manage.py subprocess (a separate OS process) via a
    # knowledge file, so its adaptation events land on this same timeline.
    if 'single' not in run_mode:
        try:
            system_name = "cv" if LOGIC_PATH.endswith("_cv") else "regression"
            exp_id = active_experiment_id(mlflow_experiment_for(system_name))
            session_run = mlflow_client.create_run(experiment_id=exp_id, run_name=f"session-{int(time.time())}")
            SESSION_RUN_ID = session_run.info.run_id
            mlflow_client.set_tag(SESSION_RUN_ID, "approach", approach)

            # Sustainability Goals Repository -> params + artifact on the
            # session run, so exactly which boundaries were active during this
            # run's adaptations is recorded, not just inferred from the code.
            thresholds_path = os.path.join(KNOWLEDGE_PATH, "thresholds.json")
            if os.path.exists(thresholds_path):
                mlflow_client.log_artifact(SESSION_RUN_ID, thresholds_path)
                try:
                    with open(thresholds_path) as f:
                        thresholds = json.load(f)
                    for k, v in thresholds.items():
                        if isinstance(v, (int, float, str)):
                            mlflow_client.log_param(SESSION_RUN_ID, f"threshold_{k}", v)
                except Exception as e:
                    logging.warning(f"[MLFLOW] Failed to log threshold params: {e}")

            # The active policy JSON (tool/policies/<prefix>_*.json) matching
            # this run's system_type + run_mode, e.g. 'reg_harmone_score.json'.
            try:
                policy_prefix = f"{system_type}_{run_mode}"
                for pf in os.listdir(POLICY_DIR):
                    if pf.startswith(policy_prefix) and pf.endswith(".json"):
                        mlflow_client.log_artifact(SESSION_RUN_ID, os.path.join(POLICY_DIR, pf))
            except Exception as e:
                logging.warning(f"[MLFLOW] Failed to log policy artifact: {e}")

            with open(os.path.join(KNOWLEDGE_PATH, "mlflow_session_run_id.txt"), "w") as f:
                f.write(SESSION_RUN_ID)
            logging.info(f"[MLFLOW] Session run created: {SESSION_RUN_ID}")
        except Exception as e:
            logging.error(
                "[MLFLOW] Could not create the session run (%s). Adaptation "
                "events for this run will NOT be recorded on a timeline.", e,
                exc_info=True,
            )

    # 5. Start Background Threads
    threading.Thread(target=push_telemetry, daemon=True).start()

    # 6. Start Subprocesses from the correct logic path
    try:
        python_cmd = get_python_command()
        inference_cmd = [python_cmd, "-u", "inference.py"]
        manage_cmd = [python_cmd, "-u", "mape_logic/manage.py"]

        # Pass the logic path as a working directory so all file paths are correct
        subprocesses.append(subprocess.Popen(inference_cmd, cwd=LOGIC_PATH))
        logging.info(f"Inference engine '{inference_cmd[2]}' started in '{LOGIC_PATH}'.")
        
        if 'single' not in run_mode:
            subprocesses.append(subprocess.Popen(manage_cmd, cwd=LOGIC_PATH))
            logging.info(f"MAPE logic '{manage_cmd[2]}' started in '{LOGIC_PATH}'.")
        else:
            logging.info("Running in 'single' (monitor-only) mode. 'manage.py' will not be started.")

    except Exception as e:
        logging.critical(f"Failed to start subprocesses: {e}")
        exit(1)

    # 7. Handle ACP-specific setup
    policy_prefix = f"{system_type}_{run_mode}" # e.g., "cv_harmone"
    
    if not register_policies_with_acp(policy_prefix):
        cleanup_processes()
        exit(1)
    
    if 'single' not in run_mode:
        threading.Thread(target=run_handler_api, daemon=True).start()
        logging.info(f"Adaptation Handler API listening on http://0.0.0.0:{HANDLER_PORT}...")

    # 8. Wait for processes to finish
    try:
        logging.info("Wrapper is running. (Press Ctrl+C to stop)")
        for p in subprocesses: p.wait()
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Terminating subprocesses...")
        cleanup_processes()
    finally:
        logging.info("Wrapper script finished.")