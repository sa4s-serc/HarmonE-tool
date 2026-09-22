# *Harmonica*


The *Harmonica* tool provides a fully runnable implementation designed to enable the sustainable operation of MLOps pipelines. It operationalizes the HarmonE approach and demonstrates how a structured MAPE-K loop can be integrated into real MLOps pipelines to support sustainability, long-term viability, and runtime resilience in Machine-Learning-enabled Systems (MLS).

Modern MLS frequently operate under environmental uncertainty - data drift, workload fluctuations, hardware variability, and changing performance expectations. Traditional MLOps practices streamline model development and deployment, but they offer limited support for detecting and responding to runtime deviations that affect system stability and cost. This implementation bridges that gap by introducing a managing system that oversees the MLS at runtime and enforces adaptation policies that are decoupled from system execution.

## Key Elements

**Managing System Server:** A Flask server that receives telemetry, analyzes it against a policy, and triggers adaptation actions.

**Managed System:** A simulator for a user's MLOps pipeline. It generates metric data, sends it to the managing server, and hosts an endpoint to receive and execute adaptation commands. Its behavior is dynamically defined by `policy.json`.

**Dashboard:** A web-based interface for visualizing runtime metrics, configuring policies, and monitoring adaptation events.

# Getting Started

## Prerequisites
- **OS:** Linux-based Operating System (Ubuntu/Debian recommended) for real RAPL energy measurements — this is the target environment the paper's results were measured on. Windows and macOS are also supported for development/testing (`harmone_start.sh` itself is Linux-only, so use the manual launch steps below on those platforms), but energy readings there are a CPU-utilization *estimate* (see [Running Harmonica + MLflow Together](#running-harmonica--mlflow-together)), not a hardware measurement.
- **Python:** Python 3.8+ is required.
- **Privileges:** Sudo access is required.
  - Side Effect: The tool uses `pyRAPL` for energy monitoring, which requires read access to Intel RAPL energy counters in /`sys/class/powercap`. You will be prompted for your sudo password during setup to grant these permissions.
- **Ports:** The tool uses ports 5000 (Managing Server), 8080 (Adaptation Handler), 8000 (dashboard static files) and 5001 (MLflow UI). Ensure these are free.
- **Internet access (first run only):** `harmone_start.sh` downloads the YOLOv8 base weights (~80MB) for the CV managed system the first time it runs. Every later run reuses them.
- **GitHub:** For cloning/downloading the tool.

## Step 1: Download and Setup

One script handles everything: virtual environment creation, dependency installation, RAPL permissions, seeding the sample datasets and initial models, and launching every server.

```bash
# 1. Navigate to the project directory
cd /home/user/<path>/HarmonE-tool/tool

# 2. Make the setup script executable (only needed once)
chmod +x harmone_start.sh

# 3. Run it
./harmone_start.sh
```

**What this actually does, in order:**

1. Creates `harmone_env/` and installs `requirements.txt` (first run only — a few minutes, PyTorch/Ultralytics are large downloads).
2. Grants read access to the RAPL energy counters (asks for your `sudo` password — see below).
3. Seeds the sample datasets: the regression traffic-flow CSV and 30 CV demo images, copied from `examples/` into each managed system's `knowledge/`/`data/` folders if not already there.
4. Seeds each managed system's MAPE-loop config (`thresholds.json`, `mape_info.json`, `model.csv`) from its tracked `knowledge_seed/` folder if missing.
5. Trains the initial regression models (LSTM/SVM/Linear, ~2 min on CPU) and downloads the YOLOv8 base weights for CV, if they aren't already present.
6. Launches the **ACP/Managing Server** (port 5000), the **MLflow UI** (port 5001), and the **dashboard's static file server** (port 8000) — each in its own terminal window if one is available, or as background processes (logged under `tool/.harmone_logs/`) otherwise.

It's **idempotent** — safe to re-run any time (e.g. after `git pull`). Every step checks whether its output already exists first, so a re-run with everything already in place just relaunches the three servers.

### Note on Permissions: 
During execution, the script will request your `sudo` password — this is **your own login password**, not something specific to the project. It's used strictly to `chmod` the RAPL energy files so the application can read energy metrics without running the entire Python application as root.

```bash
# On the same terminal you will be asked for sudo access password
[sudo] password for <username>:
```

If you decline or don't have sudo, the script continues anyway — energy readings just fall back to a CPU-utilization estimate instead of a hardware reading (see the cross-platform energy note below).

## Step 2: Access the Dashboard
Once the script is running, open your web browser and navigate to:
 http://localhost:8000/dashboard.html

The MLflow UI is also up at http://localhost:5001 (see [MLflow Tracking & Runs](#mlflow-tracking--runs) below).

## Step 3: "Play" the Artifact (Running an Experiment)

To verify the system is working and observe the HarmonE loop in action:

1. **Select Approach:** On the dashboard landing page, example: under "Regression", click "HarmonE (Score/Drift)". These are the pre-existing `managed_systems`
  Select the specific variant:
   - **HarmonE (Score/Drift)**: Full adaptive system with intelligent switching
   - **Simple Switch**: Baseline system with basic threshold switching  
   - **Single Model**: Monitor-only mode with no adaptation
- *Note: You can choose to build your own custom system - process explained in [Customization and Reuse](#customization-and-reuse)
*

1. **Load Policy:** The system will automatically load the preset policy. You can review the thresholds on the "Policy Management" tab.

2. **Start Execution:** Switch to the `Live Dashboard` tab and click the green `Start Managed System` button.

3. **Observe Results:**

    - *Graphs:* Watch the "Main Metric" (e.g., R2 Score) graph. You will see it degrade over time (drift) and then suddenly improve - this indicates an adaptation (model switch) has occurred.

    - *Pie Chart:* The "Model Distribution" chart will update to show which models (e.g., LSTM, SVM) are currently active.

    - *Terminal:* Check your terminal output to see logs of the MAPE loop detecting violations and executing switches.

# Artifact Outputs & Verification
When you run an experiment, the artifact generates several files that represent the execution history and results. These are located in the tool/ directory structure.

1. Telemetry Logs (predictions.csv)
    - Location: tool/managed_system_<type>/knowledge/predictions.csv

    - Description: A raw log of every inference made by the managed system.

    - Content: Timestamps, input data, predicted values, ground truth, and the specific model used for that inference.

    - Verification: You can compare model_used against the timestamp to verify that the model changed exactly when the dashboard reported an adaptation.

2. Adaptation Log (Dashboard Download)
    - Location: Downloadable via the "Download Telemetry (CSV)" button on the Live Dashboard.

    - Description: A consolidated CSV file containing the metrics visualized on the dashboard.

    - Content: Includes the primary metric (e.g., score), energy consumption (normalized_energy), and the active model for every reporting interval.

3. Current State (model.csv)
    - Location: tool/managed_system_<type>/knowledge/model.csv

    - Description: A single-line file containing the name of the currently active model (e.g., lstm).

    - Verification: Open this file during runtime to see the immediate effect of an adaptation action.

# Customization and Reuse
The Harmonica tool is designed to be extensible. Researchers can reuse the Managing System while swapping out the Managed System (ML Pipeline) to test different self-adaptation strategies in new contexts.
## Building a Custom Managed System

The Dashboard allows you to upload your own **MAPE (Monitor-Analyze-Plan-Execute)** logic and **Datasets** to run on top of the provided inference engines.

### 1. Prerequisites
You need the following files on your local machine:
*   **MAPE Python Files:** `monitor.py`, `analyse.py`, `plan.py`, `execute.py`.
*   **Dataset:** A CSV file (for Regression) or a ZIP file (for CV).

### 2. File Requirements & API
Your Python files must interact with the system using specific paths in the `knowledge/` folder.

#### **`monitor.py`**
*   **Input:** Reads from `knowledge/predictions.csv`. This file is automatically populated by the inference engine.
*   **Output:** Must contain a function `monitor_mape()` that returns a dictionary (e.g., `{"score": 0.85, "model_used": "lstm"}`).

#### **`plan.py`**
*   **Output:** Must contain a function `plan_mape(trigger)` that returns a string representing the target model.
*   **Valid Models (Regression):** `lstm`, `svm`, `linear`.
*   **Valid Models (CV):** `yolo_n`, `yolo_s`, `yolo_m`.

#### **`execute.py`**
*   **Action:** Must write the model name (string) to `knowledge/model.csv`. The inference engine reads this file to switch models in real-time.

### 3. Dataset Formats

#### **For Regression (Traffic Flow)**
Upload a file named **`dataset.csv`**. It must have a single column named `flow` containing numerical values.
**Example `dataset.csv`:**
```csv
flow
173.0
169.0
160.0
187.0
195.5
```

#### **For Computer Vision (Object Detection)**
Upload a **`.zip`** file. The system will extract it to the test folder.
**Internal Structure:**
The system expects images to be available for processing. Your zip file should contain `.jpg` or `.png` images. When zipped, the images should be at the root of the zip or in a single folder.

---

### 4. Custom JSON example 

Users can define custom adaptation strategies by specifying (i) the quality attribute to monitor, (ii) boundary conditions that represent sustainability intent, and (iii) tactics to execute when boundaries are violated.

```
{
  "quality_attribute": "<metric>",
  "adaptation_boundary": {
    "condition": "<operator>",
    "threshold": <value>
  },
  "tactics": [{ "tactic_type": "<action>" }]
}
```
---

### Template / Test MAPE Files
The `examples/` directory contains two subfolders:

- `templates/` – This folder provides the skeleton structure for how MAPE files should be organized and written. These are generic templates meant to guide you in building your own system.

- `HarmonE/` – This folder contains the exact MAPE files used by HarmonE. These serve as fully implemented reference examples.

Both sets of files are designed for regression-based MAPE workflows.



### How to Test
1.  Save the 5 Python files above.
2.  Save the `dataset.csv`.
3.  Go to Dashboard -> **Build Custom System**.
4.  Select **Regression**.
5.  Upload the 5 `.py` files in the file uploader.
6.  Upload `dataset.csv` in the dataset uploader.
7.  Click **Build** -> **Start**.


# MLflow Tracking & Runs

Every retrain (LSTM/SVM/Ridge for regression, YOLO fine-tuning for CV) is packaged as an **MLflow Project** and tracked against a local SQLite-backed MLflow store, so retraining is both reproducible and observable rather than a black-box `os.system` call.

**What gets tracked:** each retrain run logs its hyperparameters, a held-out evaluation metric (`val_r2`/`val_mae` for regression, Ultralytics' own training metrics for CV), and registers the resulting model in the MLflow Model Registry (`harmone-regression-<model>` / `harmone-cv-<model>`) — in addition to (not instead of) the existing `versionedMR/` file versioning.

**Where it lives:** `tool/mlflow.db` (tracking + registry metadata) and `tool/mlruns` (logged artifacts/models), shared by both managed systems under separate experiments (`harmone-regression`, `harmone-cv`).

**Viewing runs:** `harmone_start.sh` launches an MLflow UI at http://localhost:5001 alongside the dashboard. You can also start it manually:
```bash
cd tool
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

**Triggering retraining:**
- *Automatically* — this is the intended path: the MAPE-K loop triggers a retrain on drift, running through `mlflow.projects.run(...)` instead of a bare `os.system()` call, so it shows up in the MLflow UI like any other run. There is deliberately no "Retrain Now" button on the dashboard — retraining is something the adaptation loop decides to do, not something you trigger by hand.
- *Manually, from the CLI* (for debugging a specific model in isolation) — run the same MLflow Project directly, with any hyperparameter overridden:
  ```bash
  cd tool
  mlflow run managed_system_regression -e retrain --env-manager local -P lstm_epochs=100
  mlflow run managed_system_regression -e train   --env-manager local -P ridge_alpha=100
  mlflow run managed_system_cv         -e retrain --env-manager local -P epochs=10
  ```
  `--env-manager local` is required since these projects intentionally have no `conda.yaml`/`python_env.yaml` — they reuse whatever environment (e.g. `harmone_env`) you already have active.

## Running Harmonica + MLflow Together

Harmonica (the ACP/Managing Server + dashboard + managed system) and MLflow aren't two separate tools you point at each other — they share the same store (`tool/mlflow.db` + `tool/mlruns`), so running them together just means having both processes up at once against that store.

**Quickest path (Linux):** `./harmone_start.sh` (Step 1 above) launches all of it for you, each in its own terminal: the ACP server (port 5000), the MLflow UI (port 5001), and the dashboard's static file server (port 8000). It also seeds the sample datasets/config and trains or downloads the initial models first, so the dashboard's own **Start Managed System** button (which launches `run_managed_system.py` for you) works immediately — no separate console step needed.

**Manual path** (useful on Windows/macOS dev machines, or when you want each log stream visible separately) — four terminals, from `tool/`:
```bash
# Terminal 1 - ACP / Managing Server
python app.py

# Terminal 2 - MLflow tracking UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# Terminal 3 - Dashboard static files
python -m http.server 8000 --directory frontend

# Terminal 4 - start the managed system once you've picked an approach on the
# dashboard (or just click "Start Managed System" there, which does this for you)
python run_managed_system.py
```

**How they connect at runtime:**
- `run_managed_system.py` opens one MLflow run per session in the `harmone-adaptations` experiment (named `session-<timestamp>`) and logs continuous telemetry (score, energy, drift, per-model usage) to it every ~5s, plus a discrete event every switch/retrain/version-reuse — this is the adaptation timeline, live.
- Each retrain is a *separate* run, in a different experiment (`harmone-regression` / `harmone-cv`) — that's where hyperparameters and held-out accuracy for that specific retrain live.
- The dashboard's **View Active Model** / **Open in MLflow** links on the Live Dashboard tab just point into this same MLflow UI (localhost:5001) — there's no separate sync step, they're two views of one store.

**Cross-platform energy note:** on Linux with a readable `/sys/class/powercap/intel-rapl` (see the PyRAPL Permission step below), energy is a real hardware reading via `pyRAPL`. On Windows, macOS, or a Linux box without RAPL access, `tool/energy_utils.py` automatically falls back to a CPU-utilization-based *estimate* instead of crashing — so the whole stack, MLflow included, still runs end-to-end everywhere; only the *source* of the energy numbers differs. A one-time `[energy_utils] Intel RAPL not available...` warning in the managed system's log tells you which mode you're in.

## Verifying Everything Is Working

After starting Harmonica + MLflow together, work through this in order:

1. **Servers are up:**
   ```bash
   curl http://localhost:5000/          # -> "Welcome to ACP Server!"
   curl -I http://localhost:8000/dashboard.html   # -> 200 OK
   curl -I http://localhost:5001/       # -> 200 OK (MLflow UI)
   ```
2. **Start a session:** on the dashboard, pick a preset (e.g. "HarmonE (Score/Drift)"), let it load, switch to **Live Dashboard**, click **Start Managed System**.
3. **Real inference is happening:** watch `tool/managed_system_<type>/knowledge/predictions.csv` grow —
   ```bash
   watch -n 2 wc -l tool/managed_system_<type>/knowledge/predictions.csv
   ```
   (row count should be climbing, not static).
4. **The dashboard is live:** the "Main Metric" chart and "Model Distribution" pie on the Live Dashboard tab should visibly update every few seconds without a manual page refresh.
5. **MLflow is receiving the same data, live:** open http://localhost:5001, go to the `harmone-adaptations` experiment, open the newest `session-<timestamp>` run, turn on **Auto-refresh**, and confirm its `energy`/`score` metric charts are climbing in step with the dashboard — not flat, not a run from a previous session.
6. **Energy is populated, not zero/crashed:** check the managed system's terminal output — either real RAPL values with no errors (Linux), or a one-time `[energy_utils] Intel RAPL not available...` warning followed by non-zero, load-proportional `energy_uJ` values (Windows/macOS/no-RAPL Linux). If `energy_uJ` is a flat 0 the whole run, something's wrong upstream of `energy_utils.py`, not with the fallback itself.
7. **Adaptations are real, not just counted:** when a model switch fires, confirm it shows up in *both* places — a dashed vertical marker on the dashboard's main chart, and, on the corresponding MLflow run's **Overview → Tags** panel, an `event_<step>_details` tag reading `<old_model> -> <new_model>` with an actual model-name change (not `X -> X`, which just means the policy re-checked and nothing changed).
8. **Clean shutdown:** stop the managed system (dashboard reset, or `curl -X POST http://localhost:5000/api/stop-managed-system`), then confirm the MLflow run's status flipped from `Running` to `FINISHED` with a real `end_time`, and that no stray processes are left:
   ```bash
   ps aux | grep -E "run_managed_system.py|inference.py|manage.py" | grep -v grep
   # -> should print nothing
   ```

If steps 1-4 pass but 5 doesn't (dashboard updates, MLflow doesn't), check `MLFLOW_TRACKING_URI` isn't pointed somewhere else in your shell — every script pins it to `sqlite:///tool/mlflow.db` explicitly, but an inherited env var can still confuse a manually-launched `mlflow ui`.

# Troubleshooting

### Common Issues:

1. **PyRAPL Permission Errors**:
   ```bash
   sudo chmod -R a+r /sys/class/powercap/intel-rapl/
   ```

   (read access is all `pyRAPL` needs — `harmone_start.sh` uses this same command)

2. **Port Already in Use**:
   ```bash
   # Kill processes on ports 5000 and 8080
   sudo lsof -ti:5000 | xargs kill -9
   sudo lsof -ti:8080 | xargs kill -9
   ```

3. **Missing Dependencies**:
   ```bash
   pip install ultralytics opencv-python matplotlib seaborn
   ```

4. **Virtual Environment Issues**:
   ```bash
   deactivate
   rm -rf harmone_env
   ./harmone_start.sh   # idempotent - recreates the venv and reinstalls everything
   ```

5. **Model weights or sample data missing/corrupted** (e.g. after manually deleting `models/` or `knowledge/`):
   ```bash
   ./harmone_start.sh   # only re-seeds/re-trains/re-downloads what's actually missing
   ```

# System Architecture

- **app.py**: Main application that starts both managing server and Manager
- **Managing Server** (port 5000): Stores policies and telemetry data
- **Manager/Adaptor** (port 8080): Executes adaptation tactics
- **Managed System**: The system being monitored and adapted
- **Web Dashboard**: Real-time monitoring and policy management interface

# Complete Datasets

For demonstration purposes, we include only small sample datasets for both the regression and computer vision tasks. These samples allow the system to run end-to-end without requiring large storage or long execution times.
However, if you wish to run on the complete datasets, you may obtain them as follows:
### Regression Task - Traffic Flow Prediction (PEMS Dataset)

  - Due to privacy and licensing restrictions, the full PEMS dataset cannot be redistributed directly.
If you have authorized access from the ccccc, you may integrate the complete dataset into HarmonE by following these steps:

  - Download the data: https://pems.dot.ca.gov/

  - Obtain the raw CSV files from the California PEMS website. Note that the dataset may originally be in MATLAB format; you might need to convert it to CSV. The available CSV files should include a column named "Flow (Veh/5 Minutes)" which represents the traffic flow measurements.

### Computer Vision Task - Object Detection (BDD100K Dataset)

  - For large-scale object detection experiments, the full BDD100K dataset (~100k images) can be used, available from: https://bair.berkeley.edu/blog/2018/05/30/bdd/

  - This dataset includes diverse driving scenes and annotations suitable for training and evaluating object detection models.