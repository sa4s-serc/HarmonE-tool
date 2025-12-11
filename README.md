# Harmonica


The *Harmonica* tool provides a fully runnable implementation designed to enable the sustainable operation of MLOps pipelines. It operationalizes the HarmonE approach and demonstrates how a structured MAPE-K loop can be integrated into real MLOps pipelines to support sustainability, long-term viability, and runtime resilience in Machine-Learning-enabled Systems (MLS).

Modern MLS frequently operate under environmental uncertainty - data drift, workload fluctuations, hardware variability, and changing performance expectations. Traditional MLOps practices streamline model development and deployment, but they offer limited support for detecting and responding to runtime deviations that affect system stability and cost. This implementation bridges that gap by introducing a managing system that oversees the MLS at runtime and enforces adaptation policies that are decoupled from system execution.

## Key Elements

**Managing System Server:** A Flask server that receives telemetry, analyzes it against a policy, and triggers adaptation actions.

**Managed System:** A simulator for a user's MLOps pipeline. It generates metric data, sends it to the managing server, and hosts an endpoint to receive and execute adaptation commands. Its behavior is dynamically defined by `policy.json`.

**Dashboard:** A web-based interface for visualizing runtime metrics, configuring policies, and monitoring adaptation events.

# Getting Started

## Prerequisites
- **OS:** Linux-based Operating System (Ubuntu/Debian recommended).
- **Python:** Python 3.8+ is required.
- **Privileges:** Sudo access is required.
  - Side Effect: The tool uses `pyRAPL` for energy monitoring, which requires read access to Intel RAPL energy counters in /`sys/class/powercap`. You will be prompted for your sudo password during setup to grant these permissions.
- **Ports:** The tool uses ports 5000 (Managing Server) and 8080 (Adaptation Handler). Ensure these are free.
- **GitHub:** For cloning/downloading the tool.

<!-- **Note on Python Commands:**
Depending on your Linux distribution, you may need to use different Python commands:
- **Ubuntu/Debian**: `python3`, `pip3` 
- **Some distributions**: `python3.11`, `python3.12` (for specific versions)
- **Arch Linux**: `python`, `pip`

Replace `python3` with the appropriate command for your system throughout these instructions. -->

## Step 1: Download and Setup

We provide a unified script to handle virtual environment creation, dependency installation, and permission setting.

```bash
# 1. Navigate to the project directory
cd /home/user/<path>/Harmonica/tool

# 2. Make the setup script executable
chmod +x harmone_start.sh

# 3. Run the launch script
# This script creates a virtual environment, installs requirements, and launches the tool.
./harmone_start.sh

```
### Note on Permissions: 
During execution, the script will request your `sudo` password. This is strictly to execute `chmod` on the RAPL energy files to allow the application to read energy metrics without running the entire Python application as root.

```bash
# On the same terminal you will be asked for sudo access password
[sudo] password for <username>:
```


## Step 2: Access the Dashboard
Once the script is running, open your web browser and navigate to:
 http://localhost:8000/dashboard.html

## Step 3: "Play" the Artifact (Running an Experiment)

To verify the system is working and observe the HarmonE loop in action:

1. **Select Approach:** On the dashboard landing page, example: under "Regression", click "HarmonE (Score/Drift)". These are the pre-existing `managed_systems`
  Select the specific variant:
   - **HarmonE (Score/Drift)**: Full adaptive system with intelligent switching
   - **Simple Switch**: Baseline system with basic threshold switching  
   - **Single Model**: Monitor-only mode with no adaptation
 - *Note: You can choose to build your own custom system - process explained in Customization & Reuse*

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

# Customization & Reuse
The Harmonica tool is designed to be extensible. Researchers can reuse the Managing System while swapping out the Managed System (ML Pipeline) to test different self-adaptation strategies in new contexts.
## Building a Custom Managed System

The Dashboard allows you to upload your own **MAPE (Monitor-Analyze-Plan-Execute)** logic and **Datasets** to run on top of the provided inference engines.

### 1. Prerequisites
You need the following files on your local machine:
*   **MAPE Python Files:** `monitor.py`, `analyse.py`, `plan.py`, `execute.py`, and `manage.py`.
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

### Part 2: Template / Test MAPE Files
These are the template MAPE files that you can refer to for building your own system.

#### 📄 `monitor.py`
*Reads the inference output and calculates a mock score.*

```python
import pandas as pd
import os
import json

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")
PREDICTIONS_FILE = os.path.join(KNOWLEDGE_DIR, "predictions.csv")
MODEL_FILE = os.path.join(KNOWLEDGE_DIR, "model.csv")

def get_current_model():
    try:
        with open(MODEL_FILE, "r") as f:
            return f.read().strip()
    except:
        return "unknown"

def monitor_mape():
    """
    Reads the last few predictions to generate telemetry.
    """
    current_model = get_current_model()
    
    # Default values if no data exists yet
    metrics = {
        "score": 0.8,
        "normalized_energy": 0.3,
        "model_used": current_model,
        "r2_score": 0.9,
        "confidence": 0.85
    }

    if os.path.exists(PREDICTIONS_FILE):
        try:
            # Read last 10 rows to calculate current performance
            df = pd.read_csv(PREDICTIONS_FILE)
            if not df.empty:
                df = df.tail(10)
                # Simple mock logic: if model is 'svm' or 'yolo_n', score is lower
                if "svm" in current_model or "yolo_n" in current_model:
                    metrics["score"] = 0.65 # trigger adaptation
                else:
                    metrics["score"] = 0.95
        except Exception as e:
            print(f"[CustomMonitor] Error reading predictions: {e}")

    print(f"[CustomMonitor] Reporting: {metrics}")
    return metrics

def monitor_drift():
    # Placeholder for drift
    return {"kl_div": 0.05}
```

#### 📄 `analyse.py`
*Decides if the score is low enough to warrant a switch.*

```python
from monitor import monitor_mape

def analyse_mape():
    data = monitor_mape()
    if not data:
        return None

    # Simple logic: If score is below 0.7, request a switch
    if data["score"] < 0.7:
        print("[CustomAnalyse] Score is low! Switch needed.")
        return {"switch_needed": True}
    
    print("[CustomAnalyse] System healthy.")
    return {"switch_needed": False}

def analyse_drift():
    return {"drift_detected": False}
```

#### 📄 `plan.py`
*Randomly selects a model to switch to.*

```python
import random
import os

# Identify if we are running Regression or CV based on available models
# This is a hack for the template; in production, you know your system.
def plan_mape(trigger="local"):
    # Randomly pick a model for demonstration purposes
    # Models for Regression: lstm, svm, linear
    # Models for CV: yolo_n, yolo_s, yolo_m
    
    # We will pick from a combined list, but in a real scenario, know your domain.
    # If the user selected Regression, choosing 'yolo' won't break inference (it defaults to LSTM),
    # but let's try to be generic.
    
    options = ["lstm", "svm", "linear"] 
    # Uncomment below line if testing CV
    # options = ["yolo_n", "yolo_s", "yolo_m"]
    
    choice = random.choice(options)
    print(f"[CustomPlan] Planner selected: {choice}")
    return choice

def plan_drift(trigger="local"):
    return None
```

#### 📄 `execute.py`
*Writes the selected model to the shared file.*

```python
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "..", "knowledge")
MODEL_FILE = os.path.join(KNOWLEDGE_DIR, "model.csv")

def execute_mape(trigger="local"):
    from plan import plan_mape
    
    # 1. Plan
    decision = plan_mape(trigger)
    if not decision:
        return

    # 2. Execute (Write to file)
    print(f"[CustomExecute] Writing '{decision}' to model.csv")
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    with open(MODEL_FILE, "w") as f:
        f.write(decision)

def execute_drift(trigger="local"):
    pass
```

#### 📄 `manage.py`
*Runs the loop.*

```python
import time
import sys
import logging
from execute import execute_mape

logging.basicConfig(level=logging.INFO, format='[CustomManage] %(message)s')

def run_mape_loop():
    logging.info("Starting Custom MAPE Loop...")
    while True:
        # Run every 5 seconds for testing
        time.sleep(5)
        logging.info("--- Triggering MAPE Cycle ---")
        execute_mape(trigger="local")

if __name__ == "__main__":
    # The wrapper calls this file. 
    # We can perform a simple loop or listen to commands.
    run_mape_loop()
```

---

### Part 3: Test Dataset (Regression)

Save this as **`dataset.csv`**. This file is valid for the `inference.py` regression engine.

```csv
flow
173.0
169.0
160.0
187.0
195.5
205.0
210.2
180.5
150.0
140.0
135.5
130.0
125.0
140.0
155.0
165.0
175.0
185.0
190.0
195.0
```

### How to Test
1.  Save the 5 Python files above.
2.  Save the `dataset.csv`.
3.  Go to Dashboard -> **Build Custom System**.
4.  Select **Regression**.
5.  Upload the 5 `.py` files in the file uploader.
6.  Upload `dataset.csv` in the dataset uploader.
7.  Click **Build** -> **Start**.

You should see the "Model Distribution" chart on the dashboard change periodically as the `manage.py` loop triggers `plan.py`, which randomly picks a model.

## Monitor the System on website - Navigate to `Live Dashboard` tab and click `Run Managed System` 

1. **Dashboard**: Watch real-time metrics and charts in the web dashboard
2. **Inference results for download**: Click on `Download Telemetry` to obtain the CSV file of predictions.
3. **Console Output**: Monitor both terminal windows for system logs
4. **Knowledge Folder**: Check the `knowledge/` folders for saved data:
   - `predictions.csv` - System predictions  

## Troubleshooting

### Common Issues:

1. **PyRAPL Permission Errors**:
   ```bash
   sudo chmod -R 777 /sys/class/powercap/intel-rapl/
   ```

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
   python3 -m venv harmone_env
   source harmone_env/bin/activate
   # Re-install dependencies
   ```

## System Architecture

- **app.py**: Main application that starts both managing server and Manager
- **Managing Server** (port 5000): Stores policies and telemetry data
- **Manager/Adaptor** (port 8080): Executes adaptation tactics
- **Managed System**: The system being monitored and adapted
- **Web Dashboard**: Real-time monitoring and policy management interface


