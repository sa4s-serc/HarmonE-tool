# *Harmonica*


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

## Step 1: Download and Setup

We provide a unified script to handle virtual environment creation, dependency installation, and permission setting.

```bash
# 1. Navigate to the project directory
cd /home/user/<path>/harmonica/tool

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
 - *Note: You can choose to build your own custom system - process explained in [Customization & Reuse](#customization--reuse)*

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

### Part 2: Template / Test MAPE Files
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


# Troubleshooting

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