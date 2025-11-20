# Generalized Self-Adaptation Control Plane (ACP)

This project is a complete, runnable implementation of the Self-Adaptation Control Plane described in the design document. It demonstrates the MAPE-K (Monitor, Analyze, Plan, Execute, Knowledge) loop for a simulated MLOps pipeline, decoupling the adaptation policy from the execution mechanism.

The system consists of two main components that run independently:

**acp_server:** The Adaptation Control Plane (Managing System). It's a Flask server that receives telemetry, analyzes it against a policy, and triggers adaptation actions.

**managed_system:** A simulator for a user's MLOps pipeline (Managed System). It generates metric data, sends it to the ACP, and hosts an endpoint to receive and execute adaptation commands. Its behavior is defined by policy.json.

# HarmonE Tool - Setup and Running Instructions

## Prerequisites
- Python 3.8+ installed on Linux
- Sudo access (for pyRAPL energy monitoring permissions)
- Git (optional, for cloning)

**Note on Python Commands:**
Depending on your Linux distribution, you may need to use different Python commands:
- **Ubuntu/Debian**: `python3`, `pip3` 
- **Some distributions**: `python3.11`, `python3.12` (for specific versions)
- **Arch Linux**: `python`, `pip`

Replace `python3` with the appropriate command for your system throughout these instructions.

## Step 1: Create and Activate Virtual Environment

```bash
# Navigate to the project directory
cd /home/user/<path>/HarmonE-tool/tool

# Create virtual environment
python3 -m venv harmone_env

# Activate virtual environment
source harmone_env/bin/activate

# Verify activation (should show the venv path)
which python
```

## Step 2: Install Dependencies

The project has multiple requirements files for different components. Install them all:

```bash
# Install main system dependencies
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

```

## Step 3: Set PyRAPL Permissions (Energy Monitoring)

PyRAPL requires special permissions to access CPU energy counters:

```bash
# Give PyRAPL the necessary permissions
sudo chmod 777 /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
sudo chmod 777 /sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj
sudo chmod 777 /sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:1/energy_uj

# Alternative: Set broader permissions (if the above specific paths don't exist)
sudo chmod -R 777 /sys/class/powercap/intel-rapl/
```

## Step 4: Start the System Components

### 4.1 Start the Main Application (ACP Server + Manager)
```bash
# In Terminal 1
# Make sure you're in the main directory and venv is active
cd /home/user/<path>/HarmonE-tool/tool
source harmone_env/bin/activate

# Start the main application
python3 app.py
```

This will start:
- ACP Server on port 5000
- Manager/Adaptor on port 8080

### 4.2 Open the Web Dashboard
```bash

# In a new Terminal - Terminal 2
cd frontend
python3 -m http.server 8000

```

- click here: http://localhost:8000/acp_dashboard.html

## Step 5: Configure Policies

### 5.1 Select Approach in Dashboard
1. Open the web dashboard
2. Choose your approach (Regression or Computer Vision) : these are our pre-existing `managed_systems`.
3. Select the specific variant:
   - **HarmonE (Score/Drift)**: Full adaptive system with intelligent switching
   - **Simple Switch**: Baseline system with basic threshold switching  
   - **Single Model**: Monitor-only mode with no adaptation
4. You can also choose to build your very own `custom managed_system`.

### Follow 5.2.1 for trying out the pre-existing managed systems or 5.2.2 to build your custom managed system

### 5.2.1 Regression/Computer Vision Approaches: Configure Policy Settings
- If you are trying out the pre-existing Regression or Computer Vision systems: 
  - Based on your selected approach a preset will be loaded. 
  - You can try out the preset as it is, or can play around by changing thresholds, etc and if changed click on the `Save as Policy` button

  

### 5.2.2 Custom Approach: Configure Policy Settings
- If you are using your `custom managed system`: 
  1. Click on "Policy Management" tab
    2. Create custom policy
    3. Click on the `Save as Policy` button
    4. You can also download the policy JSON file
    5. Modify `approach.conf` file according to your custom managed system


<!-- ## Step 6: Configure Approach

Set the approach configuration file to match your selected policy:


```bash
# Edit the approach.conf file
nano approach.conf

# Set one of these values:
# reg_harmone    - Regression HarmonE system
# reg_switch     - Regression simple switch
# reg_single     - Regression single model
# cv_harmone     - Computer Vision HarmonE system  
# cv_switch      - Computer Vision simple switch
# cv_single      - Computer Vision single model
``` -->

## Step 6: Start the Managed System

In a **new** terminal (keep the main app.py running):

```bash
# Navigate to project directory
cd /home/user/<path>/HarmonE-tool/tool

# Activate virtual environment
source harmone_env/bin/activate

# Start the managed system
python3 run_managed_system.py
```

This will start the appropriate managed system based on your `approach.conf` setting:
- Regression systems use `managed_system_regression/`
- Computer Vision systems use `managed_system_cv/`

## Step 7: Monitor the System

1. **Dashboard**: Watch real-time metrics and charts in the web dashboard
2. **Console Output**: Monitor both terminal windows for system logs
3. **Knowledge Folder**: Check the `knowledge/` folders for saved data:
   - `model.csv` - Model performance metrics
   - `predictions.csv` - System predictions  
   - `mape_log.csv` - MAPE-K loop execution logs
   - `event_log.csv` - Adaptation events

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

- **app.py**: Main application that starts both ACP server and Manager
- **ACP Server** (port 5000): Stores policies and telemetry data
- **Manager/Adaptor** (port 8080): Executes adaptation tactics
- **Managed System**: The system being monitored and adapted
- **Web Dashboard**: Real-time monitoring and policy management interface

## Quick Start Commands Summary

```bash
# 1. Setup
cd /home/user/<path>/HarmonE-tool/tool
python3 -m venv harmone_env
source harmone_env/bin/activate
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

sudo chmod -R 777 /sys/class/powercap/intel-rapl/

# 2. Start system
python3 app.py 

# 3. Open dashboard (in browser)1   A

# 4. Configure approach.conf and policies/ folder

# 5. Start managed system
python3 run_managed_system.py
```

## This project provides a framework for running a managed system (inference.py) in one of three modes, controlled by approach.conf.

`single` (Monitor-Only): Uses the ACP tool to collect performance metrics without making changes.

`harmone_local` (Local Adaptation): Runs the original, fully-local HarmonE MAPE loop. The ACP can observe, but does not interfere.

`harmone_acp` (Tool-Driven Adaptation): The ACP acts as the central brain, using a policy.json file to make decisions. The local HarmonE scripts act as the hands, executing the commands sent by the ACP.

## Project Structure


```
.
.
├── acp_server/
│   ├── app.py
│   └── requirements.txt
│
├── managed_system/
│   ├── knowledge/
│   │   ├── dataset.csv
│   │   ├── model.csv
│   │   ├── mape_info.json
│   │   └── thresholds.json  # Used by local mode
│   │
│   ├── models/
│   │   ├── lstm.pth
│   │   ├── linear.pkl
│   │   └── svm.pkl
│   │
│   ├── mape_logic/
│   │   ├── manage.py        # Refactored to be command-driven
│   │   ├── monitor.py       # Used for rich telemetry
│   │   ├── execute.py       # Refactored to execute specific tactics
│   │   ├── analyse.py       # Only used in local mode
│   │   └── plan.py          # Only used in local mode
│   │
│   ├── inference.py         # Your original inference script (unchanged)
│   ├── run_managed_system.py# The master script to control run modes
│   ├── approach.conf        # Config: 'single', 'harmone_local', or 'harmone_acp'
│   └── policy.json          # The single source of truth for ACP-driven adaptation
│
└── README.md
```
## Requirements

- Python 3.7+

- pip for installing packages

### Setup Instructions

- First, clone or download the project files into a local directory. Then, you will need to set up the Python environment for both components.

- Set up the acp_server:
Open a terminal, navigate to the acp_server directory, and install the required packages.

```
cd acp_server
pip install -r requirements.txt
cd ..
```

- Set up the managed_system:
Open a second terminal, navigate to the managed_system directory, and install its required packages.

```
cd managed_system
pip install -r requirements.txt
cd ..
```

### Running the Simulation

- You must have two separate terminal windows open to run both systems simultaneously.

- Step 1: Start the Adaptation Control Plane (ACP)

    In your first terminal, navigate to the acp_server directory and run the Flask application.
```
cd acp_server
python app.py
```

- You should see output indicating that the Flask server is running on port 5000. This server is now listening for incoming telemetry and policy configurations.

- Step 2: Start the Managed System Simulator

    In your second terminal, navigate to the managed_system directory and run the simulator script.
```
cd managed_system
python simulator.py
```

- Upon starting, the simulator will:

    - Read its configuration from policy.json.

    - Immediately post this adaptation policy to the ACP to configure it.

    - Start its own local server on port 8080 to listen for adaptation commands.

    - Begin a monitoring loop, generating and pushing telemetry data to the ACP every 5 seconds.

- Observing the MAPE-K Loop

    - Watch the output in both terminals to see the self-adaptation process in action.

*[Monitor] Phase:* The managed_system terminal will show logs for each telemetry packet it sends (e.g., [MONITOR] Pushing telemetry: {'energy_consumption_mj': ...}). The acp_server terminal will show when it receives this data.

*[Analyze] Phase:* When the simulated energy_consumption_mj from the managed system spikes above the 500mJ static threshold and also violates the dynamic threshold (20% above the historical average), the acp_server will log a [ANALYZE] Static threshold violated and [ANALYZE] Dynamic logic condition MET message.

*[Plan] & [Execute] Phases:*

Immediately after the violation, the acp_server will enter the Plan phase, logging which tactic it has selected ([PLAN] Tactic selected: 'apply_model_quantization').

It will then enter the Execute phase, logging that it is sending a POST request to the managed system's endpoint ([EXECUTE] Sending POST request...).

### Tactic Execution:

The managed_system terminal will show that it has received the command (Received command to execute tactic...) and will simulate changing its internal state from LARGE_FP32 to QUANTIZED_INT8.

### Post-Adaptation:

After the adaptation, the managed_system will start sending new telemetry data reflecting its "quantized" state: lower energy consumption and slightly lower model accuracy.

The acp_server will continue to receive and analyze this new data, which will no longer violate the adaptation boundary, and the system will remain stable in its new, optimized state.

By following this flow, you can observe the complete, autonomous MAPE-K loop as the system identifies a problem, plans a solution, and executes a corrective action, all orchestrated through simple RESTful APIs.

Using the ACP and a Local MAPE Loop

This project provides a flexible framework for running a managed system (inference.py) in one of two modes, controlled by the approach.conf file.

single (Monitor-Only): Uses the ACP tool to collect and observe performance metrics from your inference.py script without making any changes.

harmone (MAPE Mode): Runs a completely local, decoupled self-adaptation loop using the provided MAPE files (manage.py, etc.) to automatically switch models based on performance. The ACP continues to monitor in the background.



# How to Run

You will need two terminals. First, start the ACP server as always.

# create env
- `python3 -m venv venv`
- `source venv/bin/activate`

# In Terminal 1
cd acp_server
python app.py


To Run in Monitor-Only Mode:

Edit the managed_system/approach.conf file and make sure its content is single.

In your second terminal, run the master wrapper script.

# In Terminal 2
cd managed_system
sudo python run_managed_system.py


Result: Your inference.py script will run using only the LSTM model. The wrapper will send its performance data to the ACP. No model switching will occur.

To Run in Self-Adaptation (MAPE) Mode:

Edit the managed_system/approach.conf file and change its content to harmone.

In your second terminal, run the same master wrapper script.

# In Terminal 2
cd managed_system
sudo python run_managed_system.py


Result: The wrapper will start two background processes: inference.py and mape_logic/manage.py.

inference.py will start with the LSTM model.

mape_logic/manage.py will begin monitoring the output in predictions.csv. When it detects a performance violation (based on its own internal logic), it will command a model switch by overwriting knowledge/model.csv.

inference.py will pick up the change and switch to a more efficient model.

Throughout this entire process, the wrapper continues to send all telemetry to the ACP, giving you a complete, non-invasive view of the self-adapting system.