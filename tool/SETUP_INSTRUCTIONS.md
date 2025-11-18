# HarmonE Tool - Setup and Running Instructions

## Prerequisites
- Python 3.8+ installed
- Linux system (for pyRAPL energy monitoring)
- Sudo access (for pyRAPL permissions)

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

cd frontend
python3 -m http.server 8000
# Then open: http://localhost:8000/acp_dashboard.html
```

## Step 5: Configure Policies

### 5.1 Select Approach in Dashboard
1. Open the web dashboard
2. Choose your approach (Regression or Computer Vision)
3. Select the specific variant:
   - **HarmonE (Score/Drift)**: Full adaptive system with intelligent switching
   - **Simple Switch**: Baseline system with basic threshold switching  
   - **Single Model**: Monitor-only mode with no adaptation

### 5.2 Configure Policy Settings
1. Click on "Policy Management" tab
2. Load a preset or create custom policy
3. Download the policy JSON file
4. Place the policy file in the `policies/` folder

Example policy files already exist:
- `policies/reg_harmone_score.json` - Regression HarmonE policy
- `policies/cv_harmone_score.json` - Computer Vision HarmonE policy
- `policies/reg_switch_r2.json` - Simple R² switch policy
- etc.

## Step 6: Configure Approach

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
```

## Step 7: Start the Managed System

In a new terminal (keep the main app.py running):

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

## Step 8: Monitor the System

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
python3 app.py &

# 3. Open dashboard (in browser)


# 4. Configure approach.conf and policies/ folder

# 5. Start managed system
python3 run_managed_system.py
```