# Generalized Self-Adaptation Control Plane (ACP)

This project is a complete, runnable implementation of the Self-Adaptation Control Plane described in the design document. It demonstrates the MAPE-K (Monitor, Analyze, Plan, Execute, Knowledge) loop for a simulated MLOps pipeline, decoupling the adaptation policy from the execution mechanism.

The system consists of two main components that run independently:

**acp_server:** The Adaptation Control Plane (Managing System). It's a Flask server that receives telemetry, analyzes it against a policy, and triggers adaptation actions.

**managed_system:** A simulator for a user's MLOps pipeline (Managed System). It generates metric data, sends it to the ACP, and hosts an endpoint to receive and execute adaptation commands. Its behavior is defined by policy.json.

# HarmonE Tool - Setup and Running Instructions

## Prerequisites
- Linux based OS
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

# Open terminal
# Make the script executable:
chmod +x harmone_start.sh

# Run the combined setup + launch script:
./harmone_start.sh

```


## Step 2: Set PyRAPL Permissions (Energy Monitoring)

PyRAPL requires special permissions to access CPU energy counters
```bash
# On the same terminal you will be asked for sudo access password
[sudo] password for <username>:
```

## Step 3: Open the web dashboard

- click here: http://localhost:8000/acp_dashboard.html

## Step 4: Configure Policies

### 4.1 Select Approach in Dashboard
1. Open the web dashboard
2. Choose your approach (Regression or Computer Vision) : these are our pre-existing `managed_systems`.
3. Select the specific variant:
   - **HarmonE (Score/Drift)**: Full adaptive system with intelligent switching
   - **Simple Switch**: Baseline system with basic threshold switching  
   - **Single Model**: Monitor-only mode with no adaptation
4. You can also choose to build your very own `custom managed_system`.

### Follow 4.2.1 for trying out the pre-existing managed systems or 4.2.2 to build your custom managed system

### 4.2.1 Regression/Computer Vision Approaches: Configure Policy Settings
- If you are trying out the pre-existing Regression or Computer Vision systems: 
  - Based on your selected approach a preset will be loaded. 
  - You can try out the preset as it is, or can play around by changing thresholds, etc and if changed click on the `Save as Policy` button
  - Navigate to `Live Dashboard` tab and click `Start Managed System` 

  

### 4.2.2 Custom Approach: Configure Policy Settings
- If you are using your `custom managed system`: 
  1. Click on "Policy Management" tab
    2. Create custom policy
    3. Click on the `Save as Policy` button
    4. You can also download the policy JSON file
    5. Modify `approach.conf` file according to your custom managed system


## Monitor the System on website - Navigate to `Live Dashboard` tab and click `load` (on the top right)

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

- **app.py**: Main application that starts both ACP server and Manager
- **ACP Server** (port 5000): Stores policies and telemetry data
- **Manager/Adaptor** (port 8080): Executes adaptation tactics
- **Managed System**: The system being monitored and adapted
- **Web Dashboard**: Real-time monitoring and policy management interface


