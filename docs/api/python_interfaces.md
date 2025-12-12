# Python Interface Reference (Custom Systems)

To build a custom Managed System (e.g., for a new domain like NLP), you must implement specific Python functions in the `mape_logic/` directory. The Wrapper script imports these functions dynamically.

## 1. Monitor (`monitor.py`)

Responsible for reading raw logs and calculating standardized metrics.

### `monitor_mape()`
*   **Returns:** `dict` or `None`
*   **Required Keys:**
    *   `score`: (float) The primary performance metric (0.0 to 1.0).
    *   `normalized_energy`: (float) Energy consumption normalized (0.0 to 1.0).
    *   `model_used`: (str) Name of the active model.
*   **Description:** Called periodically by the telemetry thread. Should read the latest lines from `knowledge/predictions.csv`.

### `monitor_drift()`
*   **Returns:** `dict` or `None`
*   **Required Keys:**
    *   `kl_div`: (float) The magnitude of data drift.
*   **Description:** Called to assess distribution shifts.

---

## 2. Analyze (`analyse.py`)

Responsible for determining if a violation has occurred.

### `analyse_mape()`
*   **Returns:** `dict`
*   **Required Keys:**
    *   `switch_needed`: (bool) True if adaptation is required.
    *   `threshold_violated`: (str) "score", "energy", or None.
*   **Description:** Logic to compare monitored data against thresholds.

### `analyse_drift()`
*   **Returns:** `dict`
*   **Required Keys:**
    *   `drift_detected`: (bool)
    *   `action`: (str) "switch_version", "retrain", or "replace".
    *   `best_version`: (str, optional) Path to a better historical model.
*   **Description:** Determines if drift is severe enough to warrant action and searches for reusable history.

---

## 3. Plan (`plan.py`)

Responsible for selecting the specific remediation strategy.

### `plan_mape(trigger="local")`
*   **Arguments:** `trigger` (str) - "local" (timer-based) or "acp" (server-command).
*   **Returns:** `str` (The name of the model to switch to, e.g., "yolo_m").
*   **Description:** Selects the best model based on current trade-offs (Exploitation) or random selection (Exploration).

### `plan_drift(trigger="local")`
*   **Arguments:** `trigger` (str).
*   **Returns:** `dict`
    *   `action`: "switch_version" or "retrain".
    *   `version_path`: (if switching) Path to the model file.
*   **Description:** Decides whether to reuse an old model or train a new one.

---

## 4. Execute (`execute.py`)

Responsible for changing the system state.

### `execute_mape(trigger="local")`
*   **Description:** Calls `plan_mape`, receives the target model name, and writes it to `knowledge/model.csv`.

### `execute_drift(trigger="local")`
*   **Description:** Calls `plan_drift`. If the action is "retrain", it executes the training script (e.g., `os.system("python retrain.py")`). If "switch_version", it copies the file.