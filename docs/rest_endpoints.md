# REST API Reference

Harmonica relies on two HTTP servers communicating via JSON.

## 1. Managing System
**Base URL:** `http://localhost:5000`  
**Defined in:** `tool/app.py`


### `POST /api/telemetry`
Ingests runtime metrics from the Managed System.

*   **Request Body:**
    ```json
    {
      "timestamp": 1715601234,
      "score": 0.85,
      "energy": 120.5,
      "model_used": "yolo_n",
      "kl_div": 0.05
    }
    ```
*   **Behavior:** Stores data in the Knowledge Base and triggers the `analyze_telemetry` routine to check for policy violations.

### `POST /api/policy`
Registers a new adaptation policy or updates an existing one.

*   **Request Body:** A valid Policy JSON object (see `tool/policies/` for examples).
*   **Behavior:** Updates the in-memory policy repository.

### `GET /api/knowledge/<policy_id>`
Retrieves the current state of the system for visualization.

*   **Parameters:** `policy_id` (string) - The ID of the policy to inspect.
*   **Response:**
    ```json
    {
      "policy": { ... },
      "telemetry_history": [ ... ],
      "intervention_logs": [ ... ]
    }
    ```

### `POST /api/write-approach`
Configures which managed system configuration to load on startup.

*   **Request Body:** `{"approach": "cv_harmone_score"}`
*   **Behavior:** Writes to `tool/approach.conf` and clears the Knowledge Base.

---

## 2. Managed System (Adaptation Handler)
**Base URL:** `http://localhost:8080`  
**Defined in:** `tool/run_managed_system.py`

This server runs alongside the inference engine to receive commands.

### `POST /adaptor/tactic`
The "actuator" endpoint. It triggers a specific adaptation logic on the client.

*   **Request Body:**
    ```json
    {
      "tactic_id": "execute_mape_plan"
    }
    ```
*   **Behavior:** Writes the `tactic_id` to `knowledge/command.txt`. The local `manage.py` script picks this up and executes the corresponding Python logic.

### `POST /adaptor/shutdown`
Gracefully terminates the managed system.

*   **Behavior:** Kills all subprocesses (inference engine, manager) and shuts down the wrapper.