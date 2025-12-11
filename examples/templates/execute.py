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