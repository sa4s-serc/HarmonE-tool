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