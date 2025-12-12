import random
import os

# Identify if we are running Regression or CV based on available Models

def plan_mape(trigger="local"):
    # Randomly pick a model for demonstration purposes
    # Models for Regression: lstm, svm, linear
    # Models for CV: yolo_n, yolo_s, yolo_m
    
    
    options = ["lstm", "svm", "linear"] 
    # Uncomment below line if testing CV
    # options = ["yolo_n", "yolo_s", "yolo_m"]
    
    choice = random.choice(options)
    print(f"[CustomPlan] Planner selected: {choice}")
    return choice

def plan_drift(trigger="local"):
    return None