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