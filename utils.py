# utils.py
import joblib
import os
import pandas as pd

MODEL_DIR = "models"  # ensure correct path

def load_models():
    """
    Load latency and cost surrogate models from disk.
    """
    # Updated to match the actual model filenames from train_model.py
    lat_path = os.path.join(MODEL_DIR, "best_latency.joblib")
    cost_path = os.path.join(MODEL_DIR, "best_cost.joblib")
    
    models = {}
    
    if os.path.exists(lat_path):
        try:
            models["latency"] = joblib.load(lat_path)
        except Exception as e:
            print(f"Warning: Could not load latency model: {e}")
    
    if os.path.exists(cost_path):
        try:
            models["cost"] = joblib.load(cost_path)
        except Exception as e:
            print(f"Warning: Could not load cost model: {e}")
    
    return models

def preprocess_input(data):
    """
    Converts incoming JSON-like dict into a Pandas DataFrame
    with correct feature names for the ML models.
    """
    features = {
        "bandwidth_MBps": [float(data.get("bandwidth_MBps", 50))],
        "cpu_utilization_%": [float(data.get("cpu_utilization_%", 50))],
        "ram_utilization_%": [float(data.get("ram_utilization_%", 50))],
        "request_size_MB": [float(data.get("request_size_MB", 100))],
        "distance_km": [float(data.get("distance_km", 100))],
        "server_load": [float(data.get("server_load", 0.5))],
        "cache_hit_ratio": [float(data.get("cache_hit_ratio", 0.2))],
        "storage_tier": [data.get("storage_tier", "SSD")]
    }
    return pd.DataFrame(features)