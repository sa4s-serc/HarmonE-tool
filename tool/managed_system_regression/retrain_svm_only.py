#!/usr/bin/env python3
"""
Script to retrain only the SVM model using the existing dataset.
This will update the SVM model in the models/ folder.
"""
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("versionedMR/svm", exist_ok=True)

print("🔄 Retraining SVM model...")

# Load the dataset (same one used by inference.py)
try:
    df = pd.read_csv("knowledge/dataset.csv")
    data = df["flow"].values
    print(f"✅ Dataset loaded: {len(data)} samples")
except FileNotFoundError:
    print("❌ Error: knowledge/dataset.csv not found!")
    print("Please make sure the dataset file exists.")
    exit(1)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Create time series sequences (same as inference.py uses)
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Create training sequences
seq_length = 5
X_train, y_train = create_sequences(data_scaled, seq_length)

# Split into train/validation (80% train, 20% validation)
split_idx = int(len(X_train) * 0.8)
X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]

print(f"📊 Training samples: {len(X_train_split)}")
print(f"📊 Validation samples: {len(X_val)}")

# Train SVM with current hyperparameters from retrain.py
print("🚀 Training SVM model...")
svm_model = SVR(kernel="linear", C=0.05, tol=0.16)
svm_model.fit(X_train_split, y_train_split)

# Evaluate the model
y_pred = svm_model.predict(X_val)
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

print(f"📈 Model Performance:")
print(f"   R² Score: {r2:.4f}")
print(f"   Mean Absolute Error: {mae:.4f}")

# Save the trained model
model_path = "models/svm.pkl"
with open(model_path, "wb") as f:
    pickle.dump(svm_model, f)

print(f"✅ SVM model saved to: {model_path}")

# Create a versioned copy
def get_next_version():
    """Find the next version number for SVM."""
    version_dir = "versionedMR/svm"
    existing_versions = [d for d in os.listdir(version_dir) if d.startswith("version_")]
    
    if existing_versions:
        existing_versions = sorted([int(v.split("_")[-1]) for v in existing_versions])
        return existing_versions[-1] + 1
    return 1

version = get_next_version()
version_path = f"versionedMR/svm/version_{version}"
os.makedirs(version_path, exist_ok=True)

# Save versioned model
with open(f"{version_path}/svm.pkl", "wb") as f:
    pickle.dump(svm_model, f)

# Save training data (inverse transformed for version compatibility)
train_data_original = scaler.inverse_transform(data_scaled[:len(X_train_split) + seq_length].reshape(-1, 1)).flatten()
train_df = pd.DataFrame({"train_data": train_data_original})
train_df.to_csv(f"{version_path}/data.csv", index=False)

print(f"✅ Versioned model saved to: {version_path}")
print(f"🎯 SVM retraining completed successfully!")

# Display current hyperparameters
print(f"\n📋 Current SVM Hyperparameters:")
print(f"   Kernel: {svm_model.kernel}")
print(f"   C (Regularization): {svm_model.C}")
print(f"   Tolerance: {svm_model.tol}")