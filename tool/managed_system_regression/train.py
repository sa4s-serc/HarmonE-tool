import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import mlflow

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from console_utils import force_utf8_console
from session_utils import current_session_id, session_versioned_dir

# Own entry point (`mlflow run ... -e train`) - see retrain.py.
force_utf8_console()

# Ensure base directories exist. VMR is per-session (see tool/session_utils.py),
# so initial training lands in the same session namespace a later retrain will
# add to, and version numbering restarts each run.
KNOWLEDGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "knowledge"))
VERSIONED_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "versionedMR"))
SESSION_ID = current_session_id(KNOWLEDGE_DIR)
base_dir = session_versioned_dir(VERSIONED_ROOT, KNOWLEDGE_DIR)
original_model_dir = "models"
os.makedirs(original_model_dir, exist_ok=True)

# --- MLflow tracking setup (see retrain.py for why this must be an absolute path) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKING_DB_PATH = os.path.join(BASE_DIR, "..", "mlflow.db")
# Only set this ourselves if nothing upstream already configured one via env var
# (see retrain.py for why: overriding it here would orphan an inherited run ID).
if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB_PATH}")
mlflow.set_experiment("harmonica-regression")

def get_next_version(model_name):
    """Finds the next version number for a given model."""
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)  # Ensure model-specific directory exists
    existing_versions = [d for d in os.listdir(model_dir) if d.startswith("version_")]

    if existing_versions:
        existing_versions = sorted([int(v.split("_")[-1]) for v in existing_versions])
        return existing_versions[-1] + 1
    return 1  # Start from version_1 if none exists

def save_model_and_data(model, model_name, train_data_scaled, scaler):
    """Saves the trained model and corresponding training data in both the versioned and original directory."""
    version = get_next_version(model_name)
    version_path = os.path.join(base_dir, model_name, f"version_{version}")
    os.makedirs(version_path, exist_ok=True)

    # Save model in both locations
    if model_name == "lstm":
        model_path = os.path.join(original_model_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), os.path.join(version_path, f"{model_name}.pth"))
    else:
        model_path = os.path.join(original_model_dir, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(version_path, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(model, f)

    # Inverse transform before saving
    train_data_original = scaler.inverse_transform(train_data_scaled["train_data"].values.reshape(-1, 1)).flatten()
    train_df = pd.DataFrame({"train_data": train_data_original})

    train_df.to_csv(os.path.join(version_path, "data.csv"), index=False)

    print(f"{model_name} saved at {version_path} and {model_path}")
    return version

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def log_and_register(model, model_name, X_test, y_test, version, extra_params):
    """Evaluates on the held-out test split, logs params/metrics, saves the model
    to versionedMR/models (unchanged), and registers it in the Model Registry."""
    with mlflow.start_run(run_name=f"train-{model_name}", nested=True):
        mlflow.log_params({"model_name": model_name, "trigger": "initial_train", **extra_params})
        mlflow.log_param("versionedMR_version", version)
        # See retrain.py: registry versioning stays global, this tag is what
        # scopes a registered version back to the session that produced it.
        mlflow.set_tag("harmone_session", SESSION_ID)

        # Versioned Model Repository pairs a model with its training data
        # distribution (see retrain.py for the same rationale).
        version_data_path = os.path.join(base_dir, model_name, f"version_{version}", "data.csv")
        if os.path.exists(version_data_path):
            mlflow.log_artifact(version_data_path, artifact_path="training_data")

        if len(X_test) > 0:
            if model_name == "lstm":
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
                    predictions = model(X_tensor).numpy().flatten()
            else:
                predictions = model.predict(X_test)
            test_r2 = r2_score(y_test, predictions)
            test_mae = mean_absolute_error(y_test, predictions)
            mlflow.log_metrics({"test_r2": test_r2, "test_mae": test_mae})
            print(f"🔹 {model_name.upper()} test R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
        else:
            print(f"⚠️ Not enough test data to evaluate {model_name}.")

        registered_name = f"harmone-regression-{model_name}"
        input_example = X_test[:1]
        # See retrain.py for why this is `name=` and not the deprecated
        # positional `artifact_path=`.
        logged_model_name = f"{model_name}-{SESSION_ID}-v{version}"
        if model_name == "lstm":
            lstm_input_example = torch.tensor(input_example, dtype=torch.float32).unsqueeze(-1)
            # See retrain.py: force the classic pickle format rather than the
            # newer pt2/TensorSpec export path, which this plain nn.Module isn't
            # written to satisfy.
            mlflow.pytorch.log_model(
                model, name=logged_model_name, registered_model_name=registered_name,
                input_example=lstm_input_example, serialization_format="pickle",
            )
        else:
            mlflow.sklearn.log_model(model, name=logged_model_name, registered_model_name=registered_name, input_example=input_example)

def main(args):
    seq_length = int(args.seq_length)

    # Load the dataset
    df = pd.read_csv("data/pems/flow_data_train.csv")
    data = df["flow"].values

    # Normalize data for LSTM
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Split into train/test (80% train, 20% test)
    split_idx = int(len(data) * 0.8)
    train_data, test_data = data_scaled[:split_idx], data_scaled[split_idx:]

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    train_df = pd.DataFrame({"train_data": train_data})

    with mlflow.start_run(run_name="train-all"):
        # See retrain.py: when launched via MLflow Projects these are already
        # auto-logged from the entry-point's CLI args, so only log what's missing
        # (this top-level run is the one that inherits the Project's run ID; the
        # nested per-model runs below are our own and don't need this guard).
        existing_params = mlflow.active_run().data.params
        candidate_params = {
            "ridge_alpha": args.ridge_alpha,
            "svm_c": args.svm_c,
            "svm_tol": args.svm_tol,
            "lstm_epochs": args.lstm_epochs,
            "lstm_lr": args.lstm_lr,
            "seq_length": seq_length,
        }
        new_params = {k: v for k, v in candidate_params.items() if k not in existing_params}
        if new_params:
            mlflow.log_params(new_params)

        # ---------------- LSTM Model ----------------
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

        lstm_model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=args.lstm_lr)
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)

        print("Training LSTM model...")
        for epoch in tqdm(range(int(args.lstm_epochs)), desc="LSTM Training Progress"):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = lstm_model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

        lstm_version = save_model_and_data(lstm_model, "lstm", train_df, scaler)
        log_and_register(lstm_model, "lstm", X_test, y_test, lstm_version, {"lstm_epochs": args.lstm_epochs, "lstm_lr": args.lstm_lr})

        # ---------------- Linear Regression ----------------
        print("Training Linear Regression model...")
        lr_model = Ridge(alpha=args.ridge_alpha)
        lr_model.fit(X_train, y_train)

        linear_version = save_model_and_data(lr_model, "linear", train_df, scaler)
        log_and_register(lr_model, "linear", X_test, y_test, linear_version, {"ridge_alpha": args.ridge_alpha})

        # ---------------- Support Vector Machine (SVM) ----------------
        print("Training SVM model...")
        svm_model = SVR(kernel="linear", C=args.svm_c, tol=args.svm_tol)
        svm_model.fit(X_train, y_train)

        svm_version = save_model_and_data(svm_model, "svm", train_df, scaler)
        log_and_register(svm_model, "svm", X_test, y_test, svm_version, {"svm_c": args.svm_c, "svm_tol": args.svm_tol})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ridge_alpha", type=float, default=256)
    parser.add_argument("--svm_c", type=float, default=0.08)
    parser.add_argument("--svm_tol", type=float, default=0.16)
    parser.add_argument("--lstm_epochs", type=float, default=50)
    parser.add_argument("--lstm_lr", type=float, default=0.001)
    parser.add_argument("--seq_length", type=float, default=5)
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
