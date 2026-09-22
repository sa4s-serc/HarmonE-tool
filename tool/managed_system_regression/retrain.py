import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import mlflow

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from console_utils import force_utf8_console
from session_utils import current_session_id, session_versioned_dir

# This is its own entry point (`mlflow run ... -e retrain`, or bare `python
# retrain.py`), so it can't rely on inheriting the setting from app.py /
# run_managed_system.py.
force_utf8_console()

# Ensure directories exist
KNOWLEDGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "knowledge"))
# VMR is scoped per managed-system session, so version numbers restart at 1 each
# run and drift reuse only ever considers models this session trained - see
# tool/session_utils.py.
VERSIONED_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "versionedMR"))
SESSION_ID = current_session_id(KNOWLEDGE_DIR)
base_dir = session_versioned_dir(VERSIONED_ROOT, KNOWLEDGE_DIR)
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

drift_file = "knowledge/drift.csv"
model_file = "knowledge/model.csv"

# --- MLflow tracking setup ---
# Resolve an absolute path to tool/mlflow.db regardless of the caller's cwd
# (bare `python retrain.py`, `mlflow run .`, or triggered from execute.py/app.py).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKING_DB_PATH = os.path.join(BASE_DIR, "..", "mlflow.db")
# Only set this ourselves when nothing upstream (mlflow.projects.run's caller,
# e.g. execute.py/app.py, or `mlflow run`'s own default) already configured one via
# env var — otherwise we'd point at a different store than the run ID we inherit,
# and mlflow.start_run() below would fail to find it.
if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB_PATH}")
mlflow.set_experiment("harmonica-regression")

def get_next_version(model_name):
    """Finds the next version number for a given model."""
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    existing_versions = [d for d in os.listdir(model_dir) if d.startswith("version_")]

    if existing_versions:
        existing_versions = sorted([int(v.split("_")[-1]) for v in existing_versions])
        return existing_versions[-1] + 1
    return 1

def save_model_and_data(model, model_name, train_data):
    """Saves trained model and its data in `models/` and `versionedMR/`."""
    version = get_next_version(model_name)
    version_path = os.path.join(base_dir, model_name, f"version_{version}")
    os.makedirs(version_path, exist_ok=True)

    # Save model
    if model_name == "lstm":
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), os.path.join(version_path, f"{model_name}.pth"))
    else:
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(version_path, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(model, f)

    # Save training data
    train_data.to_csv(os.path.join(version_path, "data.csv"), index=False)
    print(f"✔ {model_name} saved at {version_path} and {model_path}")
    return version

def create_sequences(data, seq_length=5):
    """Creates time series sequences for training."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    """LSTM model architecture"""
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def train_lstm(X_train, y_train, epochs, lr):
    """Trains an LSTM model."""
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=16, shuffle=True)

    for epoch in range(int(epochs)):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    return model

def predict(model, model_name, X):
    """Runs inference for whichever model type was retrained."""
    if model_name == "lstm":
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
            return model(X_tensor).numpy().flatten()
    return model.predict(X)

def retrain(args):
    """Retrains the current model using `drift.csv`."""
    if not os.path.exists(drift_file) or not os.path.exists(model_file):
        print("Missing required files: `drift.csv` or `model.csv`.")
        return

    try:
        drift_data = pd.read_csv(drift_file)["true_value"].values
        with open(model_file, "r") as f:
            model_name = f.read().strip()
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"Retraining {model_name} using drift data...")

    with mlflow.start_run(run_name=f"retrain-{model_name}"):
        # When launched via `mlflow run`/mlflow.projects.run(), the six
        # hyperparameters below are already auto-logged onto this run from the
        # entry-point's CLI args before we get here — re-logging them would
        # raise (MLflow forbids changing an already-logged param's value). Only
        # log whatever isn't already present, so this works both via MLflow
        # Projects and when run bare as `python retrain.py`.
        existing_params = mlflow.active_run().data.params
        candidate_params = {
            "model_name": model_name,
            "ridge_alpha": args.ridge_alpha,
            "svm_c": args.svm_c,
            "svm_tol": args.svm_tol,
            "lstm_epochs": args.lstm_epochs,
            "lstm_lr": args.lstm_lr,
            "seq_length": args.seq_length,
            "trigger": "drift",
            "n_drift_rows": len(drift_data),
        }
        new_params = {k: v for k, v in candidate_params.items() if k not in existing_params}
        if new_params:
            mlflow.log_params(new_params)

        # The registry stays globally versioned (MLflow's own semantics), so
        # this tag is what makes "which models came out of that one session"
        # answerable - pair it with the versionedMR_version param logged below
        # to map a registry version back to its per-session VMR directory.
        mlflow.set_tag("harmone_session", SESSION_ID)

        # Preprocess data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(drift_data.reshape(-1, 1)).flatten()
        seq_length = int(args.seq_length)

        # Holdout split so a retrain's quality is actually measurable, instead of
        # silently deploying whatever the last retrain produced.
        split_idx = int(len(data_scaled) * 0.8)
        train_scaled, val_scaled = data_scaled[:split_idx], data_scaled[split_idx:]

        X_train, y_train = create_sequences(train_scaled, seq_length)
        X_val, y_val = create_sequences(val_scaled, seq_length)

        # Train model
        if model_name == "linear":
            model = Ridge(alpha=args.ridge_alpha)
            model.fit(X_train, y_train)
        elif model_name == "svm":
            model = SVR(kernel="linear", C=args.svm_c, tol=args.svm_tol)
            model.fit(X_train, y_train)
        elif model_name == "lstm":
            model = train_lstm(X_train, y_train, args.lstm_epochs, args.lstm_lr)
        else:
            print(f"Unknown model type: {model_name}")
            mlflow.set_tag("status", "failed_unknown_model")
            return

        # Evaluate on the held-out slice before this model goes live.
        if len(X_val) > 0:
            val_predictions = predict(model, model_name, X_val)
            val_r2 = r2_score(y_val, val_predictions)
            val_mae = mean_absolute_error(y_val, val_predictions)
            mlflow.log_metrics({"val_r2": val_r2, "val_mae": val_mae})
            print(f"🔹 Holdout R²: {val_r2:.4f}, MAE: {val_mae:.4f}")
        else:
            print("⚠️ Not enough drift data for a holdout split; skipping evaluation metrics.")

        # Inverse transform before saving (only the portion actually used for training)
        train_data_original = scaler.inverse_transform(train_scaled.reshape(-1, 1)).flatten()
        train_df = pd.DataFrame({"train_data": train_data_original})

        # Save retrained model to the existing versionedMR/models locations (unchanged)
        version = save_model_and_data(model, model_name, train_df)
        mlflow.log_param("versionedMR_version", version)

        # Versioned Model Repository pairs a model with the data distribution it
        # was trained on (for later drift-matching / reuse); mirror that pairing
        # into MLflow by attaching the same data.csv to this run as an artifact,
        # not just the model.
        version_data_path = os.path.join(base_dir, model_name, f"version_{version}", "data.csv")
        if os.path.exists(version_data_path):
            mlflow.log_artifact(version_data_path, artifact_path="training_data")

        # Register the model in the MLflow Model Registry for lineage/rollback.
        # input_example is required for the LSTM flavor's default serialization
        # format in newer MLflow versions, and is good practice generally (it lets
        # the registry capture and display the model's input/output schema).
        registered_name = f"harmone-regression-{model_name}"
        input_example = X_train[:1]
        # `name=` (not the deprecated positional `artifact_path=`): in MLflow 3.x
        # this is the logged model's NAME on the experiment's Models page. The old
        # hardcoded "model" made every entry there read literally "model", so the
        # page was a list of indistinguishable rows showing no version at all.
        # <model>-<session>-v<n> makes it self-describing and matches the on-disk
        # VMR path this model was written to.
        logged_model_name = f"{model_name}-{SESSION_ID}-v{version}"
        if model_name == "lstm":
            lstm_input_example = torch.tensor(input_example, dtype=torch.float32).unsqueeze(-1)
            # `serialization_format="pickle"` avoids the newer pt2/TensorSpec export
            # path, which imposes signature constraints this plain nn.Module isn't
            # written to satisfy; pickling the whole module is the long-standing,
            # well-supported option for a custom architecture like this one.
            mlflow.pytorch.log_model(
                model, name=logged_model_name, registered_model_name=registered_name,
                input_example=lstm_input_example, serialization_format="pickle",
            )
        else:
            mlflow.sklearn.log_model(model, name=logged_model_name, registered_model_name=registered_name, input_example=input_example)

        print(f"✔ {model_name} retraining completed.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ridge_alpha", type=float, default=200)
    parser.add_argument("--svm_c", type=float, default=0.08)
    parser.add_argument("--svm_tol", type=float, default=0.16)
    parser.add_argument("--lstm_epochs", type=float, default=50)
    parser.add_argument("--lstm_lr", type=float, default=0.001)
    parser.add_argument("--seq_length", type=float, default=5)
    return parser.parse_args()

if __name__ == "__main__":
    retrain(parse_args())
