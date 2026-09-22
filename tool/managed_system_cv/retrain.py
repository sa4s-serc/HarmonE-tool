import os
import argparse
import shutil
import re
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import csv
import mlflow

# Add utility path to import drift utils, and the shared tool/ dir for energy_utils
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import energy_utils as pyRAPL
from console_utils import force_utf8_console
from session_utils import current_session_id, session_versioned_dir
from utility.drift_utils import luminance_histogram

# Own entry point (`mlflow run managed_system_cv -e retrain`) - see
# managed_system_regression/retrain.py.
force_utf8_console()
from torchvision.transforms.functional import adjust_brightness

# --- MLflow tracking setup ---
# Absolute path so this converges on the same DB regardless of the caller's cwd
# (bare `python retrain.py`, `mlflow run .`, or triggered from mape_logic/execute.py).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKING_DB_PATH = os.path.join(BASE_DIR, "..", "mlflow.db")
# Only set this ourselves if nothing upstream (mlflow.projects.run's caller, e.g.
# mape_logic/execute.py, or `mlflow run`'s own default) already configured one via
# env var — otherwise we'd orphan an inherited run ID in the wrong store (see
# managed_system_regression/retrain.py for the full explanation).
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{TRACKING_DB_PATH}"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "harmonica-cv"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("harmonica-cv")

# --- CONFIGURATION ---
DATA_DIR = Path("data/bdd100k")
MODELS_DIR = Path("base_models")
ACTIVE_MODELS_DIR = Path("models")
KNOWLEDGE_DIR = Path("knowledge")
# VMR is scoped per managed-system session, so version numbering restarts at 1
# each run and drift reuse only considers models this session trained - see
# tool/session_utils.py.
SESSION_ID = current_session_id(os.path.abspath(KNOWLEDGE_DIR))
VERSIONED_DIR = Path(session_versioned_dir(
    os.path.abspath(Path("versionedMR")), os.path.abspath(KNOWLEDGE_DIR)
))
ENERGY_LOG_FILE = KNOWLEDGE_DIR / "retrain_energy_log.csv"

# Reference data for drift comparison
REF_IMAGE_DIR = DATA_DIR / "images" / "test"
REF_LABEL_DIR = DATA_DIR / "labels" / "test" # Assuming YOLO labels exist here
N_REF_IMAGES = 1000

# Temporary directory for the augmented retraining set
RETRAIN_AUG_DIR = DATA_DIR / "images" / "retrain_augmented"
RETRAIN_AUG_LABELS_DIR = DATA_DIR / "labels" / "retrain_augmented"

# --- Statistically-driven augmentation factors (should match induce.py) ---
LUMINANCE_FACTOR = 0.35
CONTRAST_FACTOR = 0.25

# --- HELPER FUNCTIONS ---

def get_next_version(model_name):
    """Gets the next version number for a given model base name."""
    pattern = re.compile(f"{model_name}_v(\\d+)\\.pt")
    versions = [int(m.group(1)) for f in os.listdir(VERSIONED_DIR) if (m := pattern.match(f))]
    return max(versions + [0]) + 1

def get_distribution_stats(hist):
    """Calculates mean and std dev from a luminance histogram."""
    centers = np.linspace(0, 255, len(hist), endpoint=False) + (255 / len(hist)) / 2
    mean = np.sum(hist * centers)
    var = np.sum(hist * (centers - mean)**2)
    return mean, np.sqrt(max(var, 1e-9))

def log_energy(model_name, energy_uJ):
    """Appends the model name and its training energy consumption to the log file."""
    # Create file and header if it doesn't exist
    if not ENERGY_LOG_FILE.exists():
        with open(ENERGY_LOG_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["model_name", "energy_uJ"])

    with open(ENERGY_LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([model_name, energy_uJ])

def apply_darkness(image_array, factor):
    return adjust_brightness(image_array, factor)


def apply_fog(image_array, factor):
    """Reduces the contrast (std dev) of an image by a factor, preserving the mean."""
    mean = np.mean(image_array, axis=(0, 1), keepdims=True)
    foggy_array = np.clip((image_array - mean) * factor + mean, 0, 255)
    return foggy_array

# --- NEW: Superior Drift Classification Logic ---

def deduce_drift_type(ref_hist, current_hist):
    """Deduces drift type by comparing the relative change in luminance statistics."""
    ref_mean, ref_std = get_distribution_stats(ref_hist)
    cur_mean, cur_std = get_distribution_stats(current_hist)

    print(f"[DRIFT-DEDUCE] Reference Stats: Mean={ref_mean:.2f}, Std={ref_std:.2f}")
    print(f"[DRIFT-DEDUCE] Current Stats:  Mean={cur_mean:.2f}, Std={cur_std:.2f}")

    if ref_mean == 0 or ref_std == 0: return "clear"

    mean_change_ratio = cur_mean / ref_mean
    std_change_ratio = cur_std / ref_std
    print(f"[DRIFT-DEDUCE] Change Ratios: Mean={mean_change_ratio:.2%}, Std Dev={std_change_ratio:.2%}")

    is_dark = mean_change_ratio < 0.80
    is_foggy = std_change_ratio < 0.85

    if not is_dark and not is_foggy:
        print("[DRIFT-DEDUCE] No significant drop detected. Classifying as 'clear'.")
        return "clear"

    if is_dark and not is_foggy:
        print("[DRIFT-DEDUCE] Significant drop in mean only. Classifying as 'dark'.")
        return "dark"

    if is_foggy and not is_dark:
        print("[DRIFT-DEDUCE] Significant drop in std dev only. Classifying as 'fog'.")
        return "fog"

    if is_dark and is_foggy:
        print("[DRIFT-DEDUCE] Both metrics dropped, but mean drop is larger or equal. Classifying as 'dark'.")
        return "dark"

    return "clear"

def create_augmented_retrain_set(image_paths, label_dir, drift_type):
    """Creates an augmented dataset using statistically pure methods."""
    if RETRAIN_AUG_DIR.exists(): shutil.rmtree(RETRAIN_AUG_DIR)
    if RETRAIN_AUG_LABELS_DIR.exists(): shutil.rmtree(RETRAIN_AUG_LABELS_DIR)
    RETRAIN_AUG_DIR.mkdir(parents=True, exist_ok=True)
    RETRAIN_AUG_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    total_hist, images_processed = None, 0
    print(f"Creating augmented retraining set for drift type: '{drift_type}'...")
    for img_path in tqdm(image_paths, desc="Augmenting Images"):
        try:
            with Image.open(img_path).convert("RGB") as img:
                if drift_type != "clear":
                    img_array = np.array(img, dtype=np.float32)
                    if drift_type == "dark":
                        aug = apply_darkness(img, LUMINANCE_FACTOR)
                        aug_array = np.array(aug, dtype=np.float32)
                    elif drift_type == "fog":
                        aug_array = apply_fog(img_array, CONTRAST_FACTOR)
                    aug_img = Image.fromarray(aug_array.astype(np.uint8))
                else:
                    aug_img = img

                aug_img.save(RETRAIN_AUG_DIR / img_path.name)

                label_path = label_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy(label_path, RETRAIN_AUG_LABELS_DIR / label_path.name)

                hist = luminance_histogram(RETRAIN_AUG_DIR / img_path.name)
                if hist is not None:
                    if total_hist is None: total_hist = np.zeros_like(hist)
                    total_hist += hist
                    images_processed += 1
        except Exception:
            continue

    avg_hist = (total_hist / images_processed) if images_processed > 0 else None
    return avg_hist

# --- MAIN RETRAIN FUNCTION ---

def retrain_yolo(args):
    pyRAPL.setup()

    try:
        with open(KNOWLEDGE_DIR / "model.csv", "r") as f:
            model_name = f.read().strip().split('_v')[0]
    except FileNotFoundError:
        print("❌ knowledge/model.csv not found. Defaulting to yolo_s.")
        model_name = "yolo_s"

    model_path = MODELS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        print(f"❌ Base model {model_path} is missing. Cannot retrain.")
        return

    try:
        df = pd.read_csv(KNOWLEDGE_DIR / "predictions.csv")
        if len(df) < N_REF_IMAGES:
            print("❌ Not enough predictions to deduce drift. Aborting retrain.")
            return

        current_hists_str = df["histogram"].iloc[-N_REF_IMAGES:]
        current_hists = np.array([np.fromstring(h, sep=' ') for h in current_hists_str if h])
        current_dist = np.mean(current_hists, axis=0)

        ref_image_paths = sorted(list(REF_IMAGE_DIR.glob("*.jpg")))[:N_REF_IMAGES]
        ref_hists = np.array([h for p in ref_image_paths if (h := luminance_histogram(p)) is not None])
        ref_dist = np.mean(ref_hists, axis=0)

        drift_type = deduce_drift_type(ref_dist, current_dist)
    except Exception as e:
        print(f"❌ Error during drift deduction: {e}. Defaulting to 'clear'.")
        drift_type = "clear"

    image_paths_to_augment = sorted(list(REF_IMAGE_DIR.glob("*.jpg")))[:N_REF_IMAGES]
    avg_retrain_hist = create_augmented_retrain_set(image_paths_to_augment, REF_LABEL_DIR, drift_type)

    if avg_retrain_hist is None:
        print("❌ Failed to create augmented dataset. Aborting.")
        return

    train_yaml_path = "bdd100k_retrain_temp.yaml"
    with open(train_yaml_path, "w") as f:
        f.write(f"""
path: {DATA_DIR.resolve()}
train: {RETRAIN_AUG_DIR.relative_to(DATA_DIR)}
val: {RETRAIN_AUG_DIR.relative_to(DATA_DIR)}
nc: 80
names: {{ {', '.join([f'{i}: {i}' for i in range(80)])} }}
""")

    print(f"🚀 Loading model {model_path} for fine-tuning...")
    model = YOLO(str(model_path))

    for param in model.model.parameters():
        param.requires_grad = False
    for name, module in model.model.named_modules():
        if isinstance(module, type(model.model.model[-1])):
            for param in module.parameters():
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Fine-tuning setup complete. Trainable parameters: {trainable_params}")

    batch = args.batch
    if batch is None:
        batch = 4 if model_name == "yolo_n" else (1 if model_name == "yolo_m" else 2)

    meter = pyRAPL.Measurement("model_training")

    with mlflow.start_run(run_name=f"retrain-{model_name}"):
        # See managed_system_regression/retrain.py: `epochs`/`imgsz`/`patience` are
        # already auto-logged by MLflow Projects when launched via `mlflow run`
        # (they're declared MLproject parameters); only log what's missing so this
        # also works when this script is run bare.
        existing_params = mlflow.active_run().data.params
        candidate_params = {
            "model_name": model_name,
            "drift_type": drift_type,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": batch,
            "patience": args.patience,
            "trigger": "drift",
        }
        new_params = {k: v for k, v in candidate_params.items() if k not in existing_params}
        if new_params:
            mlflow.log_params(new_params)

        # Registry versioning stays global; this tag scopes a registered
        # version back to the session that produced it (see
        # managed_system_regression/retrain.py for the same rationale).
        mlflow.set_tag("harmone_session", SESSION_ID)

        meter.begin()

        # Ultralytics detects this already-active MLflow run and logs its own
        # per-epoch metrics/params into it automatically.
        model.train(
            data=train_yaml_path, epochs=int(args.epochs), imgsz=int(args.imgsz),
            batch=batch,
            workers=2, patience=int(args.patience), pretrained=True, cache="disk")

        meter.end()

        energy_used = meter.result.pkg[0] if meter.result.pkg else 0.0
        log_energy(model_name, energy_used)
        mlflow.log_metric("training_energy_uJ", energy_used)
        print(f"⚡ Energy consumed for training: {energy_used} uJ")

        v = get_next_version(model_name)
        new_version_base_name = f"{model_name}_v{v}"

        versioned_model_path = VERSIONED_DIR / f"{new_version_base_name}.pt"
        versioned_hist_path = VERSIONED_DIR / f"{new_version_base_name}_hist.json"
        model.save(str(versioned_model_path))
        with open(versioned_hist_path, 'w') as f:
            json.dump({"average_histogram": avg_retrain_hist.tolist()}, f, indent=4)
        print(f"✔ Saved versioned model to {versioned_model_path}")
        print(f"✔ Saved versioned histogram to {versioned_hist_path}")
        mlflow.log_param("versionedMR_version", v)

        # Log our own known-good copy of the weight file (rather than relying on
        # guessing Ultralytics' internal artifact layout) and register it, so the
        # Model Registry always has a valid pointer to this version's weights.
        mlflow.log_artifact(str(versioned_model_path), artifact_path="model")
        # Versioned Model Repository pairs a model with the data distribution it
        # was trained on; the luminance histogram is that pairing for CV.
        mlflow.log_artifact(str(versioned_hist_path), artifact_path="training_data")
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(
            f"runs:/{run_id}/model/{versioned_model_path.name}",
            f"harmone-cv-{model_name}",
        )

        active_model_path = ACTIVE_MODELS_DIR / f"{model_name}.pt"
        shutil.copy(versioned_model_path, active_model_path)
        print(f"✔ Updated active model at {active_model_path}")

    os.remove(train_yaml_path)
    shutil.rmtree(RETRAIN_AUG_DIR)
    shutil.rmtree(RETRAIN_AUG_LABELS_DIR)
    print("✔ Cleanup complete.")
    print(f"✅ {model_name} retrained successfully -> version {v}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--imgsz", type=float, default=640)
    parser.add_argument("--batch", type=float, default=None)
    parser.add_argument("--patience", type=float, default=3)
    return parser.parse_args()

if __name__ == "__main__":
    retrain_yolo(parse_args())