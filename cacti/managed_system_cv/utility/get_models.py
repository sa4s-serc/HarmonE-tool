import os
import urllib.request

os.makedirs("models", exist_ok=True)

YOLO_MODEL_URLS = {
    "yolo_n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "yolo_s": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
    "yolo_m": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
    # "yolo_l": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
    # "yolo_x": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
}

for model_name, url in YOLO_MODEL_URLS.items():
    dest = f"base_models/{model_name}.pt"
    if not os.path.exists(dest):
        try:
            print(f"Downloading {model_name} from {url}...")
            urllib.request.urlretrieve(url, dest)
            print(f"Saved {model_name} to {dest}")
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
    else:
        print(f"{model_name} already exists at {dest}")
