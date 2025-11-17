import os
import json
from pathlib import Path
from tqdm import tqdm

# Map BDD100K categories to YOLO class names (COCO format used by YOLO)
CATEGORY_MAP = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "bike": "bicycle",
    "motor": "motorbike",
    "person": "person",
    "rider": "person",
    "traffic sign": "traffic sign",
    "traffic light": "traffic light",
    # Ignoring area/alternative, area/drivable, lane/* for YOLO
}

YOLO_CLASS_LIST = [
    "person", "bicycle", "car", "motorbike", "bus", "truck", "traffic light", "traffic sign"
]
YOLO_CLASS_INDEX = {name: idx for idx, name in enumerate(YOLO_CLASS_LIST)}


def convert_box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = abs(x2 - x1) / img_w
    height = abs(y2 - y1) / img_h
    return x_center, y_center, width, height


def convert_labels_to_yolo(label_dir, img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    label_files = list(Path(label_dir).glob("*.json"))
    
    for label_file in tqdm(label_files):
        with open(label_file, "r") as f:
            data = json.load(f)

        name = data["name"]
        frame = data["frames"][0]  # only one frame per file in BDD100K
        objects = frame.get("objects", [])

        # Get image path and its resolution
        image_path = Path(img_dir) / f"{name}.jpg"
        if not image_path.exists():
            continue

        from PIL import Image
        img = Image.open(image_path)
        img_w, img_h = img.size

        yolo_lines = []
        for obj in objects:
            category = obj["category"]
            if "box2d" not in obj:
                continue  # Skip poly2d objects for YOLO

            mapped_category = CATEGORY_MAP.get(category)
            if mapped_category is None or mapped_category not in YOLO_CLASS_INDEX:
                continue

            cls_id = YOLO_CLASS_INDEX[mapped_category]
            box = obj["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            x_center, y_center, w, h = convert_box_to_yolo(x1, y1, x2, y2, img_w, img_h)

            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            output_path = Path(output_dir) / f"{name}.txt"
            with open(output_path, "w") as out_f:
                out_f.write("\n".join(yolo_lines))


if __name__ == "__main__":
    # Convert train and test sets
    convert_labels_to_yolo("data/bdd100k/labels/train", "data/bdd100k/images/train", "data/bdd100k/label_yolo/train")
    convert_labels_to_yolo("data/bdd100k/labels/test", "data/bdd100k/images/test", "data/bdd100k/label_yolo/test")
