import argparse
import os
import shutil
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


def build_yolo_dataset(src_root: str, dst_root: str, class_names: list[str]):
    images_dir = Path(dst_root) / "images"
    labels_dir = Path(dst_root) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    # Iterate over class folders in source dataset
    for class_name in os.listdir(src_root):
        class_dir = Path(src_root) / class_name
        if not class_dir.is_dir():
            continue

        # Maintain consistent class-to-index mapping
        if class_name not in class_names:
            class_names.append(class_name)
        class_id = class_names.index(class_name)

        labels_src = class_dir / "Label"
        if not labels_src.exists():
            print(f"WARNING: No Labels folder in {class_dir}")
            continue

        # Process each annotation file
        for txt_file in labels_src.glob("*.txt"):
            img_file = class_dir / (txt_file.stem + ".jpg")
            if not img_file.exists():
                continue

            # Read image size for normalization
            from PIL import Image
            w_img, h_img = Image.open(img_file).size

            yolo_lines = []

            # Convert bounding boxes to YOLO format (cx, cy, w, h)
            with open(txt_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    _, x1, y1, x2, y2 = parts
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                    cx = (x1 + x2) / 2 / w_img
                    cy = (y1 + y2) / 2 / h_img
                    bw = (x2 - x1) / w_img
                    bh = (y2 - y1) / h_img

                    yolo_lines.append(
                        f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                    )

            # Copy image and write corresponding label file
            shutil.copy(img_file, images_dir / img_file.name)
            with open(labels_dir / txt_file.name, "w") as f:
                f.write("\n".join(yolo_lines))

            copied += 1

    print(f"Copied {copied} images to {dst_root}")
    return class_names


def write_yaml(yaml_path: str, train_dir: str, val_dir: str, class_names: list[str]):
    """Write YOLO dataset config file."""
    train_abs = str((Path(train_dir) / "images").resolve())
    val_abs = str((Path(val_dir) / "images").resolve())

    with open(yaml_path, "w") as f:
        f.write(f"train: {train_abs}\n")
        f.write(f"val: {val_abs}\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")


def main(args):
    class_names = []

    print("Preparing train dataset...")
    train_dst = os.path.join(args.output_dir, "train")
    class_names = build_yolo_dataset(args.train_path, train_dst, class_names)

    print("Preparing val dataset...")
    val_dst = os.path.join(args.output_dir, "val")
    build_yolo_dataset(args.test_path, val_dst, class_names)

    print(f"Classes ({len(class_names)}): {class_names}")

    yaml_path = os.path.join(args.output_dir, "dataset.yaml")
    write_yaml(yaml_path, train_dst, val_dst, class_names)

    # Save class names for inference
    names_path = os.path.join(args.output_dir, "class_names.txt")
    with open(names_path, "w") as f:
        f.write("\n".join(class_names))

    print("Training YOLO model...")
    model = YOLO(args.base_model)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=args.output_dir,
        name="weights",
        exist_ok=True,
    )

    print(f"Done. Best weights: {args.output_dir}/weights/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO animal detector")
    parser.add_argument("--train_path", type=str, required=True, help="Path to train folder")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test folder")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/yolo", help="Output directory")
    parser.add_argument("--base_model", type=str, default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    main(args)