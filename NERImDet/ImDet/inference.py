import argparse
from pathlib import Path

from ultralytics import YOLO


class AnimalDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def predict(self, image_path: str) -> str | None:
        """
        Run detection on image and return the highest-confidence animal class name.
        Returns None if nothing detected above threshold.
        """
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)

        best_class = None
        best_conf = 0.0

        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                if conf > best_conf:
                    best_conf = conf
                    best_class = result.names[int(box.cls)]

        return best_class


def main(args):
    detector = AnimalDetector(args.model_path, conf_threshold=args.conf)

    if args.image:
        animal = detector.predict(args.image)
        if animal:
            print(f"Detected: {animal}")
        else:
            print("No animal detected")

    elif args.images_dir:
        image_files = list(Path(args.images_dir).glob("*.jpg")) + \
                      list(Path(args.images_dir).glob("*.png"))

        for img_path in image_files:
            animal = detector.predict(str(img_path))
            result = animal if animal else "nothing detected"
            print(f"{img_path.name}: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animal detection inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained YOLO weights (.pt)")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--images_dir", type=str, help="Path to folder with images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    if not args.image and not args.images_dir:
        parser.error("Provide --image or --images_dir")

    main(args)