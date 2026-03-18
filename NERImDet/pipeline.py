import argparse

from NER.inference import AnimalNER
from ImDet.inference import AnimalDetector


class AnimalPipeline:
    def __init__(self, ner_model_path: str, yolo_model_path: str, conf_threshold: float = 0.25):
        self.ner = AnimalNER(ner_model_path)
        self.detector = AnimalDetector(yolo_model_path, conf_threshold=conf_threshold)

    def run(self, text: str, image_path: str) -> bool:
        # Step 1: extract animal mentions from text
        entities = self.ner.predict(text)

        if not entities:
            print(f"[Pipeline] NER found nothing in: '{text}'")
            return False

        ner_animals = [e['entity'].lower().strip() for e in entities]
        print(f"[Pipeline] NER extracted: {ner_animals}")

        # Step 2: detect animal in image
        detected = self.detector.predict(image_path)

        if detected is None:
            print(f"[Pipeline] YOLO detected nothing")
            return False

        print(f"[Pipeline] YOLO detected: {detected}")

        # Step 3: exact case-insensitive match
        return detected.lower().strip() in ner_animals


def main(args):
    pipeline = AnimalPipeline(
        ner_model_path=args.ner_model,
        yolo_model_path=args.yolo_model,
        conf_threshold=args.conf
    )

    result = pipeline.run(text=args.text, image_path=args.image)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animal pipeline: text + image ? bool")
    parser.add_argument("--ner_model", type=str, required=True, help="Path to NER model")
    parser.add_argument("--yolo_model", type=str, required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = parser.parse_args()
    main(args)