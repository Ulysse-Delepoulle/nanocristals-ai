"""
Single-image inference with a trained YOLOv8 model.

Displays the annotated image and prints per-class detection counts.
Optionally saves the annotated result to disk.

Usage:
  python src/predict.py --image path/to/image.jpg
  python src/predict.py --image path/to/image.jpg --model path/to/best.pt --save output.jpg
"""

import argparse

import matplotlib.pyplot as plt
from ultralytics import YOLO

DEFAULT_MODEL = "yolo_results/train2/weights/best.pt"


def predict(model_path, image_path, conf=0.25, save_path=None):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, verbose=False)

    for result in results:
        annotated = result.plot(show=False)

        plt.figure(figsize=(10, 8))
        plt.imshow(annotated[..., ::-1])  # BGR → RGB
        plt.axis("off")
        plt.title(f"Detections (conf >= {conf})")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Annotated image saved to {save_path}")
        plt.show()

        class_counts = {}
        for box in result.boxes:
            name = result.names[int(box.cls.item())]
            class_counts[name] = class_counts.get(name, 0) + 1

        print("\nDetections:")
        if class_counts:
            for cls, count in sorted(class_counts.items()):
                print(f"  {cls}: {count}")
        else:
            print("  None")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 inference on a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to model weights (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", default=None, help="Path to save the annotated output image")
    args = parser.parse_args()

    predict(args.model, args.image, args.conf, args.save)


if __name__ == "__main__":
    main()
