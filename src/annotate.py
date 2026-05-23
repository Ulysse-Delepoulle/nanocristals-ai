"""
Nanocrystal annotation utilities - three subcommands:

  annotate   Draw bounding boxes on images interactively (saves JSON)
  convert    Convert JSON annotation files to YOLO .txt format
  visualize  Draw YOLO labels on an image to verify correctness

Classes: cubic (0) | pseudo-cubic (1) | ball (2)

Usage:
  # Step 1: annotate images interactively
  python src/annotate.py annotate --input images/ --output annotations/

  # Step 2: convert JSON to YOLO format
  python src/annotate.py convert --json-dir annotations/ --image-dir images/ --output labels/

  # Verify a label file visually
  python src/annotate.py visualize --image images/001.jpg --labels labels/001.txt
  python src/annotate.py visualize --image images/001.jpg --labels labels/001.txt --save out.jpg
"""

import cv2
import json
import os
import argparse

CLASSES = ["cubic", "pseudo-cubic", "ball"]

# ── Interactive annotator ───────────────────────────────────────────────────

_annotations = []
_current = None
_image = None
_image_copy = None
_done = False


def _on_mouse(event, x, y, flags, param):
    global _current, _image, _image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        _current = {"start": (x, y), "end": (x, y)}

    elif event == cv2.EVENT_MOUSEMOVE and _current is not None:
        _current["end"] = (x, y)
        _image_copy = _image.copy()
        cv2.rectangle(_image_copy, _current["start"], _current["end"], (0, 255, 0), 2)
        cv2.imshow("annotate", _image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        _current["end"] = (x, y)
        cv2.rectangle(_image, _current["start"], _current["end"], (0, 255, 0), 2)
        cv2.imshow("annotate", _image)
        label = input(f"  Label {CLASSES}: ").strip()
        if label in CLASSES:
            _current["label"] = label
            _annotations.append(dict(_current))
            print(f"  Saved: {label}")
        else:
            print(f"  Unknown label '{label}' - discarded")
        _current = None


def _annotate_single(image_path, output_dir):
    global _annotations, _done, _image, _image_copy
    _annotations = []
    _done = False

    _image = cv2.imread(image_path)
    if _image is None:
        print(f"Failed to load: {image_path}")
        return
    _image_copy = _image.copy()

    cv2.namedWindow("annotate")
    cv2.setMouseCallback("annotate", _on_mouse)
    print(f"\nAnnotating: {image_path}")
    print("  Draw boxes with left-click drag. Press [n] for next image, [q] to quit.")

    while not _done:
        cv2.imshow("annotate", _image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            _done = True

    cv2.destroyAllWindows()

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base}_annotations.json")
    with open(out_path, "w") as f:
        json.dump(_annotations, f, indent=4)
    print(f"  Saved {len(_annotations)} annotation(s) to {out_path}")


def cmd_annotate(args):
    image_folder = args.input
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
    files = sorted(f for f in os.listdir(image_folder) if f.endswith(extensions))
    print(f"Found {len(files)} images in {image_folder}")
    for fname in files:
        _annotate_single(os.path.join(image_folder, fname), args.output)


# ── JSON → YOLO converter ───────────────────────────────────────────────────

def _to_yolo_coords(size, box):
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box["start"][0] + box["end"][0]) / 2.0 * dw
    y = (box["start"][1] + box["end"][1]) / 2.0 * dh
    w = abs(box["end"][0] - box["start"][0]) * dw
    h = abs(box["end"][1] - box["start"][1]) * dh
    return x, y, w, h


def cmd_convert(args):
    json_folder = args.json_dir
    image_folder = args.image_dir
    out_folder = args.output
    os.makedirs(out_folder, exist_ok=True)

    json_files = [f for f in os.listdir(json_folder) if f.endswith("_annotations.json")]
    print(f"Converting {len(json_files)} annotation files...")

    for json_file in json_files:
        base = json_file.replace("_annotations.json", "")

        img_path = None
        for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"):
            candidate = os.path.join(image_folder, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"  Image not found for {json_file} - skipped")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  Failed to load image: {img_path} - skipped")
            continue
        h, w = img.shape[:2]

        with open(os.path.join(json_folder, json_file)) as f:
            annotations = json.load(f)

        lines = []
        for ann in annotations:
            label = ann.get("label", "")
            if label not in CLASSES:
                print(f"  Unknown label '{label}' in {json_file} - skipped")
                continue
            class_id = CLASSES.index(label)
            xc, yc, wn, hn = _to_yolo_coords((w, h), ann)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        out_path = os.path.join(out_folder, base + ".txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  {json_file} → {out_path}  ({len(lines)} boxes)")


# ── YOLO label visualizer ───────────────────────────────────────────────────

def cmd_visualize(args):
    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to load: {args.image}")
        return
    h, w = image.shape[:2]

    with open(args.labels) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            xc = float(parts[1]) * w
            yc = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h
            x1, y1 = int(xc - bw / 2), int(yc - bh / 2)
            x2, y2 = int(xc + bw / 2), int(yc + bh / 2)
            label = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            ty = max(y1 - 10, 10)
            cv2.putText(image, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if args.save:
        cv2.imwrite(args.save, image)
        print(f"Saved annotated image to {args.save}")
    else:
        cv2.imshow("labels", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nanocrystal annotation utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ann = sub.add_parser("annotate", help="Interactively annotate images (saves JSON)")
    p_ann.add_argument("--input", required=True, help="Folder of images to annotate")
    p_ann.add_argument("--output", default="annotations", help="Folder to save JSON files")

    p_conv = sub.add_parser("convert", help="Convert JSON annotations to YOLO .txt format")
    p_conv.add_argument("--json-dir", required=True, help="Folder of *_annotations.json files")
    p_conv.add_argument("--image-dir", required=True, help="Folder of corresponding images")
    p_conv.add_argument("--output", default="labels", help="Output folder for YOLO .txt files")

    p_viz = sub.add_parser("visualize", help="Draw YOLO labels on an image")
    p_viz.add_argument("--image", required=True, help="Path to image")
    p_viz.add_argument("--labels", required=True, help="Path to YOLO .txt label file")
    p_viz.add_argument("--save", default=None, help="Save annotated image instead of displaying")

    args = parser.parse_args()
    {"annotate": cmd_annotate, "convert": cmd_convert, "visualize": cmd_visualize}[args.cmd](args)


if __name__ == "__main__":
    main()
