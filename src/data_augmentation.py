"""
Patch-based data augmentation for nanocrystal object detection.

Generates synthetic YOLO-format training images by compositing individual
crystal cutouts onto SEM background images with random rotation, scale,
and position. One synthetic image is produced per source cutout.

Folder structure expected (relative to working directory):
  background/background_1.png   -- SEM background tile
  cubic/                        -- cubic crystal cutout PNGs (RGBA or RGB)
  pseudo-cubic/                 -- pseudo-cubic cutout PNGs
  ball/                         -- ball cutout PNGs

Output:
  img_patch/   -- generated composite images
  txt/         -- corresponding YOLO label files

Usage:
  python src/data_augmentation.py
  python src/data_augmentation.py --classes cubic ball --scale-min 0.5 --scale-max 2.0
"""

import os
import random
import argparse

from PIL import Image


CLASS_CONFIG = {
    "cubic":        {"class_id": 0, "folder": "cubic",        "prefix": "cubic_"},
    "pseudo-cubic": {"class_id": 1, "folder": "pseudo-cubic", "prefix": "pseudo-cubic_"},
    "ball":         {"class_id": 2, "folder": "ball",         "prefix": "ball_"},
}


def convert_to_yolo_format(x, y, w, h, image_width, image_height):
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    return x_center, y_center, w / image_width, h / image_height


def find_next_index(folder, prefix, suffix):
    """Return the next available integer index so existing files are not overwritten."""
    max_index = -1
    for filename in os.listdir(folder):
        if filename.startswith(prefix) and filename.endswith(suffix):
            index_part = filename[len(prefix): -len(suffix)]
            if index_part.isdigit():
                max_index = max(max_index, int(index_part))
    return max_index + 1


def augment_class(class_name, cfg, background_path, output_img_folder, output_txt_folder,
                  scale_range=(0.7, 1.5)):
    object_folder = cfg["folder"]
    if not os.path.isdir(object_folder):
        print(f"  [{class_name}] skipped: folder '{object_folder}' not found")
        return 0

    object_files = [f for f in os.listdir(object_folder) if f.lower().endswith(".png")]
    if not object_files:
        print(f"  [{class_name}] skipped: no PNG files in '{object_folder}'")
        return 0

    background = Image.open(background_path)
    bg_width, bg_height = background.size
    prefix = cfg["prefix"]
    class_id = cfg["class_id"]
    start_index = find_next_index(output_img_folder, prefix, ".png")

    for idx, obj_file in enumerate(object_files, start=start_index):
        obj_img = Image.open(os.path.join(object_folder, obj_file))

        angle = random.uniform(0, 360)
        obj_img = obj_img.rotate(angle, expand=True)

        scale = random.uniform(*scale_range)
        new_w = max(1, int(obj_img.width * scale))
        new_h = max(1, int(obj_img.height * scale))
        obj_img = obj_img.resize((new_w, new_h))

        max_x = max(bg_width - new_w, 0)
        max_y = max(bg_height - new_h, 0)
        x_off = random.randint(0, max_x) if max_x else 0
        y_off = random.randint(0, max_y) if max_y else 0

        bg = background.copy()
        mask = obj_img if obj_img.mode == "RGBA" else None
        bg.paste(obj_img, (x_off, y_off), mask)

        base_name = f"{prefix}{idx}"
        bg.save(os.path.join(output_img_folder, f"{base_name}.png"))

        xc, yc, wn, hn = convert_to_yolo_format(x_off, y_off, new_w, new_h, bg_width, bg_height)
        with open(os.path.join(output_txt_folder, f"{base_name}.txt"), "w") as f:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    print(f"  [{class_name}] {len(object_files)} images generated (starting at index {start_index})")
    return len(object_files)


def main():
    parser = argparse.ArgumentParser(
        description="Patch-based synthetic data augmentation for nanocrystal detection"
    )
    parser.add_argument(
        "--background",
        default="background/background_1.png",
        help="SEM background image to composite objects onto",
    )
    parser.add_argument(
        "--output-images", default="img_patch", help="Output folder for synthetic images"
    )
    parser.add_argument(
        "--output-labels", default="txt", help="Output folder for YOLO .txt label files"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(CLASS_CONFIG.keys()),
        choices=list(CLASS_CONFIG.keys()),
        help="Classes to augment (default: all three)",
    )
    parser.add_argument("--scale-min", type=float, default=0.7, help="Minimum scale factor")
    parser.add_argument("--scale-max", type=float, default=1.5, help="Maximum scale factor")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.output_images, exist_ok=True)
    os.makedirs(args.output_labels, exist_ok=True)

    print(f"Augmenting classes: {args.classes}")
    total = 0
    for cls in args.classes:
        total += augment_class(
            cls,
            CLASS_CONFIG[cls],
            args.background,
            args.output_images,
            args.output_labels,
            scale_range=(args.scale_min, args.scale_max),
        )
    print(f"\nDone. {total} synthetic images generated.")


if __name__ == "__main__":
    main()
