"""
Parse YOLOv8 training results and print a clean metrics summary.

Usage:
  python src/evaluate.py
  python src/evaluate.py --results path/to/results.csv
"""

import argparse
import csv


DEFAULT_RESULTS = "yolo_results/train2/results.csv"


def load_results(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [{k.strip(): v.strip() for k, v in row.items()} for row in reader]


def best_epoch(rows, metric="metrics/mAP50(B)"):
    return max(rows, key=lambda r: float(r[metric]))


def print_table(row, label):
    print(f"\n{'─' * 40}")
    print(f" {label}")
    print(f"{'─' * 40}")
    print(f"  Epoch        {row['epoch']:>10}")
    print(f"  mAP@50       {float(row['metrics/mAP50(B)']):>10.4f}")
    print(f"  mAP@50-95    {float(row['metrics/mAP50-95(B)']):>10.4f}")
    print(f"  Precision    {float(row['metrics/precision(B)']):>10.4f}")
    print(f"  Recall       {float(row['metrics/recall(B)']):>10.4f}")
    print(f"  Val box loss {float(row['val/box_loss']):>10.4f}")
    print(f"  Val cls loss {float(row['val/cls_loss']):>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Summarize YOLOv8 training metrics")
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
        help=f"Path to results.csv (default: {DEFAULT_RESULTS})",
    )
    args = parser.parse_args()

    rows = load_results(args.results)
    print_table(best_epoch(rows), "Best epoch  (highest mAP@50)")
    print_table(rows[-1], "Final epoch")
    print()


if __name__ == "__main__":
    main()
