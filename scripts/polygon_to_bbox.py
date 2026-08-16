"""
polygon_to_bbox.py — Convert polygon/segmentation annotations to YOLO bounding box format.

Processes label files in a YOLO dataset directory structure and converts any polygon
annotation lines (>5 values) to bounding box format (exactly 5 values:
class_id x_center y_center width height).

Usage:
    python polygon_to_bbox.py --dataset /path/to/dataset

The dataset directory should contain train/labels/, valid/labels/, and/or test/labels/.
Originals are backed up automatically before conversion.
"""

import argparse
import shutil
from pathlib import Path


def is_polygon_line(line):
    parts = line.strip().split()
    return len(parts) > 5


def polygon_to_bbox(line):
    """Convert a polygon annotation line to bounding box format.

    Input:  "class_id x1 y1 x2 y2 x3 y3 ... xN yN" (normalized coordinates)
    Output: "class_id x_center y_center width height"
    """
    parts = line.strip().split()
    class_id = parts[0]
    coords = [float(v) for v in parts[1:]]

    xs = coords[0::2]
    ys = coords[1::2]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def backup_labels(label_dir):
    """Back up label directory to a sibling *_backup_polygon directory."""
    backup_dir = label_dir.parent / "labels_backup_polygon"
    if backup_dir.exists():
        print(f"  Backup already exists: {backup_dir} — skipping backup")
        return
    shutil.copytree(label_dir, backup_dir)
    print(f"  Backup created: {backup_dir}")


def process_label_file(filepath):
    """Convert polygon lines in a single label file. Returns count of converted lines."""
    lines = filepath.read_text().splitlines()
    new_lines = []
    converted = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if is_polygon_line(stripped):
            new_lines.append(polygon_to_bbox(stripped))
            converted += 1
        else:
            new_lines.append(stripped)

    filepath.write_text("\n".join(new_lines) + "\n" if new_lines else "")
    return converted


def process_directory(label_dir):
    """Process all .txt label files in a directory. Returns (files, converted_lines)."""
    txt_files = sorted(label_dir.glob("*.txt"))
    total_converted = 0

    for f in txt_files:
        total_converted += process_label_file(f)

    return len(txt_files), total_converted


def verify_no_polygons(label_dir):
    """Count remaining polygon lines across all label files. Should be 0 after conversion."""
    remaining = 0
    for f in sorted(label_dir.glob("*.txt")):
        for line in f.read_text().splitlines():
            if line.strip() and is_polygon_line(line):
                remaining += 1
    return remaining


def main():
    parser = argparse.ArgumentParser(
        description="Convert polygon annotations to YOLO bounding box format"
    )
    parser.add_argument(
        "--dataset", required=True, type=str,
        help="Path to dataset root containing train/valid/test subdirectories"
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        print(f"Error: dataset directory not found: {dataset}")
        return

    splits = ["train", "valid", "test"]
    grand_total_files = 0
    grand_total_converted = 0

    for split in splits:
        label_dir = dataset / split / "labels"
        if not label_dir.is_dir():
            print(f"[{split}] labels directory not found — skipping")
            continue

        print(f"[{split}] Processing {label_dir}")
        backup_labels(label_dir)

        files, converted = process_directory(label_dir)
        grand_total_files += files
        grand_total_converted += converted
        print(f"  Files processed: {files}")
        print(f"  Lines converted: {converted}")

        remaining = verify_no_polygons(label_dir)
        if remaining == 0:
            print(f"  Verification: PASS — 0 polygon lines remaining")
        else:
            print(f"  Verification: FAIL — {remaining} polygon lines still present")

    print(f"\nSummary: {grand_total_files} files processed, "
          f"{grand_total_converted} polygon lines converted to bbox")


if __name__ == "__main__":
    main()
