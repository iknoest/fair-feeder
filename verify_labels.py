"""
verify_labels.py — Visual diagnostic for YOLO label verification.

Draws bounding boxes on a random sample of images from a YOLO dataset
to visually confirm that labels are correctly aligned with objects.

Usage:
    python verify_labels.py --dataset /path/to/dataset --split train --samples 12
    python verify_labels.py --dataset /path/to/dataset --split train --samples 12 --save output.jpg
"""

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["Dan", "Sanbo", "Dan_hand", "Bowl", "Kibble"]
CLASS_COLORS = {
    0: (255, 100, 0),    # Dan — Blue (BGR)
    1: (0, 165, 255),    # Sanbo — Orange
    2: (0, 255, 0),      # Dan_hand — Green
    3: (0, 255, 255),    # Bowl — Yellow
    4: (255, 0, 255),    # Kibble — Magenta
}

THUMB_W, THUMB_H = 320, 240
GRID_COLS = 4


def load_labels(label_path):
    """Read a YOLO label file. Returns list of (class_id, x_ctr, y_ctr, w, h)."""
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        x_ctr, y_ctr, w, h = map(float, parts[1:])
        labels.append((cls, x_ctr, y_ctr, w, h))
    return labels


def draw_boxes(image, labels):
    """Draw colour-coded bounding boxes and class names on image."""
    img_h, img_w = image.shape[:2]
    for cls, x_ctr, y_ctr, w, h in labels:
        x1 = int((x_ctr - w / 2) * img_w)
        y1 = int((y_ctr - h / 2) * img_h)
        x2 = int((x_ctr + w / 2) * img_w)
        y2 = int((y_ctr + h / 2) * img_h)
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, name, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image


def create_grid(images, cols=GRID_COLS):
    """Arrange thumbnail images into a grid."""
    rows = math.ceil(len(images) / cols)
    # Pad with blank images if needed
    while len(images) < rows * cols:
        images.append(np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8))
    grid_rows = []
    for r in range(rows):
        row_imgs = images[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_imgs))
    return np.vstack(grid_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Visual diagnostic — draw YOLO bounding boxes on sample images"
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset root")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--samples", type=int, default=12, help="Number of sample images (default: 12)")
    parser.add_argument("--save", type=str, default=None, help="Save grid to file instead of displaying")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    image_dir = dataset / args.split / "images"
    label_dir = dataset / args.split / "labels"

    if not image_dir.is_dir():
        print(f"Error: image directory not found: {image_dir}")
        return

    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not image_files:
        print(f"Error: no images found in {image_dir}")
        return

    sample_size = min(args.samples, len(image_files))
    sampled = random.sample(image_files, sample_size)
    print(f"Sampled {sample_size} images from {image_dir}")

    thumbnails = []
    for img_path in sampled:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Warning: could not read {img_path.name}")
            continue

        label_path = label_dir / (img_path.stem + ".txt")
        labels = load_labels(label_path)
        image = draw_boxes(image, labels)
        thumb = cv2.resize(image, (THUMB_W, THUMB_H))
        thumbnails.append(thumb)

    if not thumbnails:
        print("Error: no images could be loaded")
        return

    grid = create_grid(thumbnails)

    if args.save:
        cv2.imwrite(args.save, grid)
        print(f"Grid saved to {args.save}")
    else:
        cv2.imshow("Label Verification", grid)
        print("Press any key to close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\nClass Legend:")
    for i, name in enumerate(CLASS_NAMES):
        b, g, r = CLASS_COLORS[i]
        print(f"  {i}: {name} — RGB({r}, {g}, {b})")


if __name__ == "__main__":
    main()
