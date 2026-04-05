"""
train.py - YOLOv11 training entrypoint for the current Fair Feeder dataset.

This is the script version of the training flow documented in fair_feeder_v14.ipynb.
Use it when you want a reproducible CLI path instead of driving training from a notebook.

Usage:
    python train.py --data data.yaml --weights yolo11s.pt
    python train.py --data data.yaml --weights yolo11s.pt --epochs 50 --batch 4
"""

import argparse


def get_device():
    """Auto-detect GPU or fall back to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"Device: GPU - {name}")
            return 0
        print("Device: CPU (no GPU detected - training will be slow)")
        return "cpu"
    except ImportError:
        print("Device: CPU (torch not available)")
        return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 on the current Fair Feeder dataset"
    )
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", default="yolo11s.pt", help="Pretrained model (default: yolo11s.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[1280, 720], help="Image size as W H (default: 1280 720)")
    parser.add_argument("--device", default=None, help="Device override (e.g. 0, cpu)")
    parser.add_argument("--project", default="runs/fair-feeder", help="Output project directory")
    parser.add_argument("--name", default="v14", help="Experiment name")
    args = parser.parse_args()

    from ultralytics import YOLO

    device = args.device if args.device is not None else get_device()

    print("\nFair Feeder Training")
    print(f"  Model:   {args.weights}")
    print(f"  Data:    {args.data}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch}")
    print(f"  ImgSize: {args.imgsz}")
    print(f"  Device:  {device}")
    print()

    model = YOLO(args.weights)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        rect=True,              # Preserve 16:9, prevents letterbox re-introduction

        # Geometric
        fliplr=0.5,             # Horizontal flip - Sanbo often approaches from the left
        flipud=0.0,             # Off - fixed overhead camera
        degrees=0.0,            # Off - avoid coordinate drift on a fixed setup

        # Color - only value/brightness matters for IR grayscale footage
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,

        # Advanced
        mosaic=1.0,             # Still useful for small-object robustness
        copy_paste=0.0,         # Disabled for the current v14-era class balance
        mixup=0.0,              # Off - destroys small kibble detail

        # Training config
        batch=args.batch,
        epochs=args.epochs,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,

        # Output
        project=args.project,
        name=args.name,
        device=device,
        verbose=True,
        plots=True,
    )

    print(f"\nTraining complete. Results saved to {args.project}/{args.name}")


if __name__ == "__main__":
    main()
