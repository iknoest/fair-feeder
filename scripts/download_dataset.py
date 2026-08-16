"""
download_dataset.py — Download Fair Feeder V13 dataset from Roboflow.

Reads the API key from the ROBOFLOW_API_KEY environment variable.

Usage (Colab/Kaggle):
    import os
    os.environ["ROBOFLOW_API_KEY"] = "your_key_here"  # or use Colab secrets
    !python download_dataset.py

Usage (terminal):
    export ROBOFLOW_API_KEY="your_key_here"
    python download_dataset.py
"""

import os
import sys

from roboflow import Roboflow

WORKSPACE = "test-7vyqo"
PROJECT = "ir-kibble"
VERSION = 13
FORMAT = "yolov8"


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY environment variable is not set.")
        print()
        print("Set it before running this script:")
        print('  export ROBOFLOW_API_KEY="your_key_here"')
        sys.exit(1)

    print(f"Downloading {PROJECT} v{VERSION} ({FORMAT} format)...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download(FORMAT)

    print(f"\nDataset downloaded to: {dataset.location}")
    print()
    print("Next steps:")
    print(f"  1. python polygon_to_bbox.py --dataset {dataset.location}")
    print(f"  2. python verify_labels.py --dataset {dataset.location} --save check.jpg")
    print(f"  3. python train.py --data {dataset.location}/data.yaml")


if __name__ == "__main__":
    main()
