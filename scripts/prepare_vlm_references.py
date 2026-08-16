import os
import sys
import shutil
import glob
import random
import argparse
from roboflow import Roboflow
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not found in environment")
        sys.exit(1)

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("test-7vyqo").project("ir-kibble")
        version = project.version(13)
        
        import tempfile
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)
        dataset = version.download("yolov8")
        dataset_path = dataset.location
    except Exception as e:
        print(f"[ERROR] Roboflow download failed: {type(e).__name__} - {str(e)[:100]}")
        sys.exit(1)

    import yaml
    yaml_path = os.path.join(dataset_path, "data.yaml")
    dan_cls_id, sanbo_cls_id = None, None
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            names = data.get("names", [])
            if isinstance(names, dict):
                # YOLOv5 format dict: {0: 'Dan', 1: 'Sanbo'}
                for k, v in names.items():
                    if v == "Dan": dan_cls_id = k
                    elif v == "Sanbo": sanbo_cls_id = k
            elif isinstance(names, list):
                # YOLOv8 format list: ['Bowl', 'Dan', ...]
                try:
                    dan_cls_id = names.index("Dan")
                    sanbo_cls_id = names.index("Sanbo")
                except ValueError:
                    pass
                    
    if dan_cls_id is None or sanbo_cls_id is None:
        print("[ERROR] Could not resolve Dan and Sanbo class IDs from data.yaml")
        sys.exit(1)
        
    print(f"[INFO] Resolved class IDs: Dan={dan_cls_id}, Sanbo={sanbo_cls_id}")

    dan_images = []
    sanbo_images = []

    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        images_dir = os.path.join(dataset_path, split, 'images')
        if not os.path.exists(labels_dir): continue
        
        for label_file in sorted(glob.glob(os.path.join(labels_dir, "*.txt"))):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            has_dan, has_sanbo = False, False
            is_good_dan, is_good_sanbo = False, False
            
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                cls_id, w, h = int(parts[0]), float(parts[3]), float(parts[4])
                area = w * h
                
                if cls_id == dan_cls_id:
                    has_dan = True
                    if area > 0.05: is_good_dan = True
                elif cls_id == sanbo_cls_id:
                    has_sanbo = True
                    if area > 0.05: is_good_sanbo = True
            
            if has_dan and has_sanbo: continue
                
            base = os.path.splitext(os.path.basename(label_file))[0]
            img_file = os.path.join(images_dir, base + ".jpg")
            if not os.path.exists(img_file):
                img_file = os.path.join(images_dir, base + ".png")
                
            if not os.path.exists(img_file): continue
            
            if is_good_dan and not has_sanbo:
                dan_images.append(img_file)
            elif is_good_sanbo and not has_dan:
                sanbo_images.append(img_file)

    random.seed(42)
    random.shuffle(dan_images)
    random.shuffle(sanbo_images)

    dest_dan = os.path.join(args.out_dir, "dan")
    dest_sanbo = os.path.join(args.out_dir, "sanbo")
    os.makedirs(dest_dan, exist_ok=True)
    os.makedirs(dest_sanbo, exist_ok=True)

    for i, img in enumerate(dan_images[:4]):
        shutil.copy(img, os.path.join(dest_dan, f"dan_ref_{i}.jpg"))
    for i, img in enumerate(sanbo_images[:4]):
        shutil.copy(img, os.path.join(dest_sanbo, f"sanbo_ref_{i}.jpg"))

    print(f"Reference gallery built safely at {args.out_dir}")
    shutil.rmtree(tmpdir)

if __name__ == "__main__":
    main()
