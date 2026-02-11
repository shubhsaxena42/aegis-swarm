# 1. Install & Import
!pip install ultralytics --quiet

import os
from pathlib import Path
from ultralytics import YOLO

# 2. EXACT PATHS (Based on your second screenshot)
INPUT_BASE = Path("/kaggle/input/visdrone-dataset/VisDrone_Dataset")
WORKING_DIR = Path("/kaggle/working/VisDrone")

# 3. LINKING LOGIC
def prepare_visdrone_direct():
    split_map = {
        'train': 'VisDrone2019-DET-train',
        'val':   'VisDrone2019-DET-val',
        'test':  'VisDrone2019-DET-test-dev'
    }

    for split, folder_name in split_map.items():
        src_root = INPUT_BASE / folder_name
        if not src_root.exists():
            print(f"⚠️ Warning: {folder_name} not found.")
            continue
            
        print(f"📦 Linking {split}...")
        
        # Create destination directories in /working/
        (WORKING_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (WORKING_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

        # Source folders from your screenshot
        src_img_dir = src_root / "images"
        src_lab_dir = src_root / "labels"

        # Link Images
        img_count = 0
        for img_file in src_img_dir.glob("*.jpg"):
            dst_img = WORKING_DIR / split / "images" / img_file.name
            if not dst_img.exists():
                os.symlink(img_file, dst_img)
            img_count += 1
            
        # Link Labels (Since they are already in YOLO format!)
        lab_count = 0
        for lab_file in src_lab_dir.glob("*.txt"):
            dst_lab = WORKING_DIR / split / "labels" / lab_file.name
            if not dst_lab.exists():
                os.symlink(lab_file, dst_lab)
            lab_count += 1
            
        print(f"   ✅ Linked {img_count} images and {lab_count} labels for {split}")

if __name__ == "__main__":
    prepare_visdrone_direct()
    
    # 4. CREATE YAML
    yaml_content = f"""
path: {WORKING_DIR}
train: train/images
val: val/images
test: test/images
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    with open('/kaggle/working/VisDrone.yaml', 'w') as f:
        f.write(yaml_content)
    print("\n✅ YAML created at /kaggle/working/VisDrone.yaml")


# --- 4. GPU Detection & Training ---

import torch
from ultralytics import YOLO
import os
# The rest of your code remains the same...
num_gpus = torch.cuda.device_count()
print(f"🔥 Detected {num_gpus} GPU(s)")
print("\n" + "=" * 50)
print("🚀 PHASE 3: Training")
print("=" * 50)

num_gpus = torch.cuda.device_count()
print(f"🔥 Detected {num_gpus} GPU(s)")

# Clear memory before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load Model
model = YOLO("yolo26s.pt")

# Configure device based on available GPUs
if num_gpus == 0:
    print("⚠️ No GPU found. Training on CPU (will be slow).")
    device = "cpu"
elif num_gpus == 1:
    print("✅ Using single GPU (device=0)")
    device = 0
else:
    print(f"✅ Using {num_gpus} GPUs with native DDP (device=[0,1])")
    device = [0, 1]

checkpoint_path = "visdrone_yolo26s/kaggle_run/weights/last.pt"

if os.path.exists(checkpoint_path):
    print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
    model = YOLO(checkpoint_path)
    resume = True
else:
    print("🆕 Starting fresh training with yolo26s.pt")
    model = YOLO("yolo26s.pt")
    resume = False
# Train
print("\n🏋️ Starting YOLO26s Training on VisDrone...")
results = model.train(
    data="./VisDrone.yaml",
    imgsz=640,
    epochs=100,
    batch=16 if num_gpus <= 1 else 32,  # Larger batch for multi-GPU
    resume=resume,
    device=device,
    project="visdrone_yolo26s",
    name="kaggle_run",
    exist_ok=True,
    patience=20,
    save_period=10,  # Save checkpoint every 10 epochs
    verbose=True
)

print("\n" + "=" * 50)
print("✅ Training Complete!")
print("=" * 50)
print(f"📁 Best model saved to: visdrone_yolo26s/kaggle_run/weights/best.pt")
