import sys
import os
import cv2
import torch
from pathlib import Path

# 1. ÉP PYTORCH LOAD MODEL TÙY CHỈNH
original_load = torch.load
def custom_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = custom_load

# 2. TRỎ ĐƯỜNG DẪN VÀO REPO ĐÃ TẢI
base_path = Path(__file__).resolve().parent.parent
doclayout_repo_path = str(base_path / "DocLayout-YOLO")

if doclayout_repo_path not in sys.path:
    sys.path.insert(0, doclayout_repo_path)

# 3. IMPORT ĐÚNG CLASS YOLOv10 TỪ REPO TÁC GIẢ
try:
    # BẮT BUỘC dùng YOLOv10 thay vì YOLO để tương thích với output dạng dict
    from doclayout_yolo import YOLOv10
    print("--- [INFO] Đã kết nối class YOLOv10 từ package doclayout_yolo ---")
except ImportError:
    try:
        from ultralytics import YOLOv10
        print("--- [INFO] Đã kết nối class YOLOv10 (ưu tiên local repo) ---")
    except ImportError as e:
        print(f"--- [ERROR] Không import được mô hình YOLOv10: {e} ---")

def load_layout_model():
    model_name = "doclayout_yolo_docstructbench_imgsz1024.pt"
    full_model_path = str(Path(__file__).resolve().parent.parent / "weights" / model_name)
    return YOLOv10(full_model_path)

def run_layout(input_dir: Path, output_dir: Path, model=None):  # thêm model=None
    if model is None:
        model = load_layout_model()  # fallback khi gọi từ run_pipeline.py
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    layout_results = []
    for p in files:
        results = model.predict(str(p), imgsz=1024, conf=0.25)
        res = results[0]
        items = []
        if res.boxes is not None:
            for box in res.boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls[0])
                label = res.names[cls_id]
                items.append({"bbox": bbox, "label": label})
        annotated = res.plot()
        cv2.imwrite(str(output_dir / f"{p.stem}_layout.jpg"), annotated)
        layout_results.append({"image": p.name, "boxes": items})
        print(f"[LAYOUT] Xong: {p.name}")
    return layout_results