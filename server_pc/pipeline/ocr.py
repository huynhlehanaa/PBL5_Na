import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# 1. XỬ LÝ ĐƯỜNG DẪN ĐỂ TÌM VIETOCR-MASTER
# Vì file này nằm trong pipeline/, nên .parent.parent sẽ trỏ ra D:\PBL5_Na
base_path = Path(__file__).resolve().parent.parent
vietocr_path = str(base_path / "vietnamese-ocr-master")

# Thêm vào đầu danh sách tìm kiếm của Python để ưu tiên bản Master
if vietocr_path not in sys.path:
    sys.path.insert(0, vietocr_path)

# 2. IMPORT TỪ BẢN MASTER
try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    print("--- [INFO] Đã kết nối thành công VietOCR Master ---")
except ModuleNotFoundError:
    print(f"--- [ERROR] Không tìm thấy VietOCR tại: {vietocr_path} ---")
    # Nếu vẫn lỗi, thử thêm đường dẫn tuyệt đối trực tiếp làm phương án dự phòng
    sys.path.append(os.path.abspath("vietnamese-ocr-master"))
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg

def _load_ocr():
    # Load cấu hình model vgg_transformer (phổ biến nhất của VietOCR)
    config = Cfg.load_config_from_name('vgg_transformer')
    
    # Quan trọng cho Raspberry Pi: dùng CPU để tiết kiệm RAM
    config['device'] = 'cpu' 
    
    # Predictor sẽ tự động tải weights (.pth) về nếu chưa có
    return Predictor(config)

def run_ocr(pre_dir: Path, layout_results: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr = _load_ocr()
    
    all_results = []
    for entry in layout_results:
        img_path = pre_dir / entry["image"]
        img = cv2.imread(str(img_path))
        if img is None: 
            print(f"[WARN] Không tìm thấy ảnh: {entry['image']}")
            continue
        
        # Sắp xếp các box từ trên xuống dưới (y1) để nội dung văn bản đúng thứ tự
        boxes = sorted(entry["boxes"], key=lambda b: b["bbox"][1])
        
        text_lines = []
        for box in boxes:
            # Cắt ảnh theo tọa độ từ YOLO
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0: continue
            
            # VietOCR yêu cầu định dạng PIL Image và hệ màu RGB
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_crop)
            
            # Nhận diện chữ từ vùng đã cắt
            try:
                pred_text = ocr.predict(pil_img)
                text_lines.append(pred_text)
            except Exception as e:
                print(f"[ERR] Lỗi nhận diện box: {e}")
            
        # Lưu kết quả text cho mỗi ảnh
        txt_path = output_dir / f"{img_path.stem}.txt"
        txt_path.write_text("\n".join(text_lines), encoding="utf-8")
        
        all_results.append({"image": entry["image"], "content": text_lines})
        print(f"[OCR] Hoàn thành: {entry['image']} ({len(text_lines)} dòng)")
    
    return all_results