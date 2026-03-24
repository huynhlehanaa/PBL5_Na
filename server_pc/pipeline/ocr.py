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

def load_ocr_model():
    # Load cấu hình model vgg_transformer (phổ biến nhất của VietOCR)
    config = Cfg.load_config_from_name('vgg_transformer')
    
    # Quan trọng cho Raspberry Pi: dùng CPU để tiết kiệm RAM
    config['device'] = 'cpu'

    # Trỏ đường dẫn tới file .pth bạn vừa huấn luyện xong
    # custom_weight_path = str(Path(__file__).resolve().parent.parent / "weights" / "ten_file_vietocr_cua_ban.pth")
    # config['weights'] = custom_weight_path
    
    # Predictor sẽ tự động tải weights (.pth) về nếu chưa có
    return Predictor(config)

def run_ocr(pre_dir: Path, layout_results: list, output_dir: Path, predictor=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr = predictor if predictor is not None else load_ocr_model()

    # Gộp nhiều biến thể nhãn text để đỡ phụ thuộc đúng 1 tên class.
    TEXT_LABELS = {
        "text", "title", "plain text", "plaintext", "paragraph",
        "header", "footer", "caption", "list", "footnote", "formula"
    }
    all_results = []

    for entry in layout_results:
        img_path = pre_dir / entry["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Không tìm thấy ảnh: {entry['image']}")
            continue

        boxes = sorted(entry["boxes"], key=lambda b: b["bbox"][1])

        text_lines = []           # dùng cho run_pipeline.py (chỉ cần text)
        content_with_labels = []  # dùng cho _build_docx (cần cả label)

        for box in boxes:
            label = str(box.get("label", "")).strip().lower()
            if label not in TEXT_LABELS:
                continue
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            try:
                pred_text = ocr.predict(pil_img)
                text_lines.append(pred_text)
                content_with_labels.append({"label": label, "text": pred_text})
            except Exception as e:
                print(f"[ERR] Lỗi nhận diện box: {e}")

        # Fallback: nếu layout không trả về vùng text nào, OCR toàn trang để tránh ra file rỗng.
        if not text_lines:
            try:
                full_page = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                pred_text = ocr.predict(full_page)
                pred_text = pred_text.strip()
                if pred_text:
                    text_lines.append(pred_text)
                    content_with_labels.append({"label": "plain text", "text": pred_text})
                    print(f"[OCR] Fallback toàn trang cho: {entry['image']}")
            except Exception as e:
                print(f"[ERR] Lỗi OCR fallback toàn trang: {e}")

        txt_path = output_dir / f"{img_path.stem}.txt"
        txt_path.write_text("\n".join(text_lines), encoding="utf-8")

        all_results.append({
            "image": entry["image"],
            "content": text_lines,                  # app_client.py preview dùng cái này
            "content_with_labels": content_with_labels,  # _build_docx dùng cái này
        })
        print(f"[OCR] Hoàn thành: {entry['image']} ({len(text_lines)} dòng)")

    return all_results