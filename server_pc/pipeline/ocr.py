import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from difflib import SequenceMatcher

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
    # Ưu tiên độ đúng hơn tốc độ.
    config['predictor']['beamsearch'] = True

    # Trỏ đường dẫn tới file .pth bạn vừa huấn luyện xong
    # custom_weight_path = str(Path(__file__).resolve().parent.parent / "weights" / "ten_file_vietocr_cua_ban.pth")
    # config['weights'] = custom_weight_path
    
    # Predictor sẽ tự động tải weights (.pth) về nếu chưa có
    return Predictor(config)


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return (inter_area / union) if union > 0 else 0.0


def _non_max_suppress_boxes(boxes, iou_thresh=0.6):
    if not boxes:
        return []
    # Ưu tiên box có score cao, sau đó giữ thứ tự đọc top->bottom.
    candidates = sorted(
        boxes,
        key=lambda b: float(b.get("score", 0.0)),
        reverse=True,
    )
    kept = []
    for b in candidates:
        bbox = [int(v) for v in b["bbox"]]
        drop = False
        for k in kept:
            kbbox = [int(v) for v in k["bbox"]]
            if _iou(bbox, kbbox) >= iou_thresh:
                drop = True
                break
        if not drop:
            kept.append(b)
    return sorted(kept, key=lambda b: (b["bbox"][1], b["bbox"][0]))


def _merge_boxes_into_lines(boxes, y_center_thresh=22):
    """
    Gộp các box gần cùng một dòng để OCR theo dòng thay vì mảnh nhỏ.
    Ưu tiên độ đúng hơn tốc độ.
    """
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    lines = []
    for b in sorted_boxes:
        x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
        cy = (y1 + y2) / 2.0
        placed = False
        for ln in lines:
            if abs(cy - ln["cy"]) <= y_center_thresh:
                ln["members"].append(b)
                ln["cy"] = (ln["cy"] * (len(ln["members"]) - 1) + cy) / len(ln["members"])
                placed = True
                break
        if not placed:
            lines.append({"cy": cy, "members": [b]})

    merged = []
    for ln in lines:
        members = sorted(ln["members"], key=lambda m: m["bbox"][0])
        xs1 = [int(m["bbox"][0]) for m in members]
        ys1 = [int(m["bbox"][1]) for m in members]
        xs2 = [int(m["bbox"][2]) for m in members]
        ys2 = [int(m["bbox"][3]) for m in members]
        labels = [str(m.get("label", "plain text")).strip().lower() for m in members]
        label = "title" if "title" in labels else "plain text"
        score = max(float(m.get("score", 0.0)) for m in members)
        merged.append({
            "bbox": [min(xs1), min(ys1), max(xs2), max(ys2)],
            "label": label,
            "score": score,
        })
    return sorted(merged, key=lambda b: (b["bbox"][1], b["bbox"][0]))


def _enhance_crop_for_ocr(crop):
    """
    Tăng độ rõ chữ cho từng crop trước khi đưa vào VietOCR.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    h, w = gray.shape[:2]
    # Upscale giúp model đọc nét chữ nhỏ tốt hơn.
    scale = 2.0 if max(h, w) < 1200 else 1.5
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return Image.fromarray(gray)


def _prepare_ocr_variants(crop):
    """
    Tạo nhiều biến thể crop để OCR rồi chọn kết quả tốt nhất theo confidence.
    """
    variants = []

    # Variant 1: grayscale + denoise + CLAHE (mặc định tốt cho văn bản mờ)
    variants.append(_enhance_crop_for_ocr(crop))

    # Variant 2: adaptive threshold để làm nét ký tự mảnh
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    variants.append(Image.fromarray(th))

    # Variant 3: giữ màu gốc, upscale để hạn chế mất thông tin dấu
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(Image.fromarray(rgb))

    return variants


def _normalize_text(text):
    text = " ".join(str(text).strip().split())
    return text.lower()


def _is_duplicate_text(text, existing_texts, sim_threshold=0.97):
    ntext = _normalize_text(text)
    if not ntext:
        return True
    for ex in existing_texts:
        nex = _normalize_text(ex)
        if not nex:
            continue
        if ntext == nex:
            return True
        if SequenceMatcher(None, ntext, nex).ratio() >= sim_threshold:
            return True
    return False


def _text_quality_bonus(text):
    txt = str(text).strip()
    if not txt:
        return -1.0
    length = len(txt)
    alpha = sum(ch.isalpha() for ch in txt)
    digits = sum(ch.isdigit() for ch in txt)
    noise = sum(ch in "|_~`" for ch in txt)
    alpha_ratio = alpha / max(1, length)
    digit_ratio = digits / max(1, length)
    noise_penalty = noise / max(1, length)

    # Thưởng nhẹ cho câu có chữ cái; phạt chuỗi nhiễu.
    bonus = 0.25 * alpha_ratio
    if digit_ratio > 0.7:
        bonus -= 0.05
    bonus -= 0.35 * noise_penalty
    return bonus


def _predict_best_text(ocr, crop):
    best_text = ""
    best_score = -1e9
    for pil_img in _prepare_ocr_variants(crop):
        try:
            pred_text, prob = ocr.predict(pil_img, return_prob=True)
        except TypeError:
            # fallback nếu predictor không hỗ trợ return_prob
            pred_text = ocr.predict(pil_img)
            prob = None
        pred_text = " ".join(str(pred_text).split()).strip()
        if not pred_text:
            continue
        base_score = float(prob) if prob is not None else 0.0
        score = base_score + _text_quality_bonus(pred_text)
        if score > best_score:
            best_score = score
            best_text = pred_text
    return best_text


def _ocr_bottom_fallback(img, ocr, existing_lines):
    """
    OCR bổ sung cho phần cuối trang (thường là đoạn thơ/caption dễ bị bỏ sót).
    """
    h = img.shape[0]
    y1 = int(h * 0.68)
    strip = img[y1:h, :]
    if strip.size == 0:
        return []
    text = _predict_best_text(ocr, strip)
    text = " ".join(str(text).split()).strip()
    if not text:
        return []
    if _is_duplicate_text(text, existing_lines, sim_threshold=0.95):
        return []
    # Chỉ lấy nếu đủ dài để giảm nhiễu.
    if len(text) < 25:
        return []
    return [text]

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

        candidate_boxes = []
        for box in entry["boxes"]:
            label = str(box.get("label", "")).strip().lower()
            if label not in TEXT_LABELS:
                continue
            candidate_boxes.append(box)

        boxes = _non_max_suppress_boxes(candidate_boxes, iou_thresh=0.6)
        boxes = _merge_boxes_into_lines(boxes, y_center_thresh=24)

        text_lines = []           # dùng cho run_pipeline.py (chỉ cần text)
        content_with_labels = []  # dùng cho _build_docx (cần cả label)

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            # Nới nhẹ biên để không cắt mất dấu tiếng Việt.
            pad = 3
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img.shape[1], x2 + pad)
            y2 = min(img.shape[0], y2 + pad)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            label = str(box.get("label", "")).strip().lower()
            try:
                pred_text = _predict_best_text(ocr, crop)
                if _is_duplicate_text(pred_text, text_lines):
                    continue
                text_lines.append(pred_text)
                content_with_labels.append({"label": label, "text": pred_text})
            except Exception as e:
                print(f"[ERR] Lỗi nhận diện box: {e}")

        # Nếu nội dung gần đáy trang còn thiếu, OCR bổ sung vùng đáy.
        if boxes:
            max_y2 = max(int(b["bbox"][3]) for b in boxes)
            if max_y2 < int(img.shape[0] * 0.9):
                extra_lines = _ocr_bottom_fallback(img, ocr, text_lines)
                for t in extra_lines:
                    text_lines.append(t)
                    content_with_labels.append({"label": "plain text", "text": t})
                    print(f"[OCR] Bổ sung dòng cuối trang cho: {entry['image']}")

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