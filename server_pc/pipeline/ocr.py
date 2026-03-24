import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from difflib import SequenceMatcher

# 1. XỬ LÝ ĐƯỜNG DẪN ĐỂ TÌM VIETOCR-MASTER
base_path = Path(__file__).resolve().parent.parent
vietocr_path = str(base_path / "vietnamese-ocr-master")

if vietocr_path not in sys.path:
    sys.path.insert(0, vietocr_path)

# 2. IMPORT TỪ BẢN MASTER
try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    print("--- [INFO] Đã kết nối thành công VietOCR Master ---")
except ModuleNotFoundError:
    print(f"--- [ERROR] Không tìm thấy VietOCR tại: {vietocr_path} ---")
    sys.path.append(os.path.abspath("vietnamese-ocr-master"))
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg

def load_ocr_model():
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = True
    return Predictor(config)


def _non_max_suppress_boxes(boxes, iou_thresh=0.4, iom_thresh=0.7):
    """Khử nhiễu các hộp bị YOLO vẽ đè lên nhau (Dùng cả IoU và IoM)"""
    if not boxes: return []
    candidates = sorted(boxes, key=lambda b: float(b.get("score", 0.0)), reverse=True)
    kept = []
    for b in candidates:
        ax1, ay1, ax2, ay2 = [int(v) for v in b["bbox"]]
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        drop = False
        for k in kept:
            bx1, by1, bx2, by2 = [int(v) for v in k["bbox"]]
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            
            inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
            inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            
            union = area_a + area_b - inter_area
            iou = inter_area / union if union > 0 else 0.0
            # IoM giúp phát hiện hộp nhỏ nằm lọt thỏm trong hộp to
            iom = inter_area / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
            
            if iou >= iou_thresh or iom >= iom_thresh:
                drop = True
                break
        if not drop:
            kept.append(b)
    return sorted(kept, key=lambda b: (b["bbox"][1], b["bbox"][0]))


def _merge_boxes_into_lines(boxes, y_center_thresh=None):
    if not boxes: return []

    # --- DYNAMIC THRESHOLDING ---
    # Tính chiều cao trung bình của tất cả các box
    total_height = sum([b["bbox"][3] - b["bbox"][1] for b in boxes])
    average_height = total_height / len(boxes)
    
    # Đặt ngưỡng bằng 50% chiều cao trung bình
    y_center_thresh = average_height * 0.5

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
        xs1, ys1 = [int(m["bbox"][0]) for m in members], [int(m["bbox"][1]) for m in members]
        xs2, ys2 = [int(m["bbox"][2]) for m in members], [int(m["bbox"][3]) for m in members]
        labels = [str(m.get("label", "plain text")).strip().lower() for m in members]
        label = "title" if "title" in labels else "plain text"
        score = max(float(m.get("score", 0.0)) for m in members)
        merged.append({
            "bbox": [min(xs1), min(ys1), max(xs2), max(ys2)],
            "label": label, "score": score,
        })
    return sorted(merged, key=lambda b: (b["bbox"][1], b["bbox"][0]))

def _enhance_crop_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    h, w = gray.shape[:2]
    scale = 2.0 if max(h, w) < 1200 else 1.5
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return Image.fromarray(gray)


def _prepare_ocr_variants(crop):
    variants = [_enhance_crop_for_ocr(crop)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    variants.append(Image.fromarray(th))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(Image.fromarray(rgb))
    return variants


def _normalize_text(text):
    return " ".join(str(text).strip().split()).lower()


def _is_duplicate_text(text, existing_texts, sim_threshold=0.97):
    ntext = _normalize_text(text)
    if not ntext: return True
    for ex in existing_texts:
        nex = _normalize_text(ex)
        if not nex: continue
        if ntext == nex or SequenceMatcher(None, ntext, nex).ratio() >= sim_threshold:
            return True
    return False


def _text_quality_bonus(text):
    txt = str(text).strip()
    if not txt: return -1.0
    length = len(txt)
    alpha, digits, noise = sum(ch.isalpha() for ch in txt), sum(ch.isdigit() for ch in txt), sum(ch in "|_~`" for ch in txt)
    bonus = 0.25 * (alpha / max(1, length))
    if (digits / max(1, length)) > 0.7: bonus -= 0.05
    bonus -= 0.35 * (noise / max(1, length))
    return bonus

def _predict_best_text(ocr, crop):
    # Chỉ dùng 1 ảnh RGB duy nhất
    rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    try:
        pred_text = ocr.predict(pil_img)
        return " ".join(str(pred_text).split()).strip()
    except Exception:
        return ""

def _ocr_bottom_fallback(img, ocr, existing_lines):
    h = img.shape[0]
    strip = img[int(h * 0.68):h, :]
    if strip.size == 0: return []
    text = " ".join(str(_predict_best_text(ocr, strip)).split()).strip()
    if not text or len(text) < 25 or _is_duplicate_text(text, existing_lines, sim_threshold=0.95):
        return []
    return [text]

def _split_block_into_lines(crop_img):
    h_img, w_img = crop_img.shape[:2]
    if h_img < 50: return [{"image": crop_img, "x_local": 0, "y_local": 0}]

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # SỬA KERNEL: Giảm từ 50 xuống 25 để không làm dính đáp án B và C vào nhau
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # SỬA BỘ LỌC: Hạ xuống > 3 để nhặt lại số "185" nhỏ bé
        if h > 3 and w > 3:
            line_boxes.append((x, y, w, h))
            
    if not line_boxes: return [{"image": crop_img, "x_local": 0, "y_local": 0}]
    line_boxes = sorted(line_boxes, key=lambda b: b[1])
    
    avg_h = sum([b[3] for b in line_boxes]) / len(line_boxes)
    dynamic_thresh = avg_h * 0.45

    groups = []
    for box in line_boxes:
        cy = box[1] + box[3]/2.0
        placed = False
        for group in groups:
            if abs(group['cy'] - cy) < dynamic_thresh: 
                group['boxes'].append(box)
                group['cy'] = (group['cy'] * (len(group['boxes'])-1) + cy) / len(group['boxes'])
                placed = True
                break
        if not placed: groups.append({'cy': cy, 'boxes': [box]})
        
    final_boxes = []
    for group in groups:
        final_boxes.extend(sorted(group['boxes'], key=lambda b: b[0]))

    lines = []
    # Duyệt qua final_boxes (đã sắp xếp) thay vì line_boxes (chưa sắp xếp)
    for (x, y, w, h) in final_boxes: 
        pad_x, pad_y = 5, 8 
        y1, y2 = max(0, y - pad_y), min(h_img, y + h + pad_y)
        x1, x2 = max(0, x - pad_x), min(w_img, x + w + pad_x)
        lines.append({"image": crop_img[y1:y2, x1:x2], "x_local": x1, "y_local": y1})
    return lines

def run_ocr(pre_dir: Path, layout_results: list, output_dir: Path, predictor=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr = predictor if predictor is not None else load_ocr_model()
    TEXT_LABELS = {"text", "title", "plain text", "plaintext", "paragraph", "header", "footer", "caption", "list", "footnote", "formula"}
    all_results = []

    for entry in layout_results:
        img_path = pre_dir / entry["image"]
        img = cv2.imread(str(img_path))
        if img is None: continue

        candidate_boxes = [box for box in entry["boxes"] if str(box.get("label", "")).strip().lower() in TEXT_LABELS]
        boxes = _non_max_suppress_boxes(candidate_boxes, iou_thresh=0.4, iom_thresh=0.7)
        boxes = _merge_boxes_into_lines(boxes)

        text_lines, content_with_labels = [], []

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            # --- PADDING (MỞ RỘNG VIỀN) ---
            # Thêm 5 pixel "không gian thở" cho VietOCR
            pad_x = 18
            pad_y = 8
            
            # Dùng max/min để đảm bảo padding không bị tràn ra ngoài viền bức ảnh
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img.shape[1], x2 + pad_x)
            y2 = min(img.shape[0], y2 + pad_y)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            label = str(box.get("label", "")).strip().lower()
            line_data_list = _split_block_into_lines(crop)
            
            for l_data in line_data_list:
                l_crop = l_data["image"]
                h_c, w_c = l_crop.shape[:2]
                
                # BỘ LỌC 1: Bỏ qua các vệt nhiễu siêu mỏng (Rất quan trọng để chống ảo giác)
                if w_c < 10 or h_c < 10: continue
                
                try:
                    pred_text = _predict_best_text(ocr, l_crop)
                    if not pred_text or _is_duplicate_text(pred_text, text_lines): continue
                        
                    # BỘ LỌC 2 CHỐNG ẢO GIÁC: Chiều rộng vật lý quá hẹp nhưng AI bịa ra từ quá dài
                    if len(pred_text) > 3 and (w_c / len(pred_text)) < 5.0:
                        print(f"[WARN] Bỏ qua ảo giác '{pred_text}' (width={w_c})")
                        continue
                    
                    text_lines.append(pred_text)
                    content_with_labels.append({
                        "label": label, "text": pred_text,
                        "x": x1 + l_data["x_local"], "y": y1 + l_data["y_local"],
                        "w": w_c, "h": h_c # TRUYỀN CHIỀU RỘNG THẬT SANG BƯỚC WORD
                    })
                except Exception as e:
                    pass

        if boxes:
            max_y2 = max(int(b["bbox"][3]) for b in boxes)
            if max_y2 < int(img.shape[0] * 0.9):
                for t in _ocr_bottom_fallback(img, ocr, text_lines):
                    text_lines.append(t)
                    content_with_labels.append({"label": "plain text", "text": t})

        if not text_lines:
            try:
                pred_text = ocr.predict(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).strip()
                if pred_text:
                    text_lines.append(pred_text)
                    content_with_labels.append({"label": "plain text", "text": pred_text})
            except Exception: pass

        txt_path = output_dir / f"{img_path.stem}.txt"
        txt_path.write_text("\n".join(text_lines), encoding="utf-8")
        all_results.append({"image": entry["image"], "content": text_lines, "content_with_labels": content_with_labels})

    return all_results

def run_ocr_mem(img_array, boxes_list, predictor):
    """Chạy OCR trực tiếp trên ảnh RAM và danh sách bounding box"""
    TEXT_LABELS = {"text", "title", "plain text", "plaintext", "paragraph", "header", "footer", "caption", "list", "footnote", "formula"}
    
    candidate_boxes = [box for box in boxes_list if str(box.get("label", "")).strip().lower() in TEXT_LABELS]
    boxes = _non_max_suppress_boxes(candidate_boxes, iou_thresh=0.4, iom_thresh=0.7)
    boxes = _merge_boxes_into_lines(boxes)

    text_lines, content_with_labels = [], []

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
        # --- PADDING (MỞ RỘNG VIỀN) ---
        # Thêm 5 pixel "không gian thở" cho VietOCR
        pad_x = 18
        pad_y = 8
        
        # Dùng max/min để đảm bảo padding không bị tràn ra ngoài viền bức ảnh
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_array.shape[1], x2 + pad_x)
        y2 = min(img_array.shape[0], y2 + pad_y)
        crop = img_array[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        label = str(box.get("label", "")).strip().lower()
        line_data_list = _split_block_into_lines(crop)
        
        for l_data in line_data_list:
            l_crop = l_data["image"]
            h_c, w_c = l_crop.shape[:2]
            
            if w_c < 10 or h_c < 10: continue
            
            try:
                pred_text = _predict_best_text(predictor, l_crop)
                if not pred_text or _is_duplicate_text(pred_text, text_lines): continue
                    
                if len(pred_text) > 3 and (w_c / len(pred_text)) < 5.0:
                    continue
                
                text_lines.append(pred_text)
                content_with_labels.append({
                    "label": label, "text": pred_text,
                    "x": x1 + l_data["x_local"], "y": y1 + l_data["y_local"],
                    "w": w_c, "h": h_c 
                })
            except Exception:
                pass

    if not text_lines:
        try:
            pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            pred_text = predictor.predict(pil_img).strip()
            if pred_text:
                text_lines.append(pred_text)
                content_with_labels.append({"label": "plain text", "text": pred_text})
        except Exception: pass

    return {"content": text_lines, "content_with_labels": content_with_labels}