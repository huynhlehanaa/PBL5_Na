from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

import cv2
import numpy as np

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TEXT_LABELS = None           # đặt set nhãn nếu muốn lọc box; None = OCR mọi box
CROP_PAD = 10                # padding nới rộng bbox
UPSCALE = 1.8                # phóng to trước OCR


def _load_ocr():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ImportError("Thiếu paddleocr. Cài: pip install paddleocr==2.7.0.3") from exc
    return PaddleOCR(lang="en", use_angle_cls=True, det=True, rec=True, show_log=False)


def _preprocess(img: np.ndarray) -> np.ndarray:
    if UPSCALE != 1.0:
        img = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = cv2.copyMakeBorder(bin_img, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    return bin_img


def _flatten_paddle_result(pred):
    lines = []
    for page in (pred or []):
        if not page:
            continue
        for line in page:
            text = line[1][0]
            conf = float(line[1][1])
            lines.append({"text": text, "score": conf})
    return lines


def _norm_key(s: str) -> str:
    # chuẩn hóa khóa khử trùng lặp: bỏ khoảng trắng, lowercase
    return "".join(s.lower().split())


def _dedup_append(src: List[Dict], best: Dict[str, Dict]):
    for t in src:
        s = t["text"].strip()
        if not s:
            continue
        key = _norm_key(s)
        if key not in best or t["score"] > best[key]["score"]:
            best[key] = t


def run_ocr(input_dir: Path, layout_results: List[Dict] | None, output_dir: Path) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr = _load_ocr()
    results: List[Dict] = []

    layout_map = {item["image"]: item.get("boxes", []) for item in (layout_results or [])}
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    if not files:
        print(f"[WARN] Không có ảnh OCR tại {input_dir}")
        return []

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[SKIP] Không đọc được ảnh: {p}")
            continue

        boxes = layout_map.get(p.name, [])
        best: Dict[str, Dict] = {}

        # 1) OCR toàn ảnh
        full_pred = ocr.ocr(_preprocess(img), cls=True, det=True, rec=True)
        _dedup_append(_flatten_paddle_result(full_pred), best)

        # 2) OCR từng box (nếu có)
        for box in boxes:
            label = str(box.get("label", "")).lower()
            if TEXT_LABELS and label and label not in TEXT_LABELS:
                continue
            x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
            x1 = max(x1 - CROP_PAD, 0)
            y1 = max(y1 - CROP_PAD, 0)
            x2 = min(x2 + CROP_PAD, img.shape[1])
            y2 = min(y2 + CROP_PAD, img.shape[0])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pred = ocr.ocr(_preprocess(crop), cls=True, det=True, rec=True)
            _dedup_append(_flatten_paddle_result(pred), best)

        # Xuất kết quả (sắp xếp theo score giảm dần để ổn định)
        texts = sorted(best.values(), key=lambda x: -x["score"])
        results.append({"image": p.name, "texts": texts})
        txt_path = output_dir / f"{p.stem}_ocr.txt"
        txt_path.write_text("\n".join([t["text"] for t in texts]), encoding="utf-8")
        print(f"[OCR] {p.name}: {len(texts)} dòng, lưu {txt_path.name}")

    summary = output_dir / "ocr.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Đã lưu OCR JSON: {summary}")
    return results


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    inp = base / "data" / "preprocessed"
    out = base / "data" / "ocr"
    run_ocr(inp, None, out)