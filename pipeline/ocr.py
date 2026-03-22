from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TEXT_LABELS = {"text", "paragraph", "title", "caption", "header", "footer"}


def _load_ocr():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ImportError(
            "Thiếu paddleocr. Cài đặt: pip install paddleocr==2.7.0.3"
        ) from exc
    return PaddleOCR(lang="vi", use_angle_cls=True, det=True, rec=True, show_log=False)


def _flatten_paddle_result(pred):
    lines = []
    for page in pred or []:
        for line in page:
            text = line[1][0]
            conf = float(line[1][1])
            lines.append({"text": text, "score": conf})
    return lines


def run_ocr(
    input_dir: Path,
    layout_results: List[Dict] | None,
    output_dir: Path,
) -> List[Dict]:
    """
    Chạy OCR tiếng Việt. Nếu có layout_results thì chỉ OCR trên box text.

    - input_dir: thư mục ảnh gốc (hoặc đã preprocess).
    - layout_results: kết quả từ run_layout (có bbox).
    - output_dir: lưu json/text.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr = _load_ocr()
    results: List[Dict] = []

    # map filename -> boxes
    layout_map = {
        item["image"]: item.get("boxes", []) for item in (layout_results or [])
    }

    files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    )
    if not files:
        print(f"[WARN] Không có ảnh OCR tại {input_dir}")
        return []

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[SKIP] Không đọc được ảnh: {p}")
            continue

        boxes = layout_map.get(p.name, [])
        texts = []
        if boxes:
            for box in boxes:
                label = str(box.get("label", "")).lower()
                # Nếu label không phải dạng text thì bỏ qua; label rỗng vẫn OCR
                if label and label not in TEXT_LABELS:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
                x1_c, y1_c = max(x1, 0), max(y1, 0)
                x2_c = min(x2, img.shape[1])
                y2_c = min(y2, img.shape[0])
                crop = img[y1_c:y2_c, x1_c:x2_c]
                if crop.size == 0:
                    continue
                pred = ocr.ocr(crop, cls=True)
                lines = _flatten_paddle_result(pred)
                texts.extend(lines)
        else:
            pred = ocr.ocr(img, cls=True)
            texts = _flatten_paddle_result(pred)

        results.append({"image": p.name, "texts": texts})
        txt_path = output_dir / f"{p.stem}_ocr.txt"
        txt_path.write_text(
            "\n".join([t["text"] for t in texts]),
            encoding="utf-8",
        )
        print(f"[OCR] {p.name}: {len(texts)} dòng, lưu {txt_path.name}")

    summary = output_dir / "ocr.json"
    summary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] Đã lưu OCR JSON: {summary}")
    return results


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    inp = base / "data" / "preprocessed"
    out = base / "data" / "ocr"
    run_ocr(inp, None, out)
