# Pipeline hướng dẫn nhanh

> Mục tiêu: số hóa tài liệu đa định dạng và xuất ra Microsoft Word. Repo đã có sẵn ba mô-đun: tiền xử lý (imutils), layout (DocLayout-YOLO) và OCR tiếng Việt (PaddleOCR).

## Chuẩn bị môi trường

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
# Phụ thuộc cho pipeline
pip install opencv-python pillow
pip install doclayout-yolo torch          # Layout
pip install paddleocr==2.7.0.3            # OCR tiếng Việt
```

Thư mục dữ liệu:

- `data/input`: đặt ảnh gốc (jpg/png/...).  
- `data/preprocessed`: ảnh sau tiền xử lý.  
- `data/layout`: ảnh annotate + `layouts.json`.  
- `data/ocr`: text `.txt` + `ocr.json`.

## Chạy pipeline đầy đủ

```bash
cd pipeline
python run_pipeline.py
```

Các bước:

1. **Preprocess**: resize + làm sạch ảnh (pipeline/preprocess.py).  
2. **Layout**: DocLayout-YOLO dự đoán bố cục, lưu ảnh annotate & `layouts.json`.  
3. **OCR**: PaddleOCR tiếng Việt; nếu có layout thì chỉ OCR trên box text/paragraph.  

Kết quả text nằm ở `data/ocr/*.txt` (mỗi ảnh một file) và `ocr.json` (tất cả ảnh).

## Xuất ra Word (gợi ý)

Để xuất `.docx`, cài thêm `python-docx` rồi ghép các đoạn text:

```bash
pip install python-docx
python - <<'PY'
from pathlib import Path
import json
from docx import Document

ocr_json = Path("data/ocr/ocr.json")
if not ocr_json.exists():
    raise SystemExit("Chạy OCR trước (python pipeline/run_pipeline.py)")

data = json.loads(ocr_json.read_text(encoding="utf-8"))
doc = Document()
for item in data:
    doc.add_heading(item["image"], level=2)
    for line in item["texts"]:
        doc.add_paragraph(line["text"])
out = Path("data/output/result.docx")
out.parent.mkdir(parents=True, exist_ok=True)
doc.save(out)
print("Saved:", out)
PY
```

Bạn có thể thay đổi logic ghép (theo thứ tự box, phân đoạn, v.v.) tùy nhu cầu.

## Lưu ý khi huấn luyện/thay model

- Đổi `model_path` (đối số `run_layout`) sang checkpoint khác nếu cần.  
- Với PDF, convert sang ảnh trước khi đưa vào `data/input` (ví dụ dùng `pdftoppm` hoặc `pdf2image`).  
- Nếu GPU không khả dụng, script sẽ tự dùng CPU (chậm hơn).  

