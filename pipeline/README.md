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