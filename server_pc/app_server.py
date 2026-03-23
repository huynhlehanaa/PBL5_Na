from flask import Flask, request, jsonify
import base64
from pathlib import Path

from pipeline.layout import run_layout
from pipeline.preprocess import run_preprocess
from pipeline.ocr import run_ocr
from docx import Document

app = Flask(__name__)

# Định nghĩa đường dẫn tuyệt đối để không phụ thuộc vào thư mục chạy lệnh
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "data" / "input"
PREPROCESS_FOLDER = BASE_DIR / "data" / "preprocessed"
LAYOUT_FOLDER = BASE_DIR / "data" / "layout"
OUTPUT_FOLDER = BASE_DIR / "data" / "output"

for folder in [UPLOAD_FOLDER, PREPROCESS_FOLDER, LAYOUT_FOLDER, OUTPUT_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)


def _build_docx(ocr_results):
    """Tạo file Word từ kết quả OCR và trả về đường dẫn."""
    doc = Document()
    for res in ocr_results:
        doc.add_heading(f"Kết quả: {res['image']}", level=1)
        for line in res.get("content", []):
            doc.add_paragraph(line)

    doc_path = OUTPUT_FOLDER / "Ket_qua_PBL5.docx"
    doc.save(str(doc_path))
    return doc_path


@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # 1. Nhận ảnh từ Pi gửi lên (ghi đè để tiết kiệm dung lượng)
    file = request.files['image']
    img_path = UPLOAD_FOLDER / "capture.jpg"
    file.save(str(img_path))

    # 2. Tiền xử lý -> Phân tích bố cục -> OCR
    preprocessed_files = run_preprocess(UPLOAD_FOLDER, PREPROCESS_FOLDER)
    if not preprocessed_files:
        return jsonify({"error": "No valid images after preprocessing"}), 400

    layout_results = run_layout(PREPROCESS_FOLDER, LAYOUT_FOLDER)
    ocr_results = run_ocr(PREPROCESS_FOLDER, layout_results, OUTPUT_FOLDER)

    # 3. Xuất file Word và mã hóa base64 để Pi có thể tải về ngay
    doc_path = _build_docx(ocr_results)
    doc_b64 = base64.b64encode(doc_path.read_bytes()).decode("utf-8")

    return jsonify({
        "status": "success",
        "data": {
            "layout": layout_results,
            "ocr": ocr_results,
            "docx_filename": doc_path.name,
            "docx_base64": doc_b64,
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
