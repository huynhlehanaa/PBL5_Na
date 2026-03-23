from flask import Flask, request, jsonify
import base64
import uuid
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


def _build_docx(ocr_results, output_dir: Path, job_id: str):
    """Tạo file Word từ kết quả OCR và trả về đường dẫn."""
    doc = Document()
    for res in ocr_results:
        doc.add_heading(f"Kết quả: {res['image']}", level=1)
        for line in res.get("content", []):
            doc.add_paragraph(line)

    doc_filename = f"Ket_qua_{job_id}.docx"
    doc_path = output_dir / doc_filename
    doc.save(str(doc_path))
    return doc_path


@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    job_id = uuid.uuid4().hex
    req_input = UPLOAD_FOLDER / job_id
    req_pre = PREPROCESS_FOLDER / job_id
    req_layout = LAYOUT_FOLDER / job_id
    req_output = OUTPUT_FOLDER / job_id

    for folder in [req_input, req_pre, req_layout, req_output]:
        folder.mkdir(parents=True, exist_ok=True)

    # 1. Nhận ảnh từ Pi gửi lên (ghi đè để tiết kiệm dung lượng)
    file = request.files['image']
    img_path = req_input / "capture.jpg"
    file.save(str(img_path))

    # 2. Tiền xử lý -> Phân tích bố cục -> OCR
    preprocessed_files = run_preprocess(req_input, req_pre)
    if not preprocessed_files:
        return jsonify({"error": "No valid images after preprocessing"}), 400

    layout_results = run_layout(req_pre, req_layout)
    ocr_results = run_ocr(req_pre, layout_results, req_output)

    # 3. Xuất file Word và mã hóa base64 để Pi có thể tải về ngay
    doc_path = _build_docx(ocr_results, req_output, job_id)
    doc_b64 = base64.b64encode(doc_path.read_bytes()).decode("utf-8")

    return jsonify({
        "status": "success",
        "data": {
            "layout": layout_results,
            "ocr": ocr_results,
            "docx_filename": doc_path.name,
            "docx_base64": doc_b64,
            "job_id": job_id,
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
