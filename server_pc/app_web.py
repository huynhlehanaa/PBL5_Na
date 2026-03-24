"""
Chạy: python app_web.py
Truy cập: http://localhost:5000
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
import base64, uuid, shutil, os
from pathlib import Path

from pipeline.layout import run_layout, load_layout_model
from pipeline.preprocess import run_preprocess
from pipeline.ocr import run_ocr, load_ocr_model
from docx import Document

app = Flask(__name__, template_folder="templates")
BASE_DIR = Path(__file__).resolve().parent

# ── Load model 1 lần khi server khởi động ──
print("[INIT] Đang tải model YOLO...")
layout_model = load_layout_model()
print("[INIT] Đang tải model VietOCR...")
ocr_model = load_ocr_model()
print("[INIT] Sẵn sàng!")

# ── Giới hạn ──
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10MB
MAX_DOC_BYTES    = 10 * 1024 * 1024
MAX_DOC_CHARS    = 1_000_000
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# ── Thư mục dữ liệu ──
UPLOAD_FOLDER     = BASE_DIR / "data" / "input"
PREPROCESS_FOLDER = BASE_DIR / "data" / "preprocessed"
LAYOUT_FOLDER     = BASE_DIR / "data" / "layout"
OUTPUT_FOLDER     = BASE_DIR / "data" / "output"

for folder in [UPLOAD_FOLDER, PREPROCESS_FOLDER, LAYOUT_FOLDER, OUTPUT_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════

@app.route('/')
def index():
    """Trả về trang Web UI."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Endpoint để frontend kiểm tra server có online không."""
    return jsonify({"status": "ok"})


@app.route('/process', methods=['POST'])
def process():
    """Nhận ảnh → chạy pipeline → trả về OCR + docx base64."""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    job_id     = uuid.uuid4().hex
    req_input  = UPLOAD_FOLDER     / job_id
    req_pre    = PREPROCESS_FOLDER / job_id
    req_layout = LAYOUT_FOLDER     / job_id
    req_output = OUTPUT_FOLDER     / job_id

    for folder in [req_input, req_pre, req_layout, req_output]:
        folder.mkdir(parents=True, exist_ok=True)

    # 1. Lưu ảnh
    file = request.files['image']
    img_path = req_input / "capture.jpg"
    file.save(str(img_path))

    # 2. Pipeline
    preprocessed_files = run_preprocess(req_input, req_pre)
    if not preprocessed_files:
        _cleanup_dirs(req_input, req_pre, req_layout, req_output)
        return jsonify({"error": "No valid images after preprocessing"}), 400

    layout_results = run_layout(req_pre, req_layout, model=layout_model)
    ocr_results    = run_ocr(req_pre, layout_results, req_output, predictor=ocr_model)

    # 3. Kiểm tra giới hạn
    total_chars = _count_chars(ocr_results)
    if total_chars > MAX_DOC_CHARS:
        _cleanup_dirs(req_input, req_pre, req_layout, req_output)
        return jsonify({"error": "Document content exceeds character limit"}), 400

    # 4. Xuất Word
    doc_path = _build_docx(ocr_results, req_output, job_id)
    if doc_path.stat().st_size > MAX_DOC_BYTES:
        _cleanup_dirs(req_input, req_pre, req_layout, req_output)
        return jsonify({"error": "Document file size exceeds limit"}), 400

    # 5. Tìm ảnh layout để trả về cho Web UI xem trực tiếp
    layout_img_url = None
    layout_imgs = list(req_layout.glob("*_layout.jpg"))
    if layout_imgs:
        # Copy sang output để serve qua /output-file/
        dest = req_output / "layout_preview.jpg"
        shutil.copy(layout_imgs[0], dest)
        layout_img_url = f"/output-file/{job_id}/layout_preview.jpg"

    # 6. Dọn dẹp
    _cleanup_dirs(req_input, req_pre, req_layout)
    doc_b64 = base64.b64encode(doc_path.read_bytes()).decode()

    return jsonify({
        "status": "success",
        "data": {
            "layout": layout_results,
            "ocr": ocr_results,
            "docx_filename": doc_path.name,
            "docx_base64": doc_b64,
            "job_id": job_id,
            "layout_image_url": layout_img_url,
        }
    })


@app.route('/output-file/<job_id>/<filename>')
def serve_output_file(job_id, filename):
    """Serve ảnh layout preview cho Web UI."""
    # Giới hạn chỉ cho phép truy cập thư mục output của job đó
    folder = OUTPUT_FOLDER / job_id
    return send_from_directory(folder, filename)


# ══════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════

def _count_chars(ocr_results):
    return sum(len(line) for res in ocr_results for line in res.get("content", []))


def _build_docx(ocr_results, output_dir: Path, job_id: str) -> Path:
    """Tạo file Word từ kết quả OCR, dùng label YOLO để format đúng style."""
    doc = Document()
    for res in ocr_results:
        for item in res.get("content_with_labels", []):
            label = item.get("label", "text")
            text  = item.get("text", "").strip()
            if not text:
                continue
            if label == "title":
                doc.add_heading(text, level=1)
            elif label == "abandon":
                # Bỏ qua vùng nhiễu
                continue
            else:
                doc.add_paragraph(text)

    doc_path = output_dir / f"result_{job_id}.docx"
    doc.save(str(doc_path))
    return doc_path


def _cleanup_dirs(*dirs: Path):
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

if __name__ == '__main__':
    print("\n[DocScan Web] Khởi động tại http://0.0.0.0:5000")
    print("[DocScan Web] Truy cập từ trình duyệt: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)