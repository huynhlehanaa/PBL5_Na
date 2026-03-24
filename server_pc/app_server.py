from flask import Flask, request, jsonify
import base64, uuid, shutil, os
from pathlib import Path
from pipeline.layout import run_layout, load_layout_model
from pipeline.preprocess import run_preprocess
from pipeline.ocr import run_ocr, load_ocr_model
from docx import Document

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Load model 1 lần khi server khởi động
layout_model = load_layout_model()
ocr_model = load_ocr_model()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DOC_BYTES    = 10 * 1024 * 1024
MAX_DOC_CHARS    = 1_000_000
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

UPLOAD_FOLDER     = BASE_DIR / "data" / "input"
PREPROCESS_FOLDER = BASE_DIR / "data" / "preprocessed"
LAYOUT_FOLDER     = BASE_DIR / "data" / "layout"
OUTPUT_FOLDER     = BASE_DIR / "data" / "output"

for folder in [UPLOAD_FOLDER, PREPROCESS_FOLDER, LAYOUT_FOLDER, OUTPUT_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

def _count_chars(ocr_results):
    return sum(len(line) for res in ocr_results for line in res.get("content", []))

def _build_docx(ocr_results, output_dir, job_id):
    doc = Document()
    for res in ocr_results:
        for item in res.get("content_with_labels", []):
            if item["label"] == "title":
                doc.add_heading(item["text"], level=1)
            else:
                doc.add_paragraph(item["text"])
    doc_path = output_dir / f"result_{job_id}.docx"
    doc.save(str(doc_path))
    return doc_path

def _cleanup_dirs(*dirs):
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    job_id = uuid.uuid4().hex
    req_input  = UPLOAD_FOLDER     / job_id
    req_pre    = PREPROCESS_FOLDER / job_id
    req_layout = LAYOUT_FOLDER     / job_id
    req_output = OUTPUT_FOLDER     / job_id
    for folder in [req_input, req_pre, req_layout, req_output]:
        folder.mkdir(parents=True, exist_ok=True)

    file = request.files['image']
    file.save(str(req_input / "capture.jpg"))

    preprocessed_files = run_preprocess(req_input, req_pre)
    if not preprocessed_files:
        _cleanup_dirs(req_input, req_pre, req_layout, req_output)
        return jsonify({"error": "No valid images after preprocessing"}), 400

    # ✅ Truyền model đã load sẵn vào
    layout_results = run_layout(req_pre, req_layout, model=layout_model)
    ocr_results    = run_ocr(req_pre, layout_results, req_output, predictor=ocr_model)

    if _count_chars(ocr_results) > MAX_DOC_CHARS:
        _cleanup_dirs(req_input, req_pre, req_layout, req_output)
        return jsonify({"error": "Document content exceeds limit"}), 400

    doc_path = _build_docx(ocr_results, req_output, job_id)
    if doc_path.stat().st_size > MAX_DOC_BYTES:
        _cleanup_dirs(req_input, req_pre, req_layout, req_output)
        return jsonify({"error": "Document file size exceeds limit"}), 400

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
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
