from flask import Flask, request, jsonify
import base64, uuid, shutil, os, json, time
from pathlib import Path
from collections import Counter
from pipeline.layout import run_layout, load_layout_model
from pipeline.preprocess import run_preprocess
from pipeline.ocr import run_ocr, load_ocr_model
from pipeline.export_word import build_docx
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

def _cleanup_dirs(*dirs):
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    started_at = time.perf_counter()
    timings = {}
    job_id = uuid.uuid4().hex
    req_input  = UPLOAD_FOLDER     / job_id
    req_pre    = PREPROCESS_FOLDER / job_id
    req_pre_layout = req_pre / "layout"
    req_pre_ocr = req_pre / "ocr"
    req_layout = LAYOUT_FOLDER     / job_id
    req_output = OUTPUT_FOLDER     / job_id
    for folder in [req_input, req_pre, req_pre_layout, req_pre_ocr, req_layout, req_output]:
        folder.mkdir(parents=True, exist_ok=True)

    file = request.files['image']
    t0 = time.perf_counter()
    input_path = req_input / "capture.jpg"
    file.save(str(input_path))
    timings["save_image_sec"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    preprocessed_layout_files = run_preprocess(req_input, req_pre_layout, mode="layout")
    timings["preprocess_layout_sec"] = round(time.perf_counter() - t0, 4)
    t0 = time.perf_counter()
    preprocessed_ocr_files = run_preprocess(req_input, req_pre_ocr, mode="ocr")
    timings["preprocess_ocr_sec"] = round(time.perf_counter() - t0, 4)
    if not preprocessed_layout_files or not preprocessed_ocr_files:
        _cleanup_dirs(req_input, req_pre, req_layout, req_output)
        return jsonify({"error": "No valid images after preprocessing"}), 400

    # ✅ Truyền model đã load sẵn vào
    t0 = time.perf_counter()
    layout_results = run_layout(req_pre_layout, req_layout, model=layout_model)
    timings["layout_sec"] = round(time.perf_counter() - t0, 4)
    t0 = time.perf_counter()
    ocr_results    = run_ocr(req_pre_ocr, layout_results, req_output, predictor=ocr_model)
    timings["ocr_sec"] = round(time.perf_counter() - t0, 4)

    if _count_chars(ocr_results) > MAX_DOC_CHARS:
        _cleanup_dirs(req_input, req_pre, req_layout, req_output)
        return jsonify({"error": "Document content exceeds limit"}), 400

    t0 = time.perf_counter()
    doc_path = build_docx(ocr_results, req_output, job_id)
    timings["export_docx_sec"] = round(time.perf_counter() - t0, 4)
    if doc_path.stat().st_size > MAX_DOC_BYTES:
        _cleanup_dirs(req_input, req_pre, req_layout, req_output)
        return jsonify({"error": "Document file size exceeds limit"}), 400

    debug_report = {
        "job_id": job_id,
        "input_filename": file.filename,
        "input_saved_as": input_path.name,
        "layout_summary": _summarize_layout(layout_results),
        "ocr_summary": _summarize_ocr(ocr_results),
        "timings_sec": timings,
        "total_elapsed_sec": round(time.perf_counter() - started_at, 4),
        "docx_filename": doc_path.name,
        "docx_size_bytes": doc_path.stat().st_size,
    }
    debug_filename = _write_debug_report(req_output, debug_report)

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
            "debug_report_filename": debug_filename,
        }
    })


def _summarize_layout(layout_results):
    labels = []
    scores = []
    for page in layout_results:
        for box in page.get("boxes", []):
            labels.append(str(box.get("label", "")))
            try:
                scores.append(float(box.get("score", 0.0)))
            except Exception:
                pass
    by_label = dict(Counter(labels))
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {
        "total_boxes": len(labels),
        "boxes_by_label": by_label,
        "avg_confidence": avg_score,
    }


def _summarize_ocr(ocr_results):
    lines = [line for res in ocr_results for line in res.get("content", []) if str(line).strip()]
    return {
        "total_lines": len(lines),
        "total_chars": sum(len(line) for line in lines),
    }


def _write_debug_report(output_dir: Path, payload: dict) -> str:
    filename = "debug_report.json"
    (output_dir / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return filename

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
