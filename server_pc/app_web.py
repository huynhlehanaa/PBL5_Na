from flask import Flask, request, jsonify, render_template, send_from_directory
import base64, uuid, shutil, os, json, time
from pathlib import Path
import socket
from collections import Counter

from pipeline.layout import run_layout, load_layout_model
from pipeline.preprocess import run_preprocess
from pipeline.ocr import run_ocr, load_ocr_model
from pipeline.export_word import build_docx
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
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "error": "No image uploaded"}), 400

        started_at = time.perf_counter()
        timings = {}
        job_id     = uuid.uuid4().hex
        req_input  = UPLOAD_FOLDER     / job_id
        req_pre    = PREPROCESS_FOLDER / job_id
        req_pre_layout = req_pre / "layout"
        req_pre_ocr = req_pre / "ocr"
        req_layout = LAYOUT_FOLDER     / job_id
        req_output = OUTPUT_FOLDER     / job_id

        for folder in [req_input, req_pre, req_pre_layout, req_pre_ocr, req_layout, req_output]:
            folder.mkdir(parents=True, exist_ok=True)

        # 1. Lưu ảnh
        file = request.files['image']
        img_path = req_input / "capture.jpg"
        t0 = time.perf_counter()
        file.save(str(img_path))
        timings["save_image_sec"] = round(time.perf_counter() - t0, 4)
        print(f"[{job_id}] 📥 Ảnh đã lưu: {img_path}")

        # 2. Pipeline
        print(f"[{job_id}] ⚙️ Chạy preprocessing cho layout...")
        t0 = time.perf_counter()
        preprocessed_layout_files = run_preprocess(req_input, req_pre_layout, mode="layout")
        timings["preprocess_layout_sec"] = round(time.perf_counter() - t0, 4)
        print(f"[{job_id}] ⚙️ Chạy preprocessing cho OCR...")
        t0 = time.perf_counter()
        preprocessed_ocr_files = run_preprocess(req_input, req_pre_ocr, mode="ocr")
        timings["preprocess_ocr_sec"] = round(time.perf_counter() - t0, 4)
        if not preprocessed_layout_files or not preprocessed_ocr_files:
            _cleanup_dirs(req_input, req_pre, req_layout, req_output)
            return jsonify({"status": "error", "error": "No valid images after preprocessing"}), 400

        print(f"[{job_id}] 🔍 Chạy layout detection (YOLO)...")
        t0 = time.perf_counter()
        layout_results = run_layout(req_pre_layout, req_layout, model=layout_model)
        timings["layout_sec"] = round(time.perf_counter() - t0, 4)
        
        print(f"[{job_id}] 📝 Chạy OCR...")
        t0 = time.perf_counter()
        ocr_results    = run_ocr(req_pre_ocr, layout_results, req_output, predictor=ocr_model)
        timings["ocr_sec"] = round(time.perf_counter() - t0, 4)

        # 3. Kiểm tra giới hạn
        total_chars = _count_chars(ocr_results)
        if total_chars > MAX_DOC_CHARS:
            _cleanup_dirs(req_input, req_pre, req_layout, req_output)
            return jsonify({"status": "error", "error": "Document content exceeds character limit"}), 400

        # 4. Xuất Word
        print(f"[{job_id}] 📄 Tạo file Word...")
        t0 = time.perf_counter()
        doc_path = build_docx(ocr_results, req_output, job_id)
        timings["export_docx_sec"] = round(time.perf_counter() - t0, 4)
        if doc_path.stat().st_size > MAX_DOC_BYTES:
            _cleanup_dirs(req_input, req_pre, req_layout, req_output)
            return jsonify({"status": "error", "error": "Document file size exceeds limit"}), 400

        # 5. Tìm ảnh layout để trả về cho Web UI xem trực tiếp
        layout_img_url = None
        layout_imgs = list(req_layout.glob("*_layout.jpg"))
        if layout_imgs:
            # Copy sang output để serve qua /output-file/
            dest = req_output / "layout_preview.jpg"
            shutil.copy(layout_imgs[0], dest)
            layout_img_url = f"/output-file/{job_id}/layout_preview.jpg"

        # 6. Dọn dẹp thư mục tạm
        debug_report = {
            "job_id": job_id,
            "input_filename": file.filename,
            "input_saved_as": img_path.name,
            "layout_summary": _summarize_layout(layout_results),
            "ocr_summary": _summarize_ocr(ocr_results),
            "timings_sec": timings,
            "total_elapsed_sec": round(time.perf_counter() - started_at, 4),
            "docx_filename": doc_path.name,
            "docx_size_bytes": doc_path.stat().st_size,
        }
        debug_filename = _write_debug_report(req_output, debug_report)
        debug_report_url = f"/output-file/{job_id}/{debug_filename}"

        _cleanup_dirs(req_input, req_pre, req_layout)
        doc_b64 = base64.b64encode(doc_path.read_bytes()).decode()

        print(f"[{job_id}] ✅ Xử lý thành công! {total_chars} ký tự")
        
        return jsonify({
            "status": "success",
            "data": {
                "layout": layout_results,
                "ocr": ocr_results,
                "docx_filename": doc_path.name,
                "docx_base64": doc_b64,
                "job_id": job_id,
                "layout_image_url": layout_img_url,
                "debug_report_url": debug_report_url,
            }
        })
    
    except Exception as e:
        print(f"❌ Lỗi xử lý: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/output-file/<job_id>/<filename>')
def serve_output_file(job_id, filename):
    """Serve ảnh layout preview cho Web UI."""
    try:
        # Giới hạn chỉ cho phép truy cập thư mục output của job đó
        folder = OUTPUT_FOLDER / job_id
        return send_from_directory(folder, filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "error": "Not Found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"status": "error", "error": "Internal Server Error"}), 500


# ══════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════

def _count_chars(ocr_results):
    return sum(len(line) for res in ocr_results for line in res.get("content", []))


def _cleanup_dirs(*dirs: Path):
    """Xóa các thư mục tạm thời."""
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


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


def find_free_port(start_port=5000):
    """Tìm port khả dụng."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                s.listen(1)
                return port
        except OSError:
            continue
    return start_port


if __name__ == '__main__':
    port = find_free_port(5000)
    
    app.run(host='0.0.0.0', port=port, debug=False)