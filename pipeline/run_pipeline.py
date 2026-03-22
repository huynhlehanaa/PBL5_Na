from pathlib import Path
from preprocess import run_preprocess
from layout import run_layout
from ocr import run_ocr

def main():
    base = Path(__file__).resolve().parents[1]
    input_dir = base / "data" / "input"
    pre_dir = base / "data" / "preprocessed"
    layout_dir = base / "data" / "layout"
    ocr_dir = base / "data" / "ocr"
    output_dir = base / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== STEP 1: PREPROCESS ===")
    preprocessed_files = run_preprocess(input_dir, pre_dir)

    print("\n=== STEP 2: LAYOUT (DocLayout-YOLO) ===")
    try:
        layout_results = run_layout(pre_dir, layout_dir)
    except ImportError as exc:
        layout_results = []
        print("[SKIP] Layout: thiếu phụ thuộc:", exc)
        print("Cài đặt: pip install doclayout-yolo torch")

    print("\n=== STEP 3: OCR (PaddleOCR) ===")
    try:
        run_ocr(pre_dir, layout_results, ocr_dir)
    except ImportError as exc:
        print("[SKIP] OCR: thiếu phụ thuộc:", exc)
        print("Cài đặt: pip install paddleocr==2.7.0.3")

    print("\nDone. Tổng ảnh đã preprocess:", len(preprocessed_files))
    print("Kết quả:")
    print(f" - Ảnh tiền xử lý: {pre_dir}")
    print(f" - Layout annotate/json: {layout_dir}")
    print(f" - OCR text/json: {ocr_dir}")

if __name__ == "__main__":
    main()
