from pathlib import Path
from preprocess import run_preprocess

def main():
    base = Path(__file__).resolve().parents[1]
    input_dir = base / "data" / "input"
    pre_dir = base / "data" / "preprocessed"
    output_dir = base / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== STEP 1: PREPROCESS ===")
    preprocessed_files = run_preprocess(input_dir, pre_dir)

    print("\n=== STEP 2: LAYOUT (TODO) ===")
    print("Sẽ tích hợp DocLayout-YOLO ở bước tiếp.")

    print("\n=== STEP 3: OCR (TODO) ===")
    print("Sẽ tích hợp vietnamese-ocr sau bước layout.")

    print("\nDone. Tổng ảnh đã preprocess:", len(preprocessed_files))

if __name__ == "__main__":
    main()