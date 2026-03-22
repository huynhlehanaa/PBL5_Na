from pathlib import Path
import cv2
import imutils

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def preprocess_image(img):
    # 1) resize giữ tỉ lệ
    img = imutils.resize(img, width=1400)

    # 2) grayscale + denoise nhẹ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3) tăng tương phản cục bộ (hữu ích cho OCR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4) nhị phân thích nghi
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # 5) morphology nhẹ để làm sạch
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    return bw

def run_preprocess(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS]

    if not files:
        print(f"[WARN] Không có ảnh trong: {input_dir}")
        return []

    saved = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[SKIP] Không đọc được ảnh: {p}")
            continue

        out = preprocess_image(img)
        out_path = output_dir / p.name
        cv2.imwrite(str(out_path), out)
        saved.append(out_path)
        print(f"[OK] {p.name} -> {out_path}")

    return saved

if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    inp = base / "data" / "input"
    out = base / "data" / "preprocessed"
    run_preprocess(inp, out)