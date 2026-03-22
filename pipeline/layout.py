from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
from huggingface_hub import hf_hub_download  # ← thêm

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _select_device():
    """Chọn device tốt nhất sẵn có."""
    try:
        import torch
    except ImportError:
        raise ImportError("Thiếu torch. Cài đặt: pip install torch")

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_layout(
    input_dir: Path,
    output_dir: Path,
    model_path: str | None = None,
    conf: float = 0.25,
) -> List[Dict]:
    """
    Chạy DocLayout-YOLO trên thư mục ảnh đã tiền xử lý.

    - model_path: đường dẫn .pt cục bộ hoặc repo HF. Mặc định "juliozhao/DocLayout-YOLO-DocStructBench".
    """
    try:
        from doclayout_yolo import YOLOv10
    except ImportError as exc:
        raise ImportError("Thiếu doclayout-yolo. Cài đặt: pip install doclayout-yolo") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    if not files:
        print(f"[WARN] Không tìm thấy ảnh để layout ở {input_dir}")
        return []

    device = _select_device()
    model_id = model_path or "juliozhao/DocLayout-YOLO-DocStructBench"
    print(f"[INFO] Load DocLayout-YOLO: {model_id} (device={device})")

    # Nếu là repo HF thì tải checkpoint về cache, rồi nạp
    if "/" in model_id and not Path(model_id).exists():
        ckpt = hf_hub_download(
            repo_id=model_id,
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            repo_type="model",
        )
        model = YOLOv10(ckpt)
    else:  # đường dẫn .pt cục bộ
        model = YOLOv10(model_id)

    layout_results: List[Dict] = []
    for p in files:
        det_res = model.predict(str(p), imgsz=1024, conf=conf, device=device)
        res = det_res[0]
        annotated = res.plot(pil=True, line_width=4, font_size=18)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        annotated_path = output_dir / f"{p.stem}_layout.jpg"
        cv2.imwrite(str(annotated_path), annotated)

        boxes = res.boxes
        names = res.names if hasattr(res, "names") else getattr(model, "names", {})
        items = []
        if boxes is not None:
            cls_list = boxes.cls.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for bbox, cls_id, score in zip(xyxy, cls_list, confs):
                cls_idx = int(cls_id)
                label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                items.append({"bbox": [float(v) for v in bbox.tolist()], "label": label, "score": float(score)})

        layout_results.append({"image": p.name, "boxes": items})
        print(f"[LAYOUT] {p.name}: {len(items)} box, lưu {annotated_path.name}")

    summary_path = output_dir / "layouts.json"
    summary_path.write_text(json.dumps(layout_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Đã lưu layout JSON: {summary_path}")
    return layout_results


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    inp = base / "data" / "preprocessed"
    out = base / "data" / "layout"
    run_layout(inp, out)