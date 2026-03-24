from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def build_docx(ocr_results: list, output_dir: Path, job_id: str) -> Path:
    doc = Document()
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(13)

    for res in ocr_results:
        content_items = res.get("content_with_labels", [])
        if not content_items: continue

        # Sort theo chiều dọc trước
        content_items.sort(key=lambda x: x.get("y", 0))

        lines = []
        for item in content_items:
            text = item.get("text", "").strip()
            if not text: continue
            y = item.get("y", 0)
            
            placed = False
            if lines:
                last_line = lines[-1]
                # Nới lỏng khoảng cách nhận diện cùng 1 dòng lên 25px
                if abs(y - last_line['cy']) < 25: 
                    last_line['items'].append(item)
                    # Cập nhật lại trung tâm y của dòng
                    last_line['cy'] = (last_line['cy'] * (len(last_line['items']) - 1) + y) / len(last_line['items'])
                    placed = True
            if not placed:
                lines.append({'cy': y, 'items': [item]})

        for line in lines:
            # Sort các thành phần trong dòng từ trái qua phải
            items = sorted(line['items'], key=lambda x: x.get("x", 0))
            
            # GỘP CHỮ NẾU NẰM TRÊN CÙNG 1 DÒNG (Dùng Tab thay vì dùng Bảng)
            full_line_text = ""
            for i, it in enumerate(items):
                if i > 0:
                    # Nếu khoảng cách giữa 2 block lớn, dùng Tab, nếu gần dùng khoảng trắng
                    gap = it['x'] - (items[i-1]['x'] + items[i-1].get('w', 50))
                    if gap > 80:
                        full_line_text += "\t" * int(gap/80) + it['text']
                    else:
                        full_line_text += " " + it['text']
                else:
                    full_line_text = it['text']

            label = items[0].get("label", "plain text").lower()
            x_start = items[0].get("x", 0)

            # Thêm paragraph vào Word
            p = doc.add_paragraph(full_line_text)
            
            # Định dạng dựa trên nhãn YOLO trả về
            if label in ["title", "header"]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if p.runs: p.runs[0].bold = True 
            elif label == "table":
                p.insert_paragraph_before("--- [Bảng biểu] ---")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                # Xử lý thụt lề (Indentation)
                indent_inches = max(0, ((x_start - 60) / 1400.0) * 6.5)
                if indent_inches > 0.3: 
                    p.paragraph_format.left_indent = Inches(indent_inches)
                else: 
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_path = output_dir / f"result_{job_id}.docx"
    doc.save(str(doc_path))
    return doc_path