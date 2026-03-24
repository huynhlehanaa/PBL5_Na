from pathlib import Path
import io
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def build_docx_mem(ocr_results: list):
    doc = Document()
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(13)

    for res in ocr_results:
        content_items = res.get("content_with_labels", [])
        if not content_items: continue

        # 1. Sắp xếp sơ bộ theo TÂM Y (Center Y) thay vì Đỉnh Y
        content_items.sort(key=lambda x: x.get("y", 0) + (x.get("h", 20) / 2.0))

        lines = []
        for item in content_items:
            text = item.get("text", "").strip()
            if not text: continue
            
            y_top = item.get("y", 0)
            h = item.get("h", 20)
            
            # TÍNH TÂM Y THỰC SỰ ĐỂ GOM DÒNG
            cy = y_top + (h / 2.0)
            
            # Ngưỡng động: 45% chiều cao chữ
            dynamic_y_thresh = max(8, h * 0.45) 

            placed = False
            if lines:
                last_line = lines[-1]
                if abs(cy - last_line['cy']) < dynamic_y_thresh:
                    last_line['items'].append(item)
                    last_line['cy'] = (last_line['cy'] * (len(last_line['items']) - 1) + cy) / len(last_line['items'])
                    placed = True
                    
            if not placed:
                lines.append({'cy': cy, 'items': [item]})

        # 2. Xử lý từng dòng
        for line in lines:
            line['items'].sort(key=lambda item: item.get("x", 0))

            full_line_text = ""
            for i, it in enumerate(line['items']):
                if i > 0:
                    prev_it = line['items'][i-1]
                    gap = it['x'] - (prev_it['x'] + prev_it.get('w', 20))
                    
                    if gap > 60:
                        full_line_text += "\t" * max(1, int(gap/80)) + it['text']
                    elif gap > 0:
                        full_line_text += " " + it['text']
                    else:
                        full_line_text += " " + it['text']
                else:
                    full_line_text = it['text']

            label = line['items'][0].get("label", "plain text").lower()
            x_start = line['items'][0].get("x", 0)

            p = doc.add_paragraph(full_line_text)
            
            if label in ["title", "header"]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if p.runs: p.runs[0].bold = True 
            elif label == "table":
                p.insert_paragraph_before("--- [Bảng biểu] ---")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                indent_inches = max(0, ((x_start - 60) / 1400.0) * 6.5)
                if indent_inches > 0.3: 
                    p.paragraph_format.left_indent = Inches(indent_inches)

    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream