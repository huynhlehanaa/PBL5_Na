from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def build_docx(ocr_results: list, output_dir: Path, job_id: str) -> Path:
    """
    Tạo file Word từ kết quả OCR kết hợp nhãn (label) của YOLO.
    Định dạng văn bản dựa trên loại bố cục được nhận diện.
    """
    # Khởi tạo tài liệu Word
    doc = Document()
    
    # Thiết lập Font chữ mặc định cho toàn bộ tài liệu (Tùy chọn)
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(13)

    for res in ocr_results:
        content_items = res.get("content_with_labels", [])
        
        for item in content_items:
            label = item.get("label", "plain text").lower()
            text = item.get("text", "").strip()
            
            if not text:
                continue

            # Ánh xạ Label của YOLO sang định dạng Word
            if label == "title":
                # Tiêu đề: Chữ to, in đậm, căn giữa
                heading = doc.add_heading(level=1)
                run = heading.add_run(text)
                run.font.name = 'Times New Roman'
                run.font.color.rgb = None # Dùng màu đen mặc định
                heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
            elif label == "list":
                # Danh sách: Thêm bullet point
                doc.add_paragraph(text, style='List Bullet')
                
            elif label == "caption":
                # Chú thích (ví dụ dưới ảnh/bảng): In nghiêng, căn giữa
                p = doc.add_paragraph()
                run = p.add_run(text)
                run.italic = True
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
            elif label == "table":
                # Hiện tại OCR chưa bóc tách từng ô của bảng hoàn chỉnh, 
                # nên ta tạm đóng khung đoạn text này lại để người dùng chú ý
                p = doc.add_paragraph(f"--- [Bảng biểu] ---\n{text}\n-------------------")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
            elif label == "abandon" or label == "figure":
                # Bỏ qua các vùng nhiễu hoặc hình ảnh không có chữ
                continue
                
            else:
                # plain text hoặc các nhãn khác: Đoạn văn bình thường
                p = doc.add_paragraph(text)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY # Căn đều 2 bên

    # Đảm bảo thư mục đầu ra tồn tại
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lưu file
    doc_path = output_dir / f"result_{job_id}.docx"
    doc.save(str(doc_path))
    
    return doc_path