import base64
import cv2
import requests
from pathlib import Path
import os
import threading

SERVER_URL = os.getenv("SERVER_URL", "http://10.222.177.172:5000/process")
is_processing = False  # Cờ trạng thái để tránh gửi liên tục nhiều request

def send_image_thread(frame):
    """Hàm chạy ngầm (background) để gửi API không làm đơ camera"""
    global is_processing
    try:
        # 1. Nén JPEG ở chất lượng cao (98%) để OCR đọc rõ chữ
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 98]
        _, img_encoded = cv2.imencode(".jpg", frame, encode_param)
        
        print("\n[INFO] Đang gửi ảnh lên Server xử lý...")
        files = {"image": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}
        
        response = requests.post(SERVER_URL, files=files, timeout=60)
        
        if response.status_code != 200:
            print(f"[ERROR] Lỗi HTTP: {response.status_code} - {response.text}")
            return
            
        result = response.json()
        if result.get("status") != "success":
            print(f"[ERROR] Server báo lỗi: {result}")
            return
            
        # 2. Xử lý dữ liệu trả về
        data = result.get("data", {})
        ocr_results = data.get("ocr", [])
        docx_b64 = data.get("docx_base64")
        job_id = data.get("job_id")
        
        if ocr_results:
            preview_lines = ocr_results[0].get("content", [])[:3]
            print(f"[SUCCESS] Nhận diện được chữ: {preview_lines}")
            
        if docx_b64 and job_id:
            docx_name = data.get("docx_filename") or f"result_{job_id}.docx"
            Path(docx_name).write_bytes(base64.b64decode(docx_b64))
            print(f"[SUCCESS] Đã lưu file Word tại: {Path(docx_name).resolve()}")
            
    except Exception as e:
        print(f"[ERROR] Mất kết nối tới Server: {e}")
    finally:
        # Giải phóng cờ để cho phép chụp ảnh tiếp
        is_processing = False
        print("[INFO] Đã sẵn sàng. Nhấn 'Space' để chụp tiếp...")


def capture_and_send():
    global is_processing
    cam = cv2.VideoCapture(0)
    
    # 3. Ép độ phân giải cao cho Raspberry Pi (Cực kỳ quan trọng cho OCR)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Cố gắng bật Auto-Focus (tùy thuộc vào loại module camera của Pi)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    print("[INFO] Nhấn 'Space' để chụp ảnh, hoặc 'Esc' để thoát...")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Không đọc được luồng từ camera!")
            break
            
        # 4. Tạo bản sao để vẽ Text UI (Không vẽ chữ lên ảnh gốc gửi cho AI)
        display_frame = frame.copy()
        
        if is_processing:
            cv2.putText(display_frame, "Dang xu ly tren Server...", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            cv2.putText(display_frame, "San sang (Nhan Space)", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Raspberry Pi Camera", display_frame)
        key = cv2.waitKey(1)

        # 5. Phân luồng xử lý
        if key == 32 and not is_processing: # Bấm Space
            is_processing = True
            # Gọi thread chạy ngầm, truyền bản sao ảnh sạch (frame) vào
            threading.Thread(target=send_image_thread, args=(frame.copy(),), daemon=True).start()

        elif key == 27: # Bấm Esc
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_send()