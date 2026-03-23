import base64
from datetime import datetime
import cv2
import requests
from pathlib import Path

SERVER_URL = "http://10.222.177.172:5000/process"


def capture_and_send():
    cam = cv2.VideoCapture(0)  # Mở camera
    print("Nhan 'Space' de chup anh hoac 'Esc' de thoat...")

    while True:
        ret, frame = cam.read()
        cv2.imshow("Raspberry Pi Camera", frame)

        key = cv2.waitKey(1)
        if key == 32:  # Phím Space
            # 1. Mã hóa ảnh để gửi
            _, img_encoded = cv2.imencode(".jpg", frame)

            # 2. Gửi sang Server
            print("Dang gui anh len Server xu ly...")
            files = {"image": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}

            try:
                response = requests.post(SERVER_URL, files=files)
                if response.status_code != 200:
                    print("Loi Server:", response.text)
                    continue

                result = response.json()
                if result.get("status") != "success":
                    print("Xu ly that bai:", result)
                    continue

                data = result.get("data", {})
                ocr_results = data.get("ocr", [])
                docx_b64 = data.get("docx_base64")
                job_id = data.get("job_id")

                # Hiển thị nhanh 3 dòng đầu tiên của OCR để kiểm tra
                if ocr_results:
                    first_image = ocr_results[0]
                    preview_lines = first_image.get("content", [])[:3]
                    print("Mau ket qua OCR:", preview_lines)

                # Lưu file Word trả về từ server
                if docx_b64:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    docx_name = data.get("docx_filename") or f"Ket_qua_{job_id if job_id is not None else timestamp}.docx"
                    docx_path = Path(docx_name)
                    docx_path.write_bytes(base64.b64decode(docx_b64))
                    print(f"Da luu file Word: {docx_path.resolve()}")

            except Exception as e:
                print("Khong the ket noi den Server:", e)

        elif key == 27:  # Phím Esc
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_send()
