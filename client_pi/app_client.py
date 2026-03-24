import base64
import cv2
import requests
from pathlib import Path
import os

SERVER_URL = os.getenv("SERVER_URL", "http://10.222.177.172:5000/process")

def capture_and_send():
    cam = cv2.VideoCapture(0)
    print("Nhan 'Space' de chup anh hoac 'Esc' de thoat...")

    while True:
        ret, frame = cam.read()
        cv2.imshow("Raspberry Pi Camera", frame)
        key = cv2.waitKey(1)

        if key == 32:
            _, img_encoded = cv2.imencode(".jpg", frame)
            print("Dang gui anh len Server xu ly...")
            files = {"image": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}
            try:
                # timeout
                response = requests.post(SERVER_URL, files=files, timeout=60)
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
                if ocr_results:
                    preview_lines = ocr_results[0].get("content", [])[:3]
                    print("Mau ket qua OCR:", preview_lines)
                if docx_b64:
                    if not job_id:
                        print("Canh bao: khong co job_id")
                        continue
                    docx_name = data.get("docx_filename") or f"result_{job_id}.docx"
                    Path(docx_name).write_bytes(base64.b64decode(docx_b64))
                    print(f"Da luu file Word: {Path(docx_name).resolve()}")
            except Exception as e:
                print("Khong the ket noi den Server:", e)

        elif key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_send()
