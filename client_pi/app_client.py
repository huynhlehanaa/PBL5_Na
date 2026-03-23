import cv2
import requests
import time

SERVER_URL = "http://10.222.177.172:5000/process"

def capture_and_send():
    cam = cv2.VideoCapture(0) # Mở camera
    print("Nhan 'Space' de chup anh hoac 'Esc' de thoat...")
    
    while True:
        ret, frame = cam.read()
        cv2.imshow("Raspberry Pi Camera", frame)
        
        key = cv2.waitKey(1)
        if key == 32: # Phím Space
            # 1. Mã hóa ảnh để gửi
            _, img_encoded = cv2.imencode('.jpg', frame)
            
            # 2. Gửi sang Server
            print("Dang gui anh len Server xu ly...")
            files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            
            try:
                response = requests.post(SERVER_URL, files=files)
                if response.status_code == 200:
                    print("Ket qua tra ve:", response.json())
                else:
                    print("Loi Server:", response.text)
            except Exception as e:
                print("Khong the ket noi den Server:", e)
                
        elif key == 27: # Phím Esc
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_send()