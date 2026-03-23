from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from pathlib import Path
from pipeline.layout import run_layout

app = Flask(__name__)

# Trỏ vào đúng thư mục data/input của bạn để chứa ảnh tạm
UPLOAD_FOLDER = Path("data/input")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # 1. Nhận ảnh từ Pi gửi lên (sẽ ghi đè lên file cũ để không bị đầy ổ cứng)
    file = request.files['image']
    img_path = UPLOAD_FOLDER / "capture.jpg"
    file.save(str(img_path))
    
    # 2. Gọi hàm AI
    # Trỏ vào đúng thư mục data/layout của bạn để lưu kết quả Layout
    output_dir = Path("data/layout")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hàm run_layout sẽ quét thư mục data/input và xuất ra data/layout
    results = run_layout(UPLOAD_FOLDER, output_dir)
    
    # 3. Trả kết quả JSON về cho Pi
    return jsonify({
        "status": "success",
        "data": results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)