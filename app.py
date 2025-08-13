from flask import Flask, render_template, request, jsonify, send_file
import os
import base64
from datetime import datetime
import base64
import cv2
import re
import numpy as np
import pandas as pd
import face_recognition
from utils.face_utils import update_encoding_for_student
from utils.attendance_utils import mark_attendance


app = Flask(__name__)

# Thư mục lưu dữ liệu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENTS_DIR = os.path.join(BASE_DIR, "data", "students")
os.makedirs(STUDENTS_DIR, exist_ok=True)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/collect')
def collect():
    return render_template('collect.html')


@app.route('/attendance')
def attendance():
    return render_template('attendance.html')


@app.route('/save_images', methods=['POST'])
def save_images():
    data = request.get_json()
    name = data.get('name', '').strip()
    mssv = data.get('mssv', '').strip()
    lop = data.get('lop', '').strip()
    images = data.get('images', [])

    if not name or not mssv or not lop or not images:
        return jsonify({'status': 'error', 'message': 'Thiếu thông tin'}), 400

    # Bỏ ký tự không hợp lệ
    safe_name = re.sub(r'[\\/*?:"<>|]', '', name)
    safe_mssv = re.sub(r'[\\/*?:"<>|]', '', mssv)

    folder_name = f"{safe_mssv}_{safe_name}"
    save_dir = os.path.join('data', 'students', lop, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    saved_files = []
    for img_data in images:
        img_bytes = base64.b64decode(img_data.split(',')[1])
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, 'wb') as f:
            f.write(img_bytes)
        saved_files.append(file_path)

    # Mã hóa khuôn mặt cho từng ảnh
    for file_path in saved_files:
        update_encoding_for_student(folder_name, file_path)

    return jsonify({'status': 'success', 'message': 'Thu thập xong'})
    data = request.get_json()

    name = data.get("name")
    student_id = data.get("student_id")
    class_name = data.get("class")
    image_data = data.get("image")

    if not name or not student_id or not class_name or not image_data:
        return jsonify({"status": "error", "message": "Thiếu dữ liệu"}), 400

    # 🔹 Lọc ký tự lạ trong tên thư mục
    folder_name = f"{student_id}_{name}"
    folder_name = re.sub(r'[^a-zA-Z0-9_]', '', folder_name)

    save_path = os.path.join(STUDENTS_DIR, class_name, folder_name)
    os.makedirs(save_path, exist_ok=True)

    # Giải mã ảnh base64
    try:
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi giải mã ảnh: {e}"}), 400

    # Lưu ảnh
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    file_path = os.path.join(save_path, filename)

    try:
        with open(file_path, "wb") as f:
            f.write(image_bytes)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi lưu ảnh: {e}"}), 500

    # Gọi cập nhật encoding
    from utils.face_utils import update_encoding_for_student
    success = update_encoding_for_student(folder_name, file_path)
    if not success:
        return jsonify({"status": "error", "message": "Không thể đọc ảnh hoặc không tìm thấy khuôn mặt"}), 400

    return jsonify({"status": "success", "message": f"Đã lưu {filename}"}), 200
# Nguyên tắc là khi mở camera trên trình duyệt, mỗi frame gửi về server → server nhận diện → nếu trùng thì ghi vào attendance.xlsx.


@app.route('/check_attendance', methods=['POST'])
def check_attendance():
    data = request.get_json()
    image_data = data.get("image")

    if not image_data:
        return jsonify({"status": "error", "message": "Không có dữ liệu ảnh"}), 400

    # Giải mã base64
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Encode khuôn mặt hiện tại
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb_img)
    encodes = face_recognition.face_encodings(rgb_img, faces)

    if len(encodes) == 0:
        return jsonify({"status": "error", "message": "Không tìm thấy khuôn mặt"}), 404

    current_encode = encodes[0]

    # Load dữ liệu sinh viên
    for folder in os.listdir(STUDENTS_DIR):
        folder_path = os.path.join(STUDENTS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        # Giải tên folder
        try:
            student_id, name, class_name = folder.split("_", 2)
        except ValueError:
            continue  # Bỏ qua thư mục không đúng định dạng

        # So sánh với từng ảnh đã lưu
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            known_img = face_recognition.load_image_file(img_path)
            known_encode = face_recognition.face_encodings(known_img)
            if len(known_encode) == 0:
                continue
            match = face_recognition.compare_faces(
                [known_encode[0]], current_encode, tolerance=0.5)
            if match[0]:
                mark_attendance(student_id, name, class_name)
                return jsonify({
                    "status": "success",
                    "message": f"Điểm danh thành công: {name} ({student_id}) - {class_name}"
                })

    return jsonify({"status": "error", "message": "Không tìm thấy dữ liệu sinh viên"})
# Xuat Excel
# Bấm "Điểm danh" → chụp ảnh từ camera → gửi server nhận diện → ghi vào attendance.xlsx.

# Bấm "Xuất Excel" → tải file attendance.xlsx về máy.


@app.route('/export_excel', methods=['GET'])
def export_excel():
    if not os.path.exists(os.path.join("data", "attendance.xlsx")):
        return jsonify({"status": "error", "message": "Chưa có dữ liệu điểm danh"}), 404

    return send_file(
        os.path.join("data", "attendance.xlsx"),
        as_attachment=True,
        download_name="attendance.xlsx"
    )


# danh sách sinh viên

STUDENTS_INFO_FILE = os.path.join("data", "students_info.xlsx")


@app.route('/get_students/<class_name>', methods=['GET'])
def get_students(class_name):
    students_info_file = os.path.join("data", "students_info.xlsx")

    if not os.path.exists(students_info_file):
        return jsonify({"students": []})

    df_info = pd.read_excel(students_info_file, engine="openpyxl")
    df_filtered = df_info[df_info["Lớp"] == class_name]

    # Đảm bảo STT được sắp xếp lại
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered["STT"] = df_filtered.index + 1

    students_list = df_filtered.to_dict(orient="records")
    return jsonify({"students": students_list})


if __name__ == '__main__':
    app.run(debug=True)
