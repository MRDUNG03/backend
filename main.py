import base64
from fastapi import FastAPI, HTTPException,status,Path,Query,Form,Request
from fastapi.params import Depends
from pydantic import BaseModel
import mysql.connector
from datetime import datetime
from model.LoginRequest import LoginRequest
from model.connected_DB import connected_DB
from model.Device import Device
from model.User import username,password
from model.Alert import Alert, AlertResponse
from typing import List, Optional
from jose import JWTError,jwt
from fastapi.security import HTTPBasic,HTTPBasicCredentials
from passlib.context import CryptContext
import json
from paho.mqtt import client as mqtt_client
import threading
import requests
import os
import csv
import numpy as np
import asyncio
from scipy.stats import kurtosis, skew
import joblib
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="GIÁM SÁT TRẠNG THÁI ĐỘNG CƠ ĐIỆN ",openapi_url="/openapi.json",docs_url="/docs",description="API for monitoring electric motor status")
security = HTTPBasic()
from fastapi.security.utils import get_authorization_scheme_param
import pandas as pd
import traceback
from fastapi import FastAPI
from pydantic import BaseModel


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------ SCHEMA DỮ LIỆU ------------------
class SensorData(BaseModel):
    ax: float
    ay: float
    az: float
    current: float
    voltage: float
    temp: float
# ==================== LOAD MODEL & SCALER ====================
MODEL_PATH = r"D:\database\random_success\rf_model.pkl" 
SCALER_PATH = r"D:\database\random_success\rf_scaler.pkl"
# MODEL_PATH = r"D:\database\svm\svm_low_sensitivity_model_full.pkl"
# SCALER_PATH = r"D:\database\svm\svm_low_sensitivity_scaler_full.pkl"
# MODEL_PATH = r"D:\database\svm\model.pkl"
# SCALER_PATH = r"D:\database\svm\scaler.pkl"
try: 
    svm_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("ĐÃ TẢI THÀNH CÔNG MODEL & SCALER")
except Exception as e:
    print(f"LỖI TẢI MODEL/SCALER: {e}")
    svm_model = None
    scaler = None

# ==================== BUFFER TOÀN CỤC ====================
sensor_buffer: List[list] = []
BUFFER_SIZE = 512
LAST_CHECKED_ID = 0

# ==================== HÀM TRÍCH ĐẶC TRƯNG BACKEND ====================
def extract_features_from_segment(seg_df: pd.DataFrame) -> dict:
    def feat_1d(arr):
        if len(arr) == 0:
            return {"mean": np.nan, "rms": np.nan, "std": np.nan, "peak": np.nan}
        arr = np.asarray(arr, dtype=float)
        return {
            "mean": float(arr.mean()),
            "rms": float(np.sqrt(np.mean(arr**2))),
            "std": float(arr.std()),
            "peak": float(np.max(np.abs(arr))),
        }

    ax      = seg_df["ax"].values       if "ax"       in seg_df.columns else []
    ay      = seg_df["ay"].values       if "ay"       in seg_df.columns else []
    az      = seg_df["az"].values       if "az"       in seg_df.columns else []
    current = seg_df["current"].values  if "current"  in seg_df.columns else []
    voltage = seg_df["voltage"].values  if "voltage"  in seg_df.columns else []
    temp    = seg_df["temp"].values     if "temp"     in seg_df.columns else []

    f_ax  = feat_1d(ax)
    f_ay  = feat_1d(ay)
    f_az  = feat_1d(az)
    f_cur = feat_1d(current)
    f_vol = feat_1d(voltage)

    mean_temp = float(np.mean(temp)) if len(temp) > 0 else np.nan

    # ✅ Trả về đúng đủ cột voltage để khỏi bị thiếu khi scaler.transform
    return {
        "Mean_ax": f_ax["mean"], "RMS_ax": f_ax["rms"], "STD_ax": f_ax["std"], "Peak_ax": f_ax["peak"],
        "Mean_ay": f_ay["mean"], "RMS_ay": f_ay["rms"], "STD_ay": f_ay["std"], "Peak_ay": f_ay["peak"],
        "Mean_az": f_az["mean"], "RMS_az": f_az["rms"], "STD_az": f_az["std"], "Peak_az": f_az["peak"],
        "Mean_current": f_cur["mean"], "RMS_current": f_cur["rms"], "STD_current": f_cur["std"], "Peak_current": f_cur["peak"],

        "Mean_voltage": f_vol["mean"],
        "RMS_voltage":  f_vol["rms"],
        "STD_voltage":  f_vol["std"],
        "Peak_voltage": f_vol["peak"],

        "Mean_temp": mean_temp,
    }


# Danh sách cột FEATURE dùng cho scaler & model (PHẢI TRÙNG KHI TRAIN)
FEATURE_COLS = [
    "Mean_ax","RMS_ax","STD_ax","Peak_ax",
    "Mean_ay","RMS_ay","STD_ay","Peak_ay",
    "Mean_az","RMS_az","STD_az","Peak_az",
    "Mean_current","RMS_current","STD_current","Peak_current",
    "Mean_voltage","Mean_temp",
]
# ✅ Sau khi load scaler
if scaler is not None and hasattr(scaler, "feature_names_in_"):
    FEATURE_COLS = list(scaler.feature_names_in_)
else:
    # fallback nếu scaler không có feature_names_in_
    FEATURE_COLS = [
        "Mean_ax","RMS_ax","STD_ax","Peak_ax",
        "Mean_ay","RMS_ay","STD_ay","Peak_ay",
        "Mean_az","RMS_az","STD_az","Peak_az",
        "Mean_current","RMS_current","STD_current","Peak_current",
        "Mean_voltage","RMS_voltage","STD_voltage","Peak_voltage",
        "Mean_temp",
    ]


# ==================== RULE HẬU XỬ LÝ NHÃN ====================
def post_process_label(raw_label: str, feat: dict, seg_df: pd.DataFrame) -> str:
    """
    raw_label: nhãn model dự đoán (Normal / Misalignment / Electrical / Overheating)
    feat: dict feature (extract_features_from_segment)
    seg_df: giữ tham số để không phải sửa chỗ gọi (nhưng không dùng nữa)
    """
    mean_cur = feat.get("Mean_current", 0.0)
    mean_vol = feat.get("Mean_voltage", 0.0)

    # (tuỳ chọn) Giữ rule "chưa cấp nguồn" -> Electrical
    if mean_vol < 1.0 and mean_cur < 0.05:
        return "Electrical"

    # ✅ Không còn rule nhiệt độ: trả đúng nhãn model
    return raw_label

# ==================== HÀM PREDICT DÙNG MODEL & SCALER ====================
def run_prediction_model(seg_df: pd.DataFrame, model, scaler):
    """
    seg_df: dataframe 1 cửa sổ (vd 512 mẫu) với cột ax, ay, az, current, voltage, temp
    model : RandomForest đã load từ pickle
    scaler: StandardScaler đã load từ pickle
    """

    # 1) Trích 19 feature
    feat = extract_features_from_segment(seg_df)   # dict

    # 2) Tạo DataFrame đúng thứ tự cột
    row = [feat[c] for c in FEATURE_COLS]
    X = pd.DataFrame([row], columns=FEATURE_COLS)

    # 3) Scale + predict
    X_scaled = scaler.transform(X)
    raw_pred = model.predict(X_scaled)[0]

    # 4) Áp rule hậu xử lý (temp, no_power, ..)
    final_label = post_process_label(raw_pred, feat)

    return final_label, raw_pred, feat

# ==================== API: NHẬN DỮ LIỆU TỪ ESP32 (CHỈ LƯU) ====================
@app.post("/upload_sensor", tags=["Sensor"])
async def upload_sensor(data: List[SensorData]):
    if not data:
        return {"status": "no_data"}

    try:
        conn = connected_DB()
        cursor = conn.cursor()
        sql = "INSERT INTO sensor_data (ax, ay, az, current, voltage, temp) VALUES (%s, %s, %s, %s, %s, %s)"
        values = [(d.ax, d.ay, d.az, d.current, d.voltage, d.temp) for d in data]
        cursor.executemany(sql, values)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"ĐÃ LƯU {len(data)} MẪU VÀO DATABASE")
        return {"status": "saved", "rows": len(data)}
    except Exception as e:
        print("LỖI LƯU DỮ LIỆU:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ==================== HÀM DỰ ĐOÁN (HOÀN HẢO - HỖ TRỢ CẢ CHUỖI & SỐ) ====================
async def run_prediction():
    global sensor_buffer
    if len(sensor_buffer) < BUFFER_SIZE or svm_model is None or scaler is None:
        return

    segment = sensor_buffer[-BUFFER_SIZE:]
    df = pd.DataFrame(segment, columns=["ax", "ay", "az", "current", "voltage", "temp"])

    try:
        features = extract_features_from_segment(df)
        X = pd.DataFrame([features])
        X_scaled = scaler.transform(X)

        prediction = svm_model.predict(X_scaled)[0]

        if isinstance(prediction, str):
            fault_name = prediction
        else:
            label_map = {0: "Normal", 1: "Electrical", 2: "Misalignment", 3: "Overheating"}
            fault_name = label_map.get(int(prediction), "Unknown")

        # Confidence (giữ nguyên để debug nếu cần)
        try:
            if hasattr(svm_model, "predict_proba"):
                confidence = float(svm_model.predict_proba(X_scaled)[0].max())
            else:
                confidence = 0.95
        except:
            confidence = 0.95

        print(f"CHẨN ĐOÁN: {fault_name}")

        # ==================== XỬ LÝ CẢNH BÁO & TRẠNG THÁI HIỆN TẠI ====================
        conn = connected_DB()
        c = conn.cursor()

        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            msg = f"PHÁT HIỆN LỖI: {fault_name} - {current_time}"

            if fault_name != "Normal":
                # 1. Lưu vào lịch sử alerts (GIỮ NGUYÊN MÃI MÃI)
                c.execute(
                    "INSERT INTO alerts (device_id, alert, message, status) VALUES (1, %s, %s, 'new')",
                    (fault_name, msg)
                )
                print(f"ĐÃ LƯU VÀO LỊCH SỬ CẢNH BÁO: {fault_name}")

                # 2. Cập nhật trạng thái hiện tại = lỗi mới nhất
                c.execute("""
                    UPDATE status 
                    SET status = %s, message = %s, last_update = NOW()
                    WHERE id = 1
                """, (fault_name, msg))

            else:
                # Chỉ cập nhật trạng thái hiện tại về Bình thường
                # KHÔNG XÓA alerts → lịch sử vẫn còn đầy đủ
                c.execute("""
                    UPDATE status 
                    SET status = 'Normal', 
                        message = 'Hệ thống đang hoạt động bình thường',
                        last_update = NOW()
                    WHERE id = 1
                """)
                print("HỆ THỐNG BÌNH THƯỜNG → ĐÃ CẬP NHẬT TRẠNG THÁI HIỆN TẠI")

            conn.commit()
            print("CẬP NHẬT THÀNH CÔNG: Lịch sử + Trạng thái hiện tại")

        except Exception as e:
            conn.rollback()
            print("LỖI KHI CẬP NHẬT CƠ SỞ DỮ LIỆU:", e)
            traceback.print_exc()
        finally:
            c.close()
            conn.close()
        # ===========================================================================

        # Giữ 50% overlap cho lần dự đoán tiếp theo
        sensor_buffer = sensor_buffer[BUFFER_SIZE // 2:]

    except Exception as e:
        print("LỖI DỰ ĐOÁN:", e)
        traceback.print_exc()

# ==================== TỰ ĐỘNG QUÉT DATABASE MỖI 3 GIÂY ====================
async def auto_predict_task():
    global LAST_CHECKED_ID, sensor_buffer
    print("HỆ THỐNG TỰ ĐỘNG CHẨN ĐOÁN ĐÃ KHỞI ĐỘNG")

    while True:
        try:
            conn = connected_DB()
            c = conn.cursor()
            c.execute("""
                SELECT id, ax, ay, az, current, voltage, temp 
                FROM sensor_data 
                WHERE id > %s 
                ORDER BY id ASC
            """, (LAST_CHECKED_ID,))
            rows = c.fetchall()

            if rows:
                LAST_CHECKED_ID = rows[-1][0]
                new_data = [[float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6])] for r in rows]
                sensor_buffer.extend(new_data)
                print(f"Nhận {len(new_data)} mẫu mới → Tổng buffer: {len(sensor_buffer)} mẫu")

                while len(sensor_buffer) >= BUFFER_SIZE:
                    await run_prediction()

            c.close()
            conn.close()
        except Exception as e:
            print("Lỗi trong auto_predict_task:", e)

        await asyncio.sleep(3)
#------------------------ CẤU HÌNH BẢO MẬT MẬT KHẨU -----------------------
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
# HÀM CHUYỂN MẬT KHẨU
def hash_password(password: str) -> str:
    password = password[:72]  # bcrypt giới hạn tối đa 72 bytes
    return pwd_context.hash(password)
# HÀM SO SANH MẬT KHẨU VỚI DB ĐÃ ĐĂNG KÝ
def verify_password(plant_pass:str,hashed_pass:str):
    return pwd_context.verify(plant_pass,hashed_pass)
# ------------------- HÀM ĐĂNG KÝ USER --------------
@app.post("/Register/", tags=["Login & User Management"])
async def Register_User(request: LoginRequest):
    conn = connected_DB()
    cursor = conn.cursor(dictionary= True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (request.username,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return {"message": "❌ Username đã tồn tại"}

    hashed_pw = hash_password(request.password)
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (request.username, hashed_pw))
    conn.commit()

    cursor.close()
    conn.close()
    return {"message": f" Tạo user {request.username} thành công"}
#----------------- API ĐĂNG NHẬP CHO LOGIN ---------------------
@app.post("/Login/",tags=["Login & User Management"])
async def login_user(request: LoginRequest):
    conn = connected_DB()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM users WHERE username = %s", (request.username,))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if not user:
        return {"message": " Sai username hoặc password"}

    if not verify_password(request.password, user["password"]):
        return {"message": " Sai username hoặc password"}

    return {"message": "✅ Login thành công", "user_id": user["id"], "username": user["username"]}

# ------------- HÀM XÓA TOÀN BỘ TÀI KHOẢN ------------------------
@app.delete("/Delete all account/",tags=["Login & User Management"])
async def delete_all_account():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users")   # Xóa tất cả user trong bảng users
    rows_deleted = cursor.rowcount        # Đếm số user bị xóa
    conn.commit()
    cursor.close()
    conn.close()

    return {"message": f" Đã xóa thành công {rows_deleted} tài khoản"}


#------------------------  API XÓA THÔNG TIN TÀI KHOẢN ---------------
@app.delete("/Delete Account/",tags=["Login & User Management"])
async def delete_login_user(username: str = Query(..., description="Username của user cần xóa")):
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = %s", (username,))
    conn.commit()
    rows_deleted = cursor.rowcount  # số dòng bị xóa
    cursor.close()
    conn.close()

    if rows_deleted == 0:
        return {"message": f"Không tìm thấy user có username = {username}"}
    return {"message": f"✅ Đã xóa thành công user có username = {username}"}

# ------------------ HÀM LOGIN  ------------------

# ------------------- HÀM GÁN NHÃN DỮ LIỆU-----------------

# ------------------ TẠO BẢNG USER ------------------
def create_talble_user():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE,
            password VARCHAR(100)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
# ------------------ TẠO TABLE DATABASE ------------------
def create_table_sensor():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ax FLOAT,
            ay FLOAT,
            az FLOAT,
            current FLOAT,
            voltage FLOAT,       
            temp FLOAT
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
#------------------ TẠO BẢNG CURRENT STATUS ------------------
def create_table_status():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS status(
            id INT PRIMARY KEY DEFAULT 1,
            status VARCHAR(50),          -- 'Normal' hoặc 'Electrical'/'Misalignment'/...
            message TEXT,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """)
    # Đảm bảo luôn có 1 bản ghi
    cursor.execute("""
        INSERT IGNORE INTO status (id, status, message) 
        VALUES (1, 'Normal', 'Hệ thống đang hoạt động bình thường')
    """)
    conn.commit()
    cursor.close()
    conn.close()
#------------------ TẠO BẢNG DEVICE ------------------
def create_table_device():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            id INT AUTO_INCREMENT PRIMARY KEY,
            device_name VARCHAR(100),
            status VARCHAR(50)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
#------------------ TẠO BẢNG CẢNH BÁO ------------------
def create_table_alerts():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            device_id INT NOT NULL,
            alert VARCHAR(100) NOT NULL,
            message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(20) DEFAULT 'new'
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
# ------------------ GỬI DỮ LIỆU LÊN DATABASE  ------------------
@app.post("/PostData/",tags=["Sensor Data"])
async def post_sensor_data(data: SensorData):
    conn = connected_DB() # Kết nối tới database
    cursor = conn.cursor() # con trỏ thực thi câu lệnh
    cursor.execute("""
        INSERT INTO sensor_data (ax, ay, az, current, voltage, temp)
        VALUES (%s, %s, %s, %s,%s, %s)
    """, (data.x, data.y, data.z, data.current, data.voltage, data.temperature))
    conn.commit() # Lưu thay đổi
    cursor.close() # Đóng con trỏ
    conn.close() # đóng kết nối
    return {"message": "Data inserted successfully"}

# -------------- LẤY DỮ LIỆU TỪ DATABASE ----------------
@app.get("/GetData/",tags=["Sensor Data"])
async def get_sensor_data():
    conn = connected_DB() # Kết nối tới database
    cursor = conn.cursor(dictionary=True) # con trỏ thực thi câu lệnh
    cursor.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 100") # Lấy 100 bản ghi mới nhất
    results = cursor.fetchall() # Lấy tất cả kết quả
    cursor.close() # Đóng con trỏ
    conn.close() # đóng kết nối
    return results

# ------------------ CẬP NHẬT DỮ LIỆU  TỪNG PHẦN GỬI LÊN DATABASE ------------------
@app.patch("/UpdateData/{data_id}",tags=["Sensor Data"]) # Cập nhật dữ liệu theo ID
async def update_sensor_data(data_id: int, data: SensorData):
    conn = connected_DB() # Kết nối tới database
    cursor = conn.cursor() # con trỏ thực thi câu lệnh
    cursor.execute("""
        UPDATE sensor_data
        SET ax = %s, ay = %s, az = %s, current = %s, voltage = %s, temp = %s
        WHERE id = %s
    """, (data.x, data.y, data.z, data.current, data.voltage, data.temperature, data_id))
    conn.commit() # Lưu thay đổi
    cursor.close() # Đóng con trỏ
    conn.close() # đóng kết nối
    return {"message": f"Data with id {data_id} updated successfully"}
# -------------- XÓA DỮ LIỆU GỬI LÊN ----------------
@app.delete("/DeleteData/",tags=["Sensor Data"])
async def delete_sensor_data():
    conn = connected_DB() # Kết nối tới database
    cursor = conn.cursor() # con trỏ thực thi câu lệnh
    cursor.execute("DELETE FROM sensor_data") # Xóa tất cả dữ liệu trong bảng
    conn.commit() # Lưu thay đổi
    cursor.close() # Đóng con trỏ
    conn.close() # đóng kết nối
    return {"message": "All data deleted successfully"}
#---------- xóa toàn bộ dữ liệu id ssenssor và đếm lại ---
@app.post("/UpdateID/",tags=["Sensor Data"])
async def reset_sensor_id():
    conn = connected_DB()
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE sensor_data AUTO_INCREMENT = 1")
        conn.commit()
        return {"message": "✅ Đã reset lại ID sensor_data về 1"}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()

# ------------------ API: THÊM THIẾT BỊ ------------------
@app.post("/InsertDevice/",tags=["Device Control"])
async def add_device(device: Device):
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO devices (device_name, status)
        VALUES (%s, %s)
    """, (device.device_name, device.status))
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": "✅ Thiết bị đã được thêm thành công"}

# ------------------ API: LẤY DANH SÁCH THIẾT BỊ ------------------
@app.get("/GetDevice/",tags=["Device Control"])
async def get_devices():
    conn = connected_DB()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM devices ORDER BY id ASC")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results
#---------------------  API: XÓA THIẾT BỊ -----------------
@app.delete("/Xóa thiết bị/{device_id}",tags=["Device Control"])
async def delete_device_by_id(device_id: int = Path(..., description="ID của thiết bị cần xóa")):
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM devices WHERE id = %s", (device_id,))
    conn.commit()
    rows_deleted = cursor.rowcount  # số dòng bị xóa
    cursor.close()
    conn.close()

    if rows_deleted == 0:
        return {"message": f"Không tìm thấy thiết bị có id = {device_id}"}
    return {"message": f"Đã xóa thành công thiết bị có id = {device_id}"}
#------------ XÓA TẤT CẢ THIẾT BỊ -------------------------
@app.delete("/Delete All Device/",tags=["Device Control"])
async def delete_all_device():
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM devices")
    rows_deleted = cursor.rowcount  # số dòng bị xóa
    conn.commit()
    cursor.close()
    conn.close()

    return {"message": f" Đã xóa thành công {rows_deleted} thiết bị"}

# ------------------ API: CẬP NHẬT TRẠNG THÁI ------------------
@app.put("/Update Status Device/{device_id}",tags=["Device Control"])
async def update_device_status(device_id: int, status: str):
    if status not in ["online", "offline"]:
        raise HTTPException(status_code=400, detail="❌ Trạng thái không hợp lệ")
    conn = connected_DB()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE devices SET status = %s WHERE id = %s
    """, (status, device_id))
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": f" Trạng thái thiết bị {device_id} đã được cập nhật thành {status}"}
#---------- Reset lại ID của bảng devices ----------
@app.post("/ResetDeviceID/", tags=["Device Control"])
async def reset_device_id():
    conn = connected_DB()
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE devices AUTO_INCREMENT = 1")
        conn.commit()
        return {"message": "✅ Đã reset lại ID của bảng devices về 1"}
    except Exception as e:
        conn.rollback()
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()

# ------------------ API: TẠO CẢNH BÁO ------------------
# 1. Gửi cảnh báo (INSERT)
@app.post("/PostAlert/",tags=["Alerts & Notifications"], response_model=dict)
def create_alert(alert: Alert):
    conn = connected_DB()
    cursor = conn.cursor()

    try:
        sql = """
        INSERT INTO alerts (device_id, alert, message, status)
        VALUES (%s, %s, %s, %s)
        """
        values = (alert.device_id, alert.alert, alert.message, alert.status)
        cursor.execute(sql, values)
        conn.commit()
        return {"message": " Cảnh báo đã được lưu thành công!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()
# 2. Lấy danh sách cảnh báo (SELECT)
@app.get("/GetAlert/", response_model=List[Alert],tags=["Alerts & Notifications"])
def get_alerts():
    conn = connected_DB()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
        alerts = cursor.fetchall()
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()
@app.delete("/DeleteAllAlerts/", tags=["Alerts & Notifications"])
def delete_all_alerts():
    conn = connected_DB()
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM alerts")
        rows_deleted = cursor.rowcount   # số lượng dòng đã xoá
        conn.commit()

        return {
            "status": "success",
            "deleted_rows": rows_deleted,
            "message": "Đã xoá toàn bộ cảnh báo."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()
        # ---------- Reset lại ID của bảng alerts ----------
@app.post("/ResetAlertID/", tags=["Alerts & Notifications"])
async def reset_alert_id():
    conn = connected_DB()
    cursor = conn.cursor()
    try:
        # Xóa toàn bộ dữ liệu trong bảng alerts trước khi reset ID
        cursor.execute("DELETE FROM alerts")
        
        # Reset giá trị AUTO_INCREMENT về 1
        cursor.execute("ALTER TABLE alerts AUTO_INCREMENT = 1")
        
        conn.commit()
        return {
            "message": "✅ Đã xóa toàn bộ cảnh báo và reset ID alerts về 1 thành công!"
        }
    except Exception as e:
        conn.rollback()
        return {"error": f"Lỗi khi reset ID alerts: {str(e)}"}
    finally:
        cursor.close()
        conn.close()
@app.get("/GetStatus/", tags=["Alerts & Notifications"])
def get_status():
    conn = connected_DB()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT status, message FROM status WHERE id = 1")
        result = cursor.fetchone()
        
        # Nếu chưa có bản ghi nào (hiếm xảy ra), trả về mặc định Normal
        if result is None:
            return {
                "status": "Normal",
                "message": "Hệ thống đang hoạt động bình thường"
            }
        return result
    except Exception as e:
        print("Lỗi lấy status:", e)
        raise HTTPException(status_code=500, detail="Lỗi server")
    finally:
        cursor.close()
        conn.close()
@app.on_event("startup")
def startup_event():
    create_table_sensor()
    create_talble_user()
    create_table_device()
    create_table_alerts()
    create_table_status()
    # Tạo device mặc định
    conn = connected_DB()
    c = conn.cursor()
    c.execute("INSERT IGNORE INTO devices (id, device_name, status) VALUES (1, 'Động cơ chính', 'online')")
    conn.commit()
    c.close()
    conn.close()

    # Reset buffer & ID
    global sensor_buffer, LAST_CHECKED_ID
    sensor_buffer = []
    LAST_CHECKED_ID = 0

    # Bắt đầu tự động chẩn đoán 
    asyncio.create_task(auto_predict_task()) 
    print("HỆ THỐNG ĐÃ SẴN SÀNG - TỰ ĐỘNG CHẨN ĐOÁN MỖI 3 GIÂY!")
