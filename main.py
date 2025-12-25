from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
# ==================== IMPORT CÁC HÀM TẠO BẢNG TỪ FOLDER table/ ====================

from table.user import create_table_user
from table.sensordata import create_table_sensor
from table.device import create_table_device
from table.status import create_table_status
from table.alert import create_table_alerts  # Đảm bảo tên hàm đúng là create_table_alerts
# ==================== IMPORT CÁC SERVICE TỪ FOLDER service/ ====================
from service.alert import alert_api
from service.data import sensor_api
from service.register_login import user_api
from service.device import device_api
# ==================== IMPORT TASK CHẨN ĐOÁN TỰ ĐỘNG (ML) ====================
# Nếu bạn đã tạo folder ml/ và file auto_diagnosis.py
from ranml.auto_diagnosis import auto_diagnosis_task
app = FastAPI(title="GIÁM SÁT TRẠNG THÁI ĐỘNG CƠ ĐIỆN ",root_path="/api",docs_url="/docs",description="API for monitoring electric motor status")

# ==================== CHO PHÉP CORS (Frontend truy cập được) ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # Sau này thay bằng domain frontend thực tế
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== TẠO CÁC BẢNG KHI ỨNG DỤNG KHỞI ĐỘNG ====================
@app.on_event("startup")
def startup_event():
    print("🚀 Đang khởi động ứng dụng... Tạo các bảng database nếu chưa tồn tại")

    create_table_user()
    create_table_sensor()
    create_table_device()
    create_table_status()
    create_table_alerts()
#api
    alert_api(app)
    sensor_api(app)
    user_api(app)
    device_api(app)
    asyncio.create_task(auto_diagnosis_task())
    print("🤖 Hệ thống chẩn đoán lỗi tự động đã được kích hoạt!")


