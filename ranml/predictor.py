# ml/predictor.py
import joblib
import pandas as pd
from .feature_extractor import extract_features_from_buffer
from pathlib import Path

# Đường dẫn đúng đến folder random_success (đã tồn tại)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "random_success" / "rf_model_time_fft.pkl"
SCALER_PATH = BASE_DIR / "random_success" / "rf_scaler_time_fft.pkl"

print("=" * 60)
print("🔍 KIỂM TRA MODEL & SCALER")
print(f"Thư mục gốc: {BASE_DIR}")
print(f"Model: {MODEL_PATH}")
print(f"Scaler: {SCALER_PATH}")

if MODEL_PATH.exists():
    print(f"✅ Tìm thấy model: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size} bytes)")
else:
    print("❌ Không tìm thấy file model!")

if SCALER_PATH.exists():
    print(f"✅ Tìm thấy scaler: {SCALER_PATH.name} ({SCALER_PATH.stat().st_size} bytes)")
else:
    print("❌ Không tìm thấy file scaler!")

print("=" * 60)

model = None
scaler = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ ĐÃ TẢI THÀNH CÔNG MODEL VÀ SCALER!")
except Exception as e:
    print(f"❌ LỖI KHI LOAD MODEL/SCALER: {e}")
    import traceback
    traceback.print_exc()
    model = scaler = None

# FEATURE_COLUMNS = [
#     "Mean_ax", "RMS_ax", "STD_ax", "Peak_ax",
#     "Mean_ay", "RMS_ay", "STD_ay", "Peak_ay",
#     "Mean_az", "RMS_az", "STD_az", "Peak_az",
#     "Mean_current", "RMS_current", "STD_current", "Peak_current",
#     "Mean_voltage", "RMS_voltage", "STD_voltage", "Peak_voltage",
#     "Mean_temp"
# ]
FEATURE_COLUMNS = [
    # ===== Miền thời gian =====
    "Mean_ax", "RMS_ax", "STD_ax", "Peak_ax",
    "Mean_ay", "RMS_ay", "STD_ay", "Peak_ay",
    "Mean_az", "RMS_az", "STD_az", "Peak_az",
    "Mean_current", "RMS_current", "STD_current", "Peak_current",
    "Mean_voltage", "RMS_voltage", "STD_voltage", "Peak_voltage",

    # ===== Miền tần số (FFT) =====
    "PeakFreq_ax", "SpecEnergy_ax",
    "PeakFreq_ay", "SpecEnergy_ay",
    "PeakFreq_az", "SpecEnergy_az",
    "PeakFreq_current", "SpecEnergy_current",
    "PeakFreq_voltage", "SpecEnergy_voltage",

    # ===== Nhiệt độ =====
    "Mean_temp"
]
def predict(data_buffer: list) -> str:
    if model is None or scaler is None:
        return "Model_Error"

    features = extract_features_from_buffer(data_buffer)
    if not features:
        return "No_Data"

    try:
        X = pd.DataFrame([{
            col: features.get(col, 0.0) for col in FEATURE_COLUMNS
        }])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        return str(prediction)
    except Exception as e:
        print(f"❌ Lỗi dự đoán: {e}")
        return "Predict_Error"