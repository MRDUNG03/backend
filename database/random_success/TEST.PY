import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

MODEL_PATH  = r"C:\Users\ThanhDat\Downloads\train_nor_elc\rf_model.pkl"
SCALER_PATH = r"C:\Users\ThanhDat\Downloads\train_nor_elc\rf_scaler.pkl"


TEST_RAW_CSV = r"C:\Users\ThanhDat\Downloads\data-overheating\Overheating_1.csv"   # ✅ đổi file để test riêng
WINDOW = 512

FEATURE_COLS = [
    "Mean_ax","RMS_ax","STD_ax","Peak_ax",
    "Mean_ay","RMS_ay","STD_ay","Peak_ay",
    "Mean_az","RMS_az","STD_az","Peak_az",
    "Mean_current","RMS_current","STD_current","Peak_current",
    "Mean_voltage","RMS_voltage","STD_voltage","Peak_voltage",
    "Mean_temp",
]

REQUIRED_COLS = ["ax","ay","az","current","voltage","temp"]

label_vi = {
    "Normal": "Bình thường",
    "Electrical": "Lỗi điện",
    "Misalignment": "Lệch trục",
    "Overheating": "Quá nhiệt",
}

def feat_1d(arr):
    arr = np.asarray(arr, dtype=float)
    return {
        "mean": float(arr.mean()),
        "rms":  float(np.sqrt(np.mean(arr**2))),
        "std":  float(arr.std(ddof=0)),
        "peak": float(np.max(np.abs(arr))),
    }

def extract_backend_features(seg_df: pd.DataFrame) -> dict:
    f_ax  = feat_1d(seg_df["ax"].values)
    f_ay  = feat_1d(seg_df["ay"].values)
    f_az  = feat_1d(seg_df["az"].values)
    f_cur = feat_1d(seg_df["current"].values)
    f_vol = feat_1d(seg_df["voltage"].values)
    mean_temp = float(np.mean(seg_df["temp"].values.astype(float)))

    return {
        "Mean_ax": f_ax["mean"], "RMS_ax": f_ax["rms"], "STD_ax": f_ax["std"], "Peak_ax": f_ax["peak"],
        "Mean_ay": f_ay["mean"], "RMS_ay": f_ay["rms"], "STD_ay": f_ay["std"], "Peak_ay": f_ay["peak"],
        "Mean_az": f_az["mean"], "RMS_az": f_az["rms"], "STD_az": f_az["std"], "Peak_az": f_az["peak"],

        "Mean_current": f_cur["mean"], "RMS_current": f_cur["rms"],
        "STD_current": f_cur["std"], "Peak_current": f_cur["peak"],

        "Mean_voltage": f_vol["mean"], "RMS_voltage": f_vol["rms"],
        "STD_voltage": f_vol["std"], "Peak_voltage": f_vol["peak"],

        "Mean_temp": mean_temp,
    }

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Không thấy model: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"❌ Không thấy scaler: {SCALER_PATH}")
    if not os.path.exists(TEST_RAW_CSV):
        raise FileNotFoundError(f"❌ Không thấy file test: {TEST_RAW_CSV}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(TEST_RAW_CSV)
    print(f"📥 Test file: {TEST_RAW_CSV}")
    print(f"🔢 Số mẫu: {len(df)} | window={WINDOW}")

    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"❌ Thiếu cột: {miss}")

    feats = []
    for start in range(0, len(df), WINDOW):
        seg = df.iloc[start:start+WINDOW]
        if len(seg) < WINDOW:
            continue
        feats.append(extract_backend_features(seg))

    X = pd.DataFrame(feats)[FEATURE_COLS]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(numeric_only=True)).fillna(0.0)

    X_sc = scaler.transform(X)

    probs = model.predict_proba(X_sc)          # (n_windows, n_classes)
    classes = model.classes_

    avg_probs = probs.mean(axis=0)             # trung bình % toàn file
    best_idx = int(np.argmax(avg_probs))
    best_label = str(classes[best_idx])

    print("\n🧠 KẾT QUẢ DỰ ĐOÁN (trung bình % trên toàn file):")
    print(f"➡️ {best_label} ({label_vi.get(best_label, best_label)})\n")

    for cls, p in sorted(zip(classes, avg_probs), key=lambda x: x[1], reverse=True):
        print(f"  - {cls:12s}: {p*100:6.2f}%  ({label_vi.get(cls, cls)})")

    print(f"\n📌 Số cửa sổ dùng test: {len(X)}")

if __name__ == "__main__":
    main()
