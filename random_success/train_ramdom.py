import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.stdout.reconfigure(encoding="utf-8")

# =========================
# CẤU HÌNH
# =========================
TRAIN_CSV   = r"C:\Users\ThanhDat\Downloads\backup_3\all_features_train_feat_clean4.csv"   # ✅ file feature đã có
MODEL_OUT   = r"C:\Users\ThanhDat\Downloads\backup_3\rf_model.pkl"
SCALER_OUT  = r"C:\Users\ThanhDat\Downloads\backup_3\rf_scaler.pkl"

# 21 feature giống backend
FEATURE_COLS = [
    "Mean_ax","RMS_ax","STD_ax","Peak_ax",
    "Mean_ay","RMS_ay","STD_ay","Peak_ay",
    "Mean_az","RMS_az","STD_az","Peak_az",
    "Mean_current","RMS_current","STD_current","Peak_current",
    "Mean_voltage","RMS_voltage","STD_voltage","Peak_voltage",
    "Mean_temp",
]

# nếu file có các cột meta này thì bỏ
META_COLS = ["raw_label", "session_id", "window_index", "start_sample", "start_time", "start_time_sec"]

def main():
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"❌ Không tìm thấy TRAIN_CSV: {TRAIN_CSV}")

    print(f"📥 Load feature csv: {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    print("📊 Shape:", df.shape)

    if "label" not in df.columns:
        raise ValueError("❌ File train thiếu cột 'label'.")

    # Bỏ meta nếu có
    drop_cols = [c for c in META_COLS if c in df.columns]
    if drop_cols:
        print("🧹 Drop meta cols:", drop_cols)
        df = df.drop(columns=drop_cols)

    # Check đủ 21 feature
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Thiếu feature: {missing}")

    # X, y
    X = df[FEATURE_COLS].copy()
    y = df["label"].astype(str).copy()

    # ép numeric + xử lý NaN
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(numeric_only=True)).fillna(0.0)

    print("\n📌 Phân bố nhãn:")
    print(y.value_counts())

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # train RF
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("\n🌲 Đang train RandomForest...")
    model.fit(X_train_sc, y_train)
    print("✔ Train xong.")

    # evaluate
    y_pred = model.predict(X_test_sc)
    print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Report:\n", classification_report(y_test, y_pred))
    print("\n🧩 Confusion:\n", confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

    # save
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)

    print("\n💾 Saved model :", MODEL_OUT)
    print("💾 Saved scaler:", SCALER_OUT)

    print("\n📋 FEATURE_COLS (phải trùng backend):")
    print(FEATURE_COLS)

if __name__ == "__main__":
    main()
