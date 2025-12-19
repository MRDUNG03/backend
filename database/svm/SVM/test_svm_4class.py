import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# In tiếng Việt trên Windows
sys.stdout.reconfigure(encoding="utf-8")

# ============== CẤU HÌNH ==============
# 1) Model SVM đã train và lưu dạng .pkl
MODEL_PKL_PATH = r"D:\SVM\svm_model_from_raw.pkl"

# 2) File đặc trưng 4 lớp dùng để train (để lấy đúng danh sách feature)
FEATURE_CSV_PATH = r"D:\Data\features_4class.csv"

# 3) File RAW cần test (mỗi lần đổi 1 file)
#    Ví dụ:
#    - Electrical:   r"D:\Data\data_electrical_1\Electrical_clean.csv"
#    - Overheating:  r"D:\Data\data_overheating_1\Overheating1.csv"
#    - Misalignment: r"D:\Data\data_misalignment_1\Misalignment_clean.csv"
#    - Normal:       r"D:\Data\data_normal_1\NORMAL_1.csv"
TEST_RAW_CSV     = r"D:\SVM\Misalignment_clean.csv"
# Số cửa sổ chia mỗi file
N_SEGMENTS       = 10
# ======================================


def load_feature_cols():
    """Lấy danh sách feature đã dùng để train từ features_4class.csv"""
    if not os.path.exists(FEATURE_CSV_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {FEATURE_CSV_PATH}")

    df = pd.read_csv(FEATURE_CSV_PATH)

    # bỏ các cột không phải feature (giống khi train)
    drop_cols = ["label", "window_index", "start_time"]
    X = df.drop(columns=drop_cols, errors="ignore")

    # chỉ giữ cột số
    X = X.select_dtypes(include=[np.number])

    feature_cols = X.columns.tolist()
    print("📋 Feature dùng để train (đúng thứ tự):")
    print(feature_cols, "\n")
    return feature_cols


def extract_features_from_raw(raw_csv_path, feature_cols, n_segments=10):
    """
    Đọc file raw (ax, ay, az, current, voltage, temp, label),
    chia thành n_segments cửa sổ đều nhau,
    trích:
      Mean/RMS/STD/Peak cho ax, ay, az, current, voltage
      Mean_temp
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file raw: {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)
    print(f"📥 Đã đọc file: {raw_csv_path}")
    print("   Các cột:", df.columns.tolist())

    required_cols = ["ax", "ay", "az", "current", "voltage", "temp"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"❌ File {raw_csv_path} thiếu cột '{c}'.")

    n_samples = len(df)
    n_per_seg = n_samples // n_segments
    n_cut = n_per_seg * n_segments
    df = df.iloc[:n_cut].reset_index(drop=True)

    print(f"   Tổng mẫu: {n_samples} → dùng {n_cut}, mỗi cửa sổ: {n_per_seg} mẫu")

    # Lấy nhãn thật nếu có cột label, còn không thì cho Unknown
    if "label" in df.columns:
        true_label = str(df["label"].iloc[0])
    else:
        true_label = "Unknown"

    rows = []

    def feat_1d(x):
        x = x.astype(float).values
        return dict(
            mean=float(np.mean(x)),
            rms=float(np.sqrt(np.mean(x**2))),
            std=float(np.std(x)),
            peak=float(np.max(np.abs(x))),
        )

    for i in range(n_segments):
        start = i * n_per_seg
        end   = (i + 1) * n_per_seg
        seg   = df.iloc[start:end]

        f_ax  = feat_1d(seg["ax"])
        f_ay  = feat_1d(seg["ay"])
        f_az  = feat_1d(seg["az"])
        f_cur = feat_1d(seg["current"])
        f_vol = feat_1d(seg["voltage"])
        mean_temp = float(np.mean(seg["temp"].astype(float).values))

        row = {
            "window_index": i,  # index từ 0
            "Mean_ax": f_ax["mean"],
            "RMS_ax":  f_ax["rms"],
            "STD_ax":  f_ax["std"],
            "Peak_ax": f_ax["peak"],

            "Mean_ay": f_ay["mean"],
            "RMS_ay":  f_ay["rms"],
            "STD_ay":  f_ay["std"],
            "Peak_ay": f_ay["peak"],

            "Mean_az": f_az["mean"],
            "RMS_az":  f_az["rms"],
            "STD_az":  f_az["std"],
            "Peak_az": f_az["peak"],

            "Mean_current": f_cur["mean"],
            "RMS_current":  f_cur["rms"],
            "STD_current":  f_cur["std"],
            "Peak_current": f_cur["peak"],

            "Mean_voltage": f_vol["mean"],
            "RMS_voltage":  f_vol["rms"],
            "STD_voltage":  f_vol["std"],
            "Peak_voltage": f_vol["peak"],

            "Mean_temp": mean_temp,
            "label": true_label,
        }

        rows.append(row)

    df_feat = pd.DataFrame(rows)

    # Đảm bảo có đủ cột feature, đúng thứ tự
    for c in feature_cols:
        if c not in df_feat.columns:
            df_feat[c] = 0.0

    df_feat = df_feat[["window_index"] + feature_cols + ["label"]]
    return df_feat


def main():
    # 1. Load model
    if not os.path.exists(MODEL_PKL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy model .pkl: {MODEL_PKL_PATH}")

    print(f"✅ Đang load model từ: {MODEL_PKL_PATH}")
    with open(MODEL_PKL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Đã load model.\n")

    # 2. Lấy danh sách feature
    feature_cols = load_feature_cols()

    # 3. Trích đặc trưng từ file RAW
    print(f"🔎 Đang trích đặc trưng từ file: {TEST_RAW_CSV}")
    df_feat = extract_features_from_raw(TEST_RAW_CSV, feature_cols, n_segments=N_SEGMENTS)

    print("\n📊 5 dòng đầu đặc trưng:")
    print(df_feat.head())
    print("\nKích thước đặc trưng:", df_feat.shape, "\n")

    X_test = df_feat[feature_cols]
    y_true = df_feat["label"]

    # 4. Dự đoán
    y_pred = model.predict(X_test)

    print("=== KẾT QUẢ TỪNG CỬA SỔ ===")
    for i in range(len(X_test)):
        win = int(df_feat.loc[i, "window_index"])
        print(f"Window {win:03d}: thực tế = {y_true.iloc[i]:12s} | dự đoán = {y_pred[i]:12s}")
    print("================================\n")

    # 5. Độ chính xác trên riêng file này
    acc = accuracy_score(y_true, y_pred) * 100.0
    print(f"🎯 Độ chính xác (accuracy) cho file này: {acc:.2f}%\n")

    print("=== Classification report ===")
    print(classification_report(y_true, y_pred))

    labels_sorted = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    print("=== Confusion matrix ===")
    print("Labels:", labels_sorted)
    print(cm)

    # 6. Xác suất từng lớp (nếu model hỗ trợ)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)   # (n_windows, n_classes)
        classes = model.classes_

        # Trung bình % trên toàn bộ cửa sổ → giống kiểu bạn muốn
        mean_proba = proba.mean(axis=0)
        print("\n🔍 Xác suất từng lớp (trung bình cho CẢ FILE):")
        order = np.argsort(mean_proba)[::-1]
        for idx in order:
            cls = classes[idx]
            pct = mean_proba[idx] * 100
            print(f"- {cls}: {pct:.2f}%")

        # Nếu muốn xem ví dụ 3 cửa sổ đầu:
        print("\n🔎 Ví dụ xác suất 3 cửa sổ đầu:")
        for i in range(min(3, len(X_test))):
            print(f"  Window {int(df_feat.loc[i, 'window_index']):03d}:")
            p = proba[i]
            order_i = np.argsort(p)[::-1]
            for idx in order_i:
                cls = classes[idx]
                pct = p[idx] * 100
                print(f"    - {cls}: {pct:.2f}%")
    else:
        print("\n⚠️ Model không hỗ trợ predict_proba (cần probability=True khi train).")


if __name__ == "__main__":
    main()
