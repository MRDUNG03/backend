# train_rf_backend_features.py
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================
IN_CSV  = r"D:\SVM\all_features_backend_style.csv"
OUT_DIR = r"D:\SVM\model_rf_backend_style"

os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH  = os.path.join(OUT_DIR, "model_rf_backend.pkl")
SCALER_PATH = os.path.join(OUT_DIR, "scaler_rf_backend.pkl")

# ==========================
# 2. ĐỌC DỮ LIỆU
# ==========================
df = pd.read_csv(IN_CSV)
print(f"📂 Đã đọc: {IN_CSV}")
print("Kích thước:", df.shape)
print("Cột:", df.columns.tolist())

# kiểm tra label & session_id
for col in ["label", "session_id"]:
    if col not in df.columns:
        raise ValueError(f"Thiếu cột {col} trong file feature!")

# 👉 LẤY DANH SÁCH CỘT FEATURE:
#    tất cả cột trừ 'label' và 'session_id'
FEATURE_COLS = [c for c in df.columns if c not in ["label", "session_id"]]

print("🎯 FEATURE_COLS dùng để train:")
print(FEATURE_COLS)

X_all = df[FEATURE_COLS]
y_all = df["label"].values
sessions = df["session_id"].unique()

print("🔎 Các session:", sessions)

# ==========================
# 3. HÀM TẠO RANDOM FOREST
# ==========================
def make_rf():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

# ==========================
# 4. LEAVE-ONE-SESSION-OUT
# ==========================
train_accs = []
test_accs  = []

for test_sess in sessions:
    print("\n===========================")
    print(f"🧪 TEST SESSION: {test_sess}")
    print("===========================")

    is_test  = (df["session_id"] == test_sess)
    is_train = ~is_test

    X_train = df.loc[is_train, FEATURE_COLS]
    y_train = df.loc[is_train, "label"].values
    X_test  = df.loc[is_test,  FEATURE_COLS]
    y_test  = df.loc[is_test,  "label"].values

    # chuẩn hóa
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # train RF
    clf = make_rf()
    clf.fit(X_train_s, y_train)

    # đánh giá từng session
    y_tr_pred = clf.predict(X_train_s)
    y_te_pred = clf.predict(X_test_s)

    acc_tr = accuracy_score(y_train, y_tr_pred)
    acc_te = accuracy_score(y_test,  y_te_pred)
    train_accs.append(acc_tr)
    test_accs.append(acc_te)

    print(f"Train acc: {acc_tr:.4f}")
    print(f"Test  acc: {acc_te:.4f}")
    print("📜 Report (test):")
    print(classification_report(y_test, y_te_pred))
    print("🔁 Confusion matrix:")
    print(confusion_matrix(y_test, y_te_pred))

print("\n===== TỔNG KẾT LOSO (backend features) =====")
for sess, tr, te in zip(sessions, train_accs, test_accs):
    print(f"Session {sess}: train={tr:.4f}, test={te:.4f}")

print(f"\n👉 Mean train acc: {np.mean(train_accs):.4f}")
print(f"👉 Mean test  acc: {np.mean(test_accs):.4f}")

# ==========================
# 5. TRAIN FINAL MODEL
# ==========================
print("\n🚀 Train FINAL RF trên toàn bộ dữ liệu với backend-feature...")

scaler_final = StandardScaler()
scaler_final.fit(X_all)
X_all_s = scaler_final.transform(X_all)

rf_final = make_rf()
rf_final.fit(X_all_s, y_all)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(rf_final, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler_final, f)

print(f"\n💾 Đã lưu model tại : {MODEL_PATH}")
print(f"💾 Đã lưu scaler tại: {SCALER_PATH}")
print("🎉 Train RF với đặc trưng backend xong!")
