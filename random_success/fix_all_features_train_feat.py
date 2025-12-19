import pandas as pd
import sys
import os

# Cho phép in tiếng Việt
sys.stdout.reconfigure(encoding="utf-8")

# ================= CẤU HÌNH =================
INPUT_CSV  = r"C:\Users\ThanhDat\Downloads\backup_\all_features_train_feat_clean2.csv"
OUTPUT_CSV = r"C:\Users\ThanhDat\Downloads\backup_\all_features_train_feat_clean3.csv"

SESSION_NAME = "Normalnew1"
TEMP_TH = 50.0
# ============================================

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError("❌ Không tìm thấy file input")

    print(f"📥 Đang đọc file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    print("📊 Tổng số dòng ban đầu:", len(df))

    # Kiểm tra cột bắt buộc
    for col in ["session_id", "Mean_temp"]:
        if col not in df.columns:
            raise ValueError(f"❌ Thiếu cột bắt buộc: {col}")

    # Điều kiện xoá
    mask_delete = (
        (df["session_id"] == SESSION_NAME) &
        (df["Mean_temp"] >= TEMP_TH)
    )

    print(f"🧹 Số dòng bị xoá (session_id={SESSION_NAME}, temp>=50):",
          mask_delete.sum())

    # Xoá dòng
    df_clean = df.loc[~mask_delete].reset_index(drop=True)

    # Lưu file
    df_clean.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("📊 Số dòng sau khi xoá:", len(df_clean))
    print(f"💾 Đã lưu file tại: {OUTPUT_CSV}")

    # Xem nhanh vài dòng Normalnew1 còn lại
    print("\n📍 Ví dụ Normalnew1 còn lại:")
    print(df_clean[df_clean["session_id"] == SESSION_NAME][
        ["session_id", "Mean_temp"]
    ].head())

if __name__ == "__main__":
    main()
