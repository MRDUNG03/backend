import pandas as pd

IN_CSV = r"D:\Data_1\all_features_feat.csv"
OUT_CSV = r"D:\SVM\all_features_backend_style.csv"

df = pd.read_csv(IN_CSV)
print("📂 Đã đọc:", IN_CSV)
print("Kích thước ban đầu:", df.shape)

# 1) Lấy đúng 18 feature backend đang dùng
keep_cols = [
    "Mean_ax","RMS_ax","STD_ax","Peak_ax",
    "Mean_ay","RMS_ay","STD_ay","Peak_ay",
    "Mean_az","RMS_az","STD_az","Peak_az",
    "Mean_current","RMS_current","STD_current","Peak_current",
    "Mean_voltage","Mean_temp",
    "label","session_id"
]

df_new = df[keep_cols]

print("📌 Kích thước sau khi lọc backend feature:", df_new.shape)
print("Cột giữ lại:", df_new.columns.tolist())

# 2) Lưu file mới
df_new.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("💾 Đã tạo file backend feature tại:", OUT_CSV)
