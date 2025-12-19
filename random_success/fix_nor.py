import pandas as pd
import os
import sys

# Cho in tiếng Việt
sys.stdout.reconfigure(encoding="utf-8")

# ===================== CẤU HÌNH =====================
INPUT_CSV  = r"C:\Users\ThanhDat\Downloads\New_Data\New_Data\normal\Normalnew1.csv"
OUTPUT_CSV = r"C:\Users\ThanhDat\Downloads\New_Data\New_Data\normal\Normalnew1_fixed.csv"

CUT_FROM_ROW = 459033          # từ dòng này trở xuống => xoá
TEMP_START   = 458633          # bắt đầu sửa temp
TEMP_END     = 458858          # kết thúc sửa temp (inclusive)
FIX_TEMP     = 49.99
# ====================================================

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {INPUT_CSV}")

    print(f"📥 Đang đọc file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if "temp" not in df.columns:
        raise ValueError("❌ File không có cột 'temp'")

    total_before = len(df)
    print(f"📊 Tổng số dòng ban đầu: {total_before}")

    # ========= 1. CẮT BỎ TỪ DÒNG 459033 TRỞ XUỐNG =========
    df = df.iloc[:CUT_FROM_ROW].reset_index(drop=True)
    print(f"✂️ Sau khi cắt: còn {len(df)} dòng")

    # ========= 2. SỬA TEMP = 49.99 CHO ĐOẠN CHỈ ĐỊNH =========
    if TEMP_END >= len(df):
        raise ValueError("❌ TEMP_END vượt quá số dòng sau khi cắt")

    df.loc[TEMP_START:TEMP_END, "temp"] = FIX_TEMP

    print(
        f"✏️ Đã sửa cột temp = {FIX_TEMP} "
        f"cho các dòng [{TEMP_START} → {TEMP_END}]"
    )

    # ========= 3. LƯU FILE =========
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n💾 Đã lưu file mới tại: {OUTPUT_CSV}")

    print("\n📍 5 dòng kiểm tra quanh vùng sửa:")
    print(df.loc[TEMP_START-2:TEMP_START+2, ["temp"]])
    print(df.loc[TEMP_END-2:TEMP_END+2, ["temp"]])

if __name__ == "__main__":
    main()
