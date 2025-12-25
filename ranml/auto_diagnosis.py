# ml/auto_diagnosis.py
import asyncio
from model.connected_DB import connected_DB
from .predictor import predict

BUFFER_SIZE = 6000
sensor_buffer = []          # Buffer bắt đầu rỗng
last_id = 0                 # Sẽ được cập nhật khi khởi động

async def auto_diagnosis_task():
    global sensor_buffer, last_id

    # ===== BƯỚC 1: Khi khởi động, lấy ID lớn nhất hiện tại trong DB =====
    try:
        conn = connected_DB()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM sensor_data")
        max_id_result = cursor.fetchone()
        last_id = max_id_result[0] if max_id_result[0] is not None else 0
        cursor.close()
        conn.close()
        print(f"🤖 Hệ thống chẩn đoán khởi động - Bỏ qua dữ liệu cũ đến ID = {last_id}")
        print("   Chỉ xử lý dữ liệu MỚI từ thiết bị gửi lên sau thời điểm này!")
    except Exception as e:
        print(f"❌ Lỗi khi lấy MAX ID: {e}")
        last_id = 0

    print("🤖 Bắt đầu theo dõi dữ liệu mới realtime...")

    # ===== BƯỚC 2: Vòng lặp theo dõi dữ liệu mới =====
    while True:
        try:
            conn = connected_DB()
            cursor = conn.cursor()

            # Chỉ lấy dữ liệu MỚI (id > last_id)
            cursor.execute("""
                SELECT id, ax, ay, az, current, voltage, temp 
                FROM sensor_data 
                WHERE id > %s 
                ORDER BY id ASC
            """, (last_id,))

            rows = cursor.fetchall()

            if rows:
                # Cập nhật last_id thành id lớn nhất vừa lấy
                last_id = rows[-1][0]

                # Thêm dữ liệu mới vào buffer
                new_samples = [[float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6])] for r in rows]
                sensor_buffer.extend(new_samples)

                print(f"📈 Nhận {len(new_samples)} mẫu MỚI → Buffer hiện tại: {len(sensor_buffer)} mẫu")

                # Khi buffer đủ 10000 mẫu → chẩn đoán
                while len(sensor_buffer) >= BUFFER_SIZE:
                    segment = sensor_buffer[:BUFFER_SIZE]
                    result = predict(segment)
                    print(f"🔥 KẾT QUẢ CHẨN ĐOÁN (dữ liệu mới): {result}")

                    # Cập nhật bảng status
                    update_cur = conn.cursor()
                    message = "Hệ thống bình thường" if result == "Normal" else f"PHÁT HIỆN LỖI: {result}"
                    update_cur.execute("""
                        UPDATE status SET status = %s, message = %s WHERE id = 1
                    """, (result, message))
                    conn.commit()
                    update_cur.close()

                    # Overlap 50% để không bỏ sót
                    sensor_buffer = sensor_buffer[BUFFER_SIZE // 2:]

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Lỗi trong auto diagnosis: {e}")

        await asyncio.sleep(3)  # Kiểm tra giây