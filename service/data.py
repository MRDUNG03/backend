from fastapi import HTTPException, Path
from typing import Union, List, Optional
from model.connected_DB import connected_DB
from model.SensorData import SensorData
import traceback

app = None

def sensor_api(fastapi_app):
    global app
    app = fastapi_app

    # ---------------- GET DATA ----------------
    @app.get("/getdata/", tags=["Sensor Data"])
    async def get_sensor_data(limit: int = 1):
        conn = connected_DB()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT %s", (limit,))
            results = cursor.fetchall()
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ---------------- UPDATE DATA ----------------
    @app.patch("/updatedata/{data_id}", tags=["Sensor Data"])
    async def update_sensor_data(data_id: int = Path(..., description="ID của dữ liệu"), data: SensorData = None):
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE sensor_data
                SET ax=%s, ay=%s, az=%s, current=%s, voltage=%s, temp=%s
                WHERE id=%s
            """, (data.ax, data.ay, data.az, data.current, data.voltage, data.temp, data_id))
            conn.commit()
            if cursor.rowcount == 0:
                return {"message": f"Không tìm thấy dữ liệu có id = {data_id}"}
            return {"message": f"✅ Data id={data_id} đã được cập nhật"}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ---------------- DELETE DATA ----------------
    @app.delete("/deletedata/", tags=["Sensor Data"])
    async def delete_all_sensor_data():
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM sensor_data")
            rows_deleted = cursor.rowcount
            conn.commit()
            return {"message": f"✅ Đã xóa tất cả dữ liệu ({rows_deleted} rows)"}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ---------------- RESET AUTO_INCREMENT ID ----------------
    @app.post("/updateID/", tags=["Sensor Data"])
    async def reset_sensor_id():
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE sensor_data AUTO_INCREMENT = 1")
            conn.commit()
            return {"message": "✅ Đã reset lại ID sensor_data về 1"}
        except Exception as e:
            conn.rollback()
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    # ESP32 POST DATA BACKEND
    @app.post("/upload_sensor", tags=["Sensor Data"])
    async def upload_sensor(data: Union[SensorData, List[SensorData]]):
        """
        Nhận dữ liệu cảm biến:
        - Có thể là 1 object duy nhất (khi gửi từng cái một)
        - Hoặc là danh sách nhiều object (khi gửi batch)
        """
        # Chuẩn hóa về dạng list để xử lý thống nhất
        if isinstance(data, SensorData):
            data_list = [data]
        else:
            data_list = data

        if not data_list:
            return {"status": "no_data", "rows": 0}

        try:
            conn = connected_DB()
            cursor = conn.cursor()
            sql = """
                INSERT INTO sensor_data (ax, ay, az, current, voltage, temp)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            # Chuyển đổi thành tuple, temp có thể là None
            values = [
                (d.ax, d.ay, d.az, d.current, d.voltage, d.temp)
                for d in data_list
            ]
            cursor.executemany(sql, values)
            conn.commit()
            cursor.close()
            conn.close()

            print(f"Đã lưu thành công {len(data_list)} bản ghi từ ESP32")
            return {"status": "saved", "rows": len(data_list)}

        except Exception as e:
            print("LỖI LƯU DỮ LIỆU:", e)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Lỗi lưu dữ liệu vào database")
