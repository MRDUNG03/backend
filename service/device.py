from fastapi import HTTPException, Path, Query
from typing import List
from model.Connected_DB import connected_DB
from model.Device import Device
import traceback

def device_api(app):

    # ---------------- ADD DEVICE ----------------
    @app.post("/insertDevice/", tags=["Device Control"])
    async def add_device(device: Device):
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO devices (device_name, status)
                VALUES (%s, %s)
            """, (device.device_name, device.status))
            conn.commit()
            return {"message": "✅ Thiết bị đã được thêm thành công"}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ---------------- GET DEVICE ----------------
    @app.get("/getDevice/", tags=["Device Control"])
    async def get_devices():
        conn = connected_DB()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM devices ORDER BY id ASC")
            results = cursor.fetchall()
            return results
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ---------------- DELETE DEVICE BY ID ----------------
    @app.delete("/deleteDevice/{device_id}", tags=["Device Control"])
    async def delete_device_by_id(device_id: int = Path(..., description="ID của thiết bị cần xóa")):
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM devices WHERE id = %s", (device_id,))
            conn.commit()
            if cursor.rowcount == 0:
                return {"message": f"Không tìm thấy thiết bị có id = {device_id}"}
            return {"message": f"✅ Đã xóa thành công thiết bị id = {device_id}"}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ---------------- DELETE ALL DEVICE ----------------
    @app.delete("/deleteAllDevice/", tags=["Device Control"])
    async def delete_all_device():
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM devices")
            rows_deleted = cursor.rowcount
            conn.commit()
            return {"message": f"✅ Đã xóa tất cả thiết bị ({rows_deleted} rows)"}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ---------------- UPDATE DEVICE STATUS ----------------
    @app.put("/updateStatusDevice/{device_id}", tags=["Device Control"])
    async def update_device_status(device_id: int, status: str = Query(..., description="online/offline")):
        if status not in ["online", "offline"]:
            raise HTTPException(status_code=400, detail="❌ Trạng thái không hợp lệ")
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("UPDATE devices SET status = %s WHERE id = %s", (status, device_id))
            conn.commit()
            return {"message": f"Trạng thái thiết bị {device_id} đã được cập nhật thành {status}"}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ---------------- RESET AUTO_INCREMENT ID ----------------
    @app.post("/resetDeviceID/", tags=["Device Control"])
    async def reset_device_id():
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE devices AUTO_INCREMENT = 1")
            conn.commit()
            return {"message": "✅ Đã reset lại ID bảng devices về 1"}
        except Exception as e:
            conn.rollback()
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
