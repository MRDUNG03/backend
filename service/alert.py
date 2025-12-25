from fastapi import HTTPException, Path, Query
from typing import List
from model.Connected_DB import connected_DB
from model.Alert import Alert
import traceback

def alert_api(app):

    # ---------------- POST ALERT ----------------
    @app.post("/postAlert/", tags=["Alerts & Notifications"])
    async def create_alert(alert: Alert):
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO alerts (device_id, alert, message, status)
                VALUES (%s, %s, %s, %s)
            """, (alert.device_id, alert.alert, alert.message, alert.status))
            conn.commit()
            return {"message": "✅ Cảnh báo đã được lưu thành công"}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ---------------- GET ALERT ----------------
    @app.get("/getAlert/", tags=["Alerts & Notifications"], response_model=List[Alert])
    async def get_alerts():
        conn = connected_DB()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
            alerts = cursor.fetchall()
            return alerts
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ---------------- DELETE ALL ALERT ----------------
    @app.delete("/deleteAllAlerts/", tags=["Alerts & Notifications"])
    async def delete_all_alerts():
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM alerts")
            rows_deleted = cursor.rowcount
            conn.commit()
            return {"message": f"✅ Đã xoá toàn bộ cảnh báo ({rows_deleted} rows)"}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ---------------- RESET ALERT ID ----------------
    @app.post("/resetAlertID/", tags=["Alerts & Notifications"])
    async def reset_alert_id():
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM alerts")
            cursor.execute("ALTER TABLE alerts AUTO_INCREMENT = 1")
            conn.commit()
            return {"message": "✅ Đã xoá toàn bộ cảnh báo và reset ID về 1"}
        except Exception as e:
            conn.rollback()
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()

    # ---------------- GET STATUS ----------------
    @app.get("/getStatus/", tags=["Alerts & Notifications"])
    async def get_status():
        conn = connected_DB()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT status, message FROM status WHERE id = 1")
            result = cursor.fetchone()
            if result is None:
                return {
                    "status": "Normal",
                    "message": "Hệ thống đang hoạt động bình thường"
                }
            return result
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Lỗi server")
        finally:
            cursor.close()
            conn.close()
