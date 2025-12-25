from fastapi import FastAPI, HTTPException, Query, status
from model.LoginRequest import LoginRequest
from model.Connected_DB import connected_DB
from passlib.context import CryptContext

app = None
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# ------------------- HÀM TIỆN ÍCH -------------------
def hash_password(password: str) -> str:
    return pwd_context.hash(password[:72])

def verify_password(plain_pass: str, hashed_pass: str) -> bool:
    return pwd_context.verify(plain_pass, hashed_pass)

# ------------------- REGISTER USER API -------------------
def user_api(fastapi_app: FastAPI):
    global app
    app = fastapi_app

    # ------------------- REGISTER USER -------------------
    @app.post("/register/", tags=["Login & User Management"], status_code=status.HTTP_201_CREATED)
    async def register_user(request: LoginRequest):
        conn = connected_DB()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT id FROM users WHERE username=%s", (request.username,))
            if cursor.fetchone():
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username đã tồn tại")

            hashed_pw = hash_password(request.password)
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (request.username, hashed_pw))
            conn.commit()
            return {"message": f"✅ Tạo user {request.username} thành công"}
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ------------------- LOGIN USER -------------------
    @app.post("/login/", tags=["Login & User Management"])
    async def login_user(request: LoginRequest):
        conn = connected_DB()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM users WHERE username=%s", (request.username,))
            user = cursor.fetchone()
            if not user or not verify_password(request.password, user["password"]):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sai username hoặc password")
            return {"message": "✅ Login thành công", "user_id": user["id"], "username": user["username"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ------------------- DELETE ALL ACCOUNTS -------------------
    @app.delete("/delete_all_accounts/", tags=["Login & User Management"])
    async def delete_all_accounts():
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM users")
            rows_deleted = cursor.rowcount
            conn.commit()
            return {"message": f"✅ Đã xóa thành công {rows_deleted} tài khoản"}
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()

    # ------------------- DELETE SPECIFIC USER -------------------
    @app.delete("/delete_user/", tags=["Login & User Management"])
    async def delete_user(username: str = Query(..., description="Username của user cần xóa")):
        conn = connected_DB()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM users WHERE username=%s", (username,))
            rows_deleted = cursor.rowcount
            conn.commit()
            if rows_deleted == 0:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Không tìm thấy user có username = {username}")
            return {"message": f"✅ Đã xóa thành công user có username = {username}"}
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            cursor.close()
            conn.close()
