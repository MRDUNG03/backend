from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# ------------------ HÀM XÁC THỰC NGƯỜI DÙNG  ------------------
security = HTTPBasic()
# TẠO USER
username = "admin"
password = "1"
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = credentials.username == username
    correct_password = credentials.password == password
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
