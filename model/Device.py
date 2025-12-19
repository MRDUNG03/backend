from pydantic import BaseModel
class Device(BaseModel):
    device_name: str
    status: str = "offline"   # mặc định offline