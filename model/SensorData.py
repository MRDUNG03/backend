from pydantic import BaseModel

class SensorData(BaseModel):
    ax: float
    ay: float
    az: float
    current: float
    voltage: float
    temp: float
