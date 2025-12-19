from pydantic import BaseModel
from typing import List, Optional
# ---------------- MODEL ----------------
class Alert(BaseModel):
    device_id: int
    alert: str
    message: Optional[str] = None
    status: Optional[str] = "new"

class AlertResponse(Alert):
    id: int
    timestamp: str