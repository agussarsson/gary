from pydantic import BaseModel
from enum import Enum
from uuid import UUID
from typing import Optional

class EventType(str, Enum):
    pain = "pain"
    too_heavy = "too_heavy"
    too_light = "too_light"
    time = "time"
    equipment = "equipment"
    other = "other"

class Severity(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"

class BodyArea(str, Enum):
    none = "none"
    shoulder = "shoulder"
    knee = "knee"
    chest = "chest"
    bicep = "bicep"
    quad = "quad"
    hip = "hip"
    lower_back = "lower_back"
    other = "other"

class ExceptionEvent(BaseModel):
    session_id: UUID
    exercise_name: str
    event_type: EventType
    severity: Severity
    body_area: BodyArea = BodyArea.none
    note: Optional[str] = None
    confidence: float
