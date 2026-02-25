from pydantic import BaseModel, Field
from enum import Enum
from uuid import UUID
from typing import Optional, Any
from datetime import date

class EventType(str, Enum):
    pain = "pain"
    too_heavy = "too_heavy"
    too_light = "too_light"
    time = "time"
    # maybe add an option for when time is not a limiting factor,
    # but the user instead have more time available than what the workout required.
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

class Source(str, Enum):
    manual = "manual"
    rules = "rules"
    model = "model"
    llm = "llm"

class CreateProgram(BaseModel):
    name: str = Field(min_length=1)
    program_json: dict[str, Any]

class DeliverProgram(BaseModel):
    id: UUID
    name: str
    program_json: dict[str, Any]

class CreateSession(BaseModel):
    program_id: UUID
    session_date: Optional[date] = None

class DeliverSession(BaseModel):
    id: UUID
    program_id: UUID
    session_date: date
    completed: bool

class CreateExceptionEvent(BaseModel):
    session_id: UUID
    exercise_name: str
    event_type: EventType
    severity: Severity
    body_area: BodyArea = BodyArea.none
    note: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    source: Source = Source.manual

class DeliverExceptionEvent(BaseModel):
    id: UUID
    session_id: UUID
    exercise_name: str
    event_type: EventType
    severity: Severity
    body_area: BodyArea = BodyArea.none
    note: Optional[str]
    confidence: Optional[float]
    source: Source
