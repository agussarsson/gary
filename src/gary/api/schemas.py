from datetime import date
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventType(str, Enum):
    pain = "pain"
    too_heavy = "too_heavy"
    too_light = "too_light"
    time = "time"
    equipment = "equipment"
    form = "form"
    other = "other"


class Severity(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class BodyArea(str, Enum):
    none = "none"
    lower_back = "lower_back"
    shoulder = "shoulder"
    knee = "knee"
    elbow = "elbow"
    wrist = "wrist"
    hip = "hip"
    neck = "neck"
    ankle = "ankle"
    other = "other"


class Source(str, Enum):
    manual = "manual"
    rules = "rules"
    model = "model"
    llm = "llm"


class WorkoutStyle(str, Enum):
    gym_equipment = "gym_equipment"
    body_weight = "body_weight"


class SuggestionStatus(str, Enum):
    pending = "pending"
    accepted = "accepted"
    declined = "declined"
    modified = "modified"


class SuggestionType(str, Enum):
    note_feedback = "note_feedback"
    no_note_progression = "no_note_progression"
    rule_based = "rule_based"


class StrictSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class Exercise(StrictSchema):
    name: str = Field(min_length=2, max_length=80)
    sets: int = Field(ge=1, le=10)
    reps: int = Field(ge=1, le=30)
    load: Optional[float] = Field(default=None, ge=0)
    progression_rule: str = Field(default="increase_load_if_completed", min_length=1, max_length=64)


class ProgramDay(StrictSchema):
    name: str = Field(min_length=2, max_length=50)
    focus: str = Field(min_length=2, max_length=100)
    exercises: list[Exercise] = Field(min_length=1, max_length=20)


class ProgramJSON(StrictSchema):
    split: str = Field(min_length=2, max_length=30)
    days_per_week: int = Field(ge=1, le=7)
    goal: str = Field(min_length=2, max_length=100)
    experience_level: str = Field(min_length=2, max_length=30)
    notes: Optional[str] = Field(default=None, max_length=1000)
    days: list[ProgramDay] = Field(min_length=1, max_length=7)

    @model_validator(mode="after")
    def validate_days(self):
        if len(self.days) != self.days_per_week:
            raise ValueError("days_per_week must match length of days")
        return self


class CreateProgram(StrictSchema):
    name: str = Field(min_length=1, max_length=120)
    program_json: ProgramJSON


class DeliverProgram(StrictSchema):
    id: UUID
    name: str
    program_json: dict[str, Any]


class CreateSession(StrictSchema):
    program_id: UUID
    session_date: Optional[date] = None
    day_name: Optional[str] = Field(default=None, max_length=50)


class DeliverSession(StrictSchema):
    id: UUID
    program_id: UUID
    session_date: date
    completed: bool
    day_name: Optional[str] = None


class CreateExceptionEvent(StrictSchema):
    session_id: UUID
    exercise_name: str = Field(min_length=2, max_length=100)
    event_type: EventType
    severity: Severity
    body_area: BodyArea = BodyArea.none
    note: Optional[str] = Field(default=None, max_length=500)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    source: Source = Source.manual

    @model_validator(mode="after")
    def validate_domain_rules(self):
        if self.event_type == EventType.pain and self.body_area == BodyArea.none:
            raise ValueError("body_area must be set when event_type is 'pain'")
        if self.source in (Source.model, Source.llm) and self.confidence is None:
            raise ValueError("confidence is required when source is model or llm")
        return self


class DeliverExceptionEvent(StrictSchema):
    id: UUID
    session_id: UUID
    exercise_name: str
    event_type: EventType
    severity: Severity
    body_area: BodyArea = BodyArea.none
    note: Optional[str]
    confidence: Optional[float]
    source: Source


class ProgramGenerateRequest(StrictSchema):
    goal: str = Field(min_length=2, max_length=100)
    days_per_week: int = Field(ge=1, le=7)
    experience_level: str = Field(min_length=2, max_length=30)
    workout_style: WorkoutStyle = WorkoutStyle.gym_equipment
    preferences: list[str] = Field(default_factory=list, max_length=20)
    program_name: str = Field(default="My Program", min_length=1, max_length=120)


class ProgramRefineRequest(StrictSchema):
    program_json: ProgramJSON
    feedback: str = Field(min_length=2, max_length=1000)


class WorkoutFeedbackRequest(StrictSchema):
    text: str = Field(min_length=2, max_length=2000)


class DetectedEvent(StrictSchema):
    exercise: str
    event: str
    severity: str


class AdjustmentResult(StrictSchema):
    exercise_name: str
    field: str
    multiplier: Optional[float] = None
    reason: Optional[str] = None
    body_part: Optional[str] = None


class WorkoutFeedbackResponse(StrictSchema):
    events_detected: list[DetectedEvent]
    adjustments: list[AdjustmentResult]
    updated_program: dict[str, Any]


class ApplyAdjustmentResponse(StrictSchema):
    message: Optional[str] = None
    program_id: Optional[UUID] = None
    program_json: dict[str, Any]
    adjustments: list[AdjustmentResult]


class CompleteDayRequest(StrictSchema):
    day_name: str = Field(min_length=1, max_length=50)
    note: Optional[str] = Field(default=None, max_length=2000)


class SuggestionPayload(StrictSchema):
    """Proposed program changes as a list of exercise-level adjustments."""
    adjustments: list[AdjustmentResult] = Field(default_factory=list)
    program_json_patch: Optional[dict[str, Any]] = None


class DeliverSuggestion(StrictSchema):
    id: UUID
    program_id: UUID
    session_id: Optional[UUID] = None
    suggestion_type: SuggestionType
    status: SuggestionStatus
    payload: dict[str, Any]
    rationale: Optional[str] = None
    day_name: Optional[str] = None
    created_at: str


class SuggestionDecisionRequest(StrictSchema):
    decision: Literal["accept", "decline", "modify"]
    modified_program_json: Optional[ProgramJSON] = None


class GenerateProgramResponse(StrictSchema):
    program: DeliverProgram
    source: str
    message: Optional[str] = None


class AuthSignupRequest(StrictSchema):
    email: str = Field(min_length=5, max_length=200)
    password: str = Field(min_length=8, max_length=200)


class AuthLoginRequest(StrictSchema):
    email: str = Field(min_length=5, max_length=200)
    password: str = Field(min_length=8, max_length=200)


class AuthUser(StrictSchema):
    id: UUID
    email: str


class AuthMeResponse(StrictSchema):
    authenticated: bool
    user: Optional[AuthUser] = None
    llm_used_calls: int = 0
    llm_max_calls: int = 10
