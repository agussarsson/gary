import copy
import hashlib
import json
import os
from typing import Optional
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from gary.ml.inference import ExceptionModel

from .database import SessionLocal
from .auth import hash_password, new_session_token, session_expiry, utc_now, verify_password
from .engine import adjust_workout, apply_adjustments as engine_apply_adjustments
from .generation import generate_program, refine_program_json
from .schemas import (
    AuthLoginRequest,
    AuthMeResponse,
    AuthSignupRequest,
    AuthUser,
    ApplyAdjustmentResponse,
    CompleteDayRequest,
    CreateExceptionEvent,
    CreateProgram,
    CreateSession,
    DeliverExceptionEvent,
    DeliverProgram,
    DeliverSession,
    DeliverSuggestion,
    GenerateProgramResponse,
    ProgramGenerateRequest,
    ProgramRefineRequest,
    SuggestionDecisionRequest,
    WorkoutFeedbackRequest,
    WorkoutFeedbackResponse,
)
from .suggestions_service import (
    adjustments_to_results,
    build_progression_payload,
    events_to_adjustments,
    heuristic_note_to_events,
    last_three_no_note_completions,
)

app = FastAPI()
SESSION_COOKIE_NAME = "gary_session"
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))
AUTH_MAX_USERS = int(os.getenv("AUTH_MAX_USERS", "5"))
LLM_MAX_CALLS = int(os.getenv("LLM_MAX_CALLS", "10"))


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _resolve_user_from_cookie(db: Session, request: Request) -> Optional[dict]:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    now = utc_now()
    row = db.execute(
        text(
            """
            SELECT u.id, u.email, s.expires_at, s.revoked_at
            FROM auth_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.session_token_hash = :token_hash
            """
        ),
        {"token_hash": _token_hash(token)},
    ).mappings().first()
    if not row:
        return None
    if row["revoked_at"] is not None:
        return None
    if row["expires_at"] <= now:
        return None
    return dict(row)


def _require_auth_user(db: Session, request: Request) -> dict:
    user = _resolve_user_from_cookie(db, request)
    if not user:
        raise api_error(401, "Authentication required")
    return user


def _consume_llm_quota(db: Session) -> tuple[int, int]:
    db.execute(
        text(
            """
            INSERT INTO llm_quota(id, used_calls, max_calls)
            VALUES (1, 0, :max_calls)
            ON CONFLICT (id) DO NOTHING
            """
        ),
        {"max_calls": LLM_MAX_CALLS},
    )
    updated = db.execute(
        text(
            """
            UPDATE llm_quota
            SET used_calls = used_calls + 1, updated_at = now()
            WHERE id = 1 AND used_calls < max_calls
            RETURNING used_calls, max_calls
            """
        )
    ).mappings().first()
    if not updated:
        snapshot = db.execute(
            text("SELECT used_calls, max_calls FROM llm_quota WHERE id = 1")
        ).mappings().first()
        used_calls = int(snapshot["used_calls"]) if snapshot else LLM_MAX_CALLS
        max_calls = int(snapshot["max_calls"]) if snapshot else LLM_MAX_CALLS
        raise api_error(429, f"LLM call budget exhausted ({used_calls}/{max_calls})")
    return int(updated["used_calls"]), int(updated["max_calls"])


allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: ExceptionModel | None = None
model_load_error: str | None = None

try:
    model = ExceptionModel("artifacts/production/exception_model")
except Exception as exc:  # pragma: no cover - defensive startup guard
    model_load_error = str(exc)


def api_error(status_code: int, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"message": message})


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "message" in detail:
        payload = {"error": {"message": detail["message"], "status_code": exc.status_code}}
    else:
        payload = {"error": {"message": str(detail), "status_code": exc.status_code}}
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Request validation failed",
                "status_code": 422,
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, __):
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "status_code": 500}},
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.middleware("http")
async def auth_gate_middleware(request: Request, call_next):
    path = request.url.path
    open_prefixes = (
        "/health",
        "/ready",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/auth/",
    )
    if path.startswith(open_prefixes):
        return await call_next(request)
    db = SessionLocal()
    try:
        user = _resolve_user_from_cookie(db, request)
    finally:
        db.close()
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Authentication required", "status_code": 401}},
        )
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(db: Session = Depends(get_db)):
    db_ready = False
    try:
        db.execute(text("SELECT 1"))
        db_ready = True
    except Exception:
        db_ready = False

    model_ready = model is not None
    overall_status = "ready" if db_ready else "not_ready"

    return {
        "status": overall_status,
        "database": "ready" if db_ready else "not_ready",
        "model": "ready" if model_ready else "not_ready",
        "model_error": model_load_error,
    }


@app.post("/auth/signup")
def auth_signup(payload: AuthSignupRequest, response: Response, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    db.execute(text("SELECT pg_advisory_xact_lock(1001)"))
    user_count = db.execute(text("SELECT COUNT(*) AS c FROM users")).mappings().one()["c"]
    if int(user_count) >= AUTH_MAX_USERS:
        raise api_error(403, f"Signup limit reached ({AUTH_MAX_USERS} users)")
    existing = db.execute(text("SELECT id FROM users WHERE email = :email"), {"email": email}).mappings().first()
    if existing:
        raise api_error(409, "Email already registered")
    user = db.execute(
        text(
            """
            INSERT INTO users(email, password_hash)
            VALUES (:email, :password_hash)
            RETURNING id, email
            """
        ),
        {"email": email, "password_hash": hash_password(payload.password)},
    ).mappings().one()
    token = new_session_token()
    expiry = session_expiry(SESSION_TTL_HOURS)
    db.execute(
        text(
            """
            INSERT INTO auth_sessions(user_id, session_token_hash, expires_at)
            VALUES (:user_id, :token_hash, :expires_at)
            """
        ),
        {"user_id": str(user["id"]), "token_hash": _token_hash(token), "expires_at": expiry},
    )
    db.commit()
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        secure=os.getenv("COOKIE_SECURE", "false").lower() == "true",
        samesite="lax",
        max_age=SESSION_TTL_HOURS * 3600,
        path="/",
    )
    return {"authenticated": True, "user": dict(user), "llm_used_calls": 0, "llm_max_calls": LLM_MAX_CALLS}


@app.post("/auth/login")
def auth_login(payload: AuthLoginRequest, response: Response, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    row = db.execute(
        text("SELECT id, email, password_hash FROM users WHERE email = :email"),
        {"email": email},
    ).mappings().first()
    if not row or not verify_password(payload.password, row["password_hash"]):
        raise api_error(401, "Invalid credentials")
    token = new_session_token()
    expiry = session_expiry(SESSION_TTL_HOURS)
    db.execute(
        text(
            """
            INSERT INTO auth_sessions(user_id, session_token_hash, expires_at)
            VALUES (:user_id, :token_hash, :expires_at)
            """
        ),
        {"user_id": str(row["id"]), "token_hash": _token_hash(token), "expires_at": expiry},
    )
    db.commit()
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        secure=os.getenv("COOKIE_SECURE", "false").lower() == "true",
        samesite="lax",
        max_age=SESSION_TTL_HOURS * 3600,
        path="/",
    )
    quota = db.execute(text("SELECT used_calls, max_calls FROM llm_quota WHERE id = 1")).mappings().first()
    used_calls = int(quota["used_calls"]) if quota else 0
    max_calls = int(quota["max_calls"]) if quota else LLM_MAX_CALLS
    return {
        "authenticated": True,
        "user": {"id": row["id"], "email": row["email"]},
        "llm_used_calls": used_calls,
        "llm_max_calls": max_calls,
    }


@app.post("/auth/logout")
def auth_logout(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        db.execute(
            text(
                """
                UPDATE auth_sessions
                SET revoked_at = now()
                WHERE session_token_hash = :token_hash AND revoked_at IS NULL
                """
            ),
            {"token_hash": _token_hash(token)},
        )
        db.commit()
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/auth/me", response_model=AuthMeResponse)
def auth_me(request: Request, db: Session = Depends(get_db)):
    user = _resolve_user_from_cookie(db, request)
    quota = db.execute(text("SELECT used_calls, max_calls FROM llm_quota WHERE id = 1")).mappings().first()
    used_calls = int(quota["used_calls"]) if quota else 0
    max_calls = int(quota["max_calls"]) if quota else LLM_MAX_CALLS
    if not user:
        return {"authenticated": False, "user": None, "llm_used_calls": used_calls, "llm_max_calls": max_calls}
    return {
        "authenticated": True,
        "user": {"id": user["id"], "email": user["email"]},
        "llm_used_calls": used_calls,
        "llm_max_calls": max_calls,
    }


# --- Programs: generate / CRUD ---


@app.post("/programs/generate", response_model=GenerateProgramResponse)
def programs_generate(payload: ProgramGenerateRequest, db: Session = Depends(get_db)):
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        _consume_llm_quota(db)
    pj, source, msg = generate_program(payload)
    q = text(
        """
        INSERT INTO programs (name, program_json)
        VALUES (:name, CAST(:program_json AS json))
        RETURNING id, name, program_json;
        """
    )
    row = db.execute(
        q,
        {
            "name": payload.program_name,
            "program_json": json.dumps(pj.model_dump()),
        },
    ).mappings().one()
    db.commit()
    return {"program": dict(row), "source": source, "message": msg}


@app.post("/programs", response_model=DeliverProgram)
def create_program(payload: CreateProgram, db: Session = Depends(get_db)):
    q = text(
        """
        INSERT INTO programs (name, program_json)
        VALUES (:name, CAST(:program_json as json))
        RETURNING id, name, program_json;
        """
    )
    payload_dict = payload.model_dump()
    row = db.execute(
        q,
        {"name": payload_dict["name"], "program_json": json.dumps(payload_dict["program_json"])},
    ).mappings().one()
    db.commit()
    return dict(row)


@app.get("/programs/{program_id}", response_model=DeliverProgram)
def get_program(program_id: UUID, db: Session = Depends(get_db)):
    row = db.execute(
        text("SELECT id, name, program_json FROM programs WHERE id = :id"),
        {"id": str(program_id)},
    ).mappings().first()
    if not row:
        raise api_error(404, "Program not found")
    return dict(row)


@app.put("/programs/{program_id}", response_model=DeliverProgram)
def update_program(program_id: UUID, payload: CreateProgram, db: Session = Depends(get_db)):
    exists = db.execute(
        text("SELECT id FROM programs WHERE id = :id"),
        {"id": str(program_id)},
    ).mappings().first()
    if not exists:
        raise api_error(404, "Program not found")
    dumped = payload.model_dump()
    db.execute(
        text(
            """
            UPDATE programs
            SET name = :name, program_json = CAST(:pj AS jsonb)
            WHERE id = :id
            """
        ),
        {
            "name": dumped["name"],
            "pj": json.dumps(dumped["program_json"]),
            "id": str(program_id),
        },
    )
    db.commit()
    row = db.execute(
        text("SELECT id, name, program_json FROM programs WHERE id = :id"),
        {"id": str(program_id)},
    ).mappings().one()
    return dict(row)


@app.post("/programs/{program_id}/refine", response_model=DeliverProgram)
def refine_program(program_id: UUID, payload: ProgramRefineRequest, db: Session = Depends(get_db)):
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        _consume_llm_quota(db)
    row = db.execute(
        text("SELECT id, program_json FROM programs WHERE id = :id"),
        {"id": str(program_id)},
    ).mappings().first()
    if not row:
        raise api_error(404, "Program not found")

    pj, source, err = refine_program_json(payload.program_json, payload.feedback)
    if pj is None:
        raise api_error(400, err or "Refinement failed")

    db.execute(
        text("UPDATE programs SET program_json = CAST(:pj AS jsonb) WHERE id = :id"),
        {"pj": json.dumps(pj.model_dump()), "id": str(program_id)},
    )
    db.commit()
    out = db.execute(
        text("SELECT id, name, program_json FROM programs WHERE id = :id"),
        {"id": str(program_id)},
    ).mappings().one()
    return dict(out)


# --- Sessions ---


@app.post("/sessions", response_model=DeliverSession)
def create_session(payload: CreateSession, db: Session = Depends(get_db)):
    if payload.session_date:
        if payload.day_name:
            q = text(
                """
                INSERT INTO sessions (program_id, session_date, day_name)
                VALUES (:program_id, :session_date, :day_name)
                RETURNING id, program_id, session_date, completed, day_name;
                """
            )
            params = {
                "program_id": str(payload.program_id),
                "session_date": payload.session_date,
                "day_name": payload.day_name,
            }
        else:
            q = text(
                """
                INSERT INTO sessions (program_id, session_date)
                VALUES (:program_id, :session_date)
                RETURNING id, program_id, session_date, completed, day_name;
                """
            )
            params = {"program_id": str(payload.program_id), "session_date": payload.session_date}
    else:
        if payload.day_name:
            q = text(
                """
                INSERT INTO sessions (program_id, day_name)
                VALUES (:program_id, :day_name)
                RETURNING id, program_id, session_date, completed, day_name;
                """
            )
            params = {"program_id": str(payload.program_id), "day_name": payload.day_name}
        else:
            q = text(
                """
                INSERT INTO sessions (program_id)
                VALUES (:program_id)
                RETURNING id, program_id, session_date, completed, day_name;
                """
            )
            params = {"program_id": str(payload.program_id)}

    row = db.execute(q, params).mappings().first()
    if not row:
        raise api_error(400, "Failed to create session (check program_id)")
    db.commit()
    return dict(row)


@app.get("/sessions/{session_id}", response_model=DeliverSession)
def get_session(session_id: UUID, db: Session = Depends(get_db)):
    row = db.execute(
        text(
            """
            SELECT id, program_id, session_date, completed, day_name
            FROM sessions WHERE id = :id
            """
        ),
        {"id": str(session_id)},
    ).mappings().first()
    if not row:
        raise api_error(404, "Session not found")
    return dict(row)


@app.post("/sessions/{session_id}/complete-day")
def complete_day(session_id: UUID, payload: CompleteDayRequest, db: Session = Depends(get_db)):
    session_row = db.execute(
        text("SELECT id, program_id FROM sessions WHERE id = :id"),
        {"id": str(session_id)},
    ).mappings().first()
    if not session_row:
        raise api_error(404, "Session not found")

    program_id = session_row["program_id"]
    program_row = db.execute(
        text("SELECT program_json FROM programs WHERE id = :id"),
        {"id": str(program_id)},
    ).mappings().first()
    if not program_row:
        raise api_error(404, "Program not found")

    pj = program_row["program_json"]
    day_index = 0
    for i, d in enumerate(pj.get("days", [])):
        if d.get("name") == payload.day_name:
            day_index = i
            break

    db.execute(
        text(
            """
            INSERT INTO day_completions (program_id, session_id, day_name, day_index, note)
            VALUES (:program_id, :session_id, :day_name, :day_index, :note)
            """
        ),
        {
            "program_id": str(program_id),
            "session_id": str(session_id),
            "day_name": payload.day_name,
            "day_index": day_index,
            "note": payload.note,
        },
    )
    db.execute(
        text("UPDATE sessions SET completed = TRUE WHERE id = :id"),
        {"id": str(session_id)},
    )
    db.commit()

    trigger, _rows = last_three_no_note_completions(db, program_id, payload.day_name)
    created_suggestion = None
    if trigger:
        existing = db.execute(
            text(
                """
                SELECT id FROM progression_suggestions
                WHERE program_id = :pid AND day_name = :dn
                  AND suggestion_type = 'no_note_progression' AND status = 'pending'
                LIMIT 1
                """
            ),
            {"pid": str(program_id), "dn": payload.day_name},
        ).mappings().first()
        if not existing:
            adjs = build_progression_payload(pj, payload.day_name)
            ins = db.execute(
                text(
                    """
                    INSERT INTO progression_suggestions
                    (program_id, session_id, suggestion_type, status, payload, rationale, day_name)
                    VALUES
                    (:program_id, :session_id, 'no_note_progression', 'pending', CAST(:payload AS jsonb), :rationale, :day_name)
                    RETURNING id
                    """
                ),
                {
                    "program_id": str(program_id),
                    "session_id": str(session_id),
                    "payload": json.dumps({"adjustments": adjustments_to_results(adjs)}),
                    "rationale": "Three recent completions for this day had no notes. Suggest a small load increase.",
                    "day_name": payload.day_name,
                },
            ).mappings().one()
            db.commit()
            created_suggestion = str(ins["id"])

    return {
        "ok": True,
        "progression_suggestion_id": created_suggestion,
    }


@app.post("/sessions/{session_id}/suggestions/from-note", response_model=DeliverSuggestion)
def suggestions_from_note(session_id: UUID, payload: WorkoutFeedbackRequest, db: Session = Depends(get_db)):
    session_row = db.execute(
        text("SELECT program_id FROM sessions WHERE id = :id"),
        {"id": str(session_id)},
    ).mappings().first()
    if not session_row:
        raise api_error(404, "Session not found")

    program_id = session_row["program_id"]
    program_row = db.execute(
        text("SELECT program_json FROM programs WHERE id = :id"),
        {"id": str(program_id)},
    ).mappings().first()
    if not program_row:
        raise api_error(404, "Program not found")

    pj = program_row["program_json"]
    exercise_names: list[str] = []
    for day in pj.get("days", []):
        for ex in day.get("exercises", []):
            exercise_names.append(ex["name"])

    events: list[dict] = []
    if model is not None:
        session_exercises = "; ".join(exercise_names)
        for exercise in exercise_names:
            pred = model.predict(exercise, session_exercises, payload.text)
            if pred["event_type"] == "none":
                continue
            events.append(
                {
                    "exercise_name": exercise,
                    "event_type": pred["event_type"],
                    "severity": pred["severity"],
                    "body_area": "none",
                }
            )
    if not events:
        events = heuristic_note_to_events(payload.text, exercise_names)

    adjs = events_to_adjustments(events)
    rationale = "Model/heuristic inferred adjustments from your note. Review before accepting."

    row = db.execute(
        text(
            """
            INSERT INTO progression_suggestions
            (program_id, session_id, suggestion_type, status, payload, rationale, day_name)
            VALUES
            (:program_id, :session_id, 'note_feedback', 'pending', CAST(:payload AS jsonb), :rationale, NULL)
            RETURNING id, program_id, session_id, suggestion_type, status, payload, rationale, day_name, created_at
            """
        ),
        {
            "program_id": str(program_id),
            "session_id": str(session_id),
            "payload": json.dumps({"adjustments": adjustments_to_results(adjs)}),
            "rationale": rationale,
        },
    ).mappings().one()
    db.commit()
    r = dict(row)
    r["suggestion_type"] = r["suggestion_type"]
    r["created_at"] = r["created_at"].isoformat() if r.get("created_at") else ""
    return r


@app.get("/programs/{program_id}/suggestions", response_model=list[DeliverSuggestion])
def list_program_suggestions(program_id: UUID, db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            """
            SELECT id, program_id, session_id, suggestion_type, status, payload, rationale, day_name, created_at
            FROM progression_suggestions
            WHERE program_id = :pid AND status = 'pending'
            ORDER BY created_at DESC
            """
        ),
        {"pid": str(program_id)},
    ).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        d["created_at"] = d["created_at"].isoformat() if d.get("created_at") else ""
        out.append(d)
    return out


@app.post("/suggestions/{suggestion_id}/decision")
def suggestion_decision(suggestion_id: UUID, payload: SuggestionDecisionRequest, db: Session = Depends(get_db)):
    sug = db.execute(
        text(
            """
            SELECT id, program_id, session_id, status, payload, suggestion_type
            FROM progression_suggestions WHERE id = :id
            """
        ),
        {"id": str(suggestion_id)},
    ).mappings().first()
    if not sug:
        raise api_error(404, "Suggestion not found")
    if sug["status"] != "pending":
        raise api_error(400, "Suggestion already resolved")

    program_id = sug["program_id"]
    prog_row = db.execute(
        text("SELECT program_json FROM programs WHERE id = :id"),
        {"id": str(program_id)},
    ).mappings().first()
    if not prog_row:
        raise api_error(404, "Program not found")

    program_json = copy.deepcopy(prog_row["program_json"])
    new_status = payload.decision
    modified: Optional[dict] = None

    if payload.decision == "decline":
        db.execute(
            text("UPDATE progression_suggestions SET status = 'declined' WHERE id = :id"),
            {"id": str(suggestion_id)},
        )
    elif payload.decision == "modify" and payload.modified_program_json is not None:
        modified = payload.modified_program_json.model_dump()
        db.execute(
            text("UPDATE programs SET program_json = CAST(:pj AS jsonb) WHERE id = :id"),
            {"pj": json.dumps(modified), "id": str(program_id)},
        )
        db.execute(
            text("UPDATE progression_suggestions SET status = 'modified' WHERE id = :id"),
            {"id": str(suggestion_id)},
        )
    elif payload.decision == "accept":
        payload_data = sug["payload"]
        if isinstance(payload_data, str):
            payload_data = json.loads(payload_data)
        adjs = payload_data.get("adjustments", [])
        internal = [
            {
                "exercise_name": a["exercise_name"],
                "field": a["field"],
                "multiplier": a.get("multiplier"),
                "reason": a.get("reason", ""),
                "body_part": a.get("body_part"),
            }
            for a in adjs
        ]
        updated = engine_apply_adjustments(copy.deepcopy(program_json), internal)
        db.execute(
            text("UPDATE programs SET program_json = CAST(:pj AS jsonb) WHERE id = :id"),
            {"pj": json.dumps(updated), "id": str(program_id)},
        )
        for a in internal:
            db.execute(
                text(
                    """
                    INSERT INTO adjustments (program_id, exercise_name, field, old_value, new_value, reason)
                    VALUES (:program_id, :exercise_name, :field, :old_value, :new_value, :reason)
                    """
                ),
                {
                    "program_id": str(program_id),
                    "exercise_name": a["exercise_name"],
                    "field": a["field"],
                    "old_value": None,
                    "new_value": str(a.get("multiplier")),
                    "reason": a.get("reason") or "Suggestion accepted",
                },
            )
        db.execute(
            text("UPDATE progression_suggestions SET status = 'accepted' WHERE id = :id"),
            {"id": str(suggestion_id)},
        )
        modified = updated

    if modified is not None:
        db.execute(
            text(
                """
                INSERT INTO suggestion_decisions (suggestion_id, decision, modified_program_json)
                VALUES (:sid, :decision, CAST(:mj AS jsonb))
                """
            ),
            {"sid": str(suggestion_id), "decision": payload.decision, "mj": json.dumps(modified)},
        )
    else:
        db.execute(
            text(
                """
                INSERT INTO suggestion_decisions (suggestion_id, decision, modified_program_json)
                VALUES (:sid, :decision, NULL)
                """
            ),
            {"sid": str(suggestion_id), "decision": payload.decision},
        )
    db.commit()
    return {"ok": True, "program_json": modified}


# --- Exceptions ---


@app.post("/exceptions", response_model=DeliverExceptionEvent)
def create_exception(payload: CreateExceptionEvent, db: Session = Depends(get_db)):
    q = text(
        """
        INSERT INTO exception_events (
            session_id, exercise_name, event_type, severity, body_area, note, confidence, source
        )
        VALUES (
            :session_id, :exercise_name, :event_type, :severity, :body_area, :note, :confidence, :source
        )
        RETURNING id, session_id, exercise_name, event_type, severity, body_area, note, confidence, source;
        """
    )
    params = {
        "session_id": str(payload.session_id),
        "exercise_name": payload.exercise_name,
        "event_type": payload.event_type.value,
        "severity": payload.severity.value,
        "body_area": payload.body_area.value,
        "note": payload.note,
        "confidence": payload.confidence,
        "source": payload.source.value,
    }
    row = db.execute(q, params).mappings().first()
    if not row:
        raise api_error(400, "Failed to create exception (check session_id)")
    db.commit()
    return dict(row)


@app.get("/sessions/{session_id}/exceptions", response_model=list[DeliverExceptionEvent])
def list_exceptions(session_id: UUID, db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            """
            SELECT id, session_id, exercise_name, event_type, severity, body_area, note, confidence, source
            FROM exception_events
            WHERE session_id = :session_id
            ORDER BY created_at ASC;
            """
        ),
        {"session_id": str(session_id)},
    ).mappings().all()
    return [dict(r) for r in rows]


@app.post("/sessions/{session_id}/apply-adjustment", response_model=ApplyAdjustmentResponse)
def apply_adjustment(session_id: UUID, db: Session = Depends(get_db)):
    session_row = db.execute(
        text("SELECT id, program_id FROM sessions WHERE id = :id"),
        {"id": str(session_id)},
    ).mappings().first()

    if not session_row:
        raise api_error(404, "Session not found")

    program_id = session_row["program_id"]

    program_row = db.execute(
        text("SELECT id, program_json FROM programs WHERE id = :id"),
        {"id": str(program_id)},
    ).mappings().first()

    if not program_row:
        raise api_error(404, "Program not found")

    program_json = program_row["program_json"]

    events = db.execute(
        text(
            """
            SELECT *
            FROM exception_events
            WHERE session_id = :sid
            ORDER BY created_at ASC
            """
        ),
        {"sid": str(session_id)},
    ).mappings().all()

    if not events:
        return {"message": "No exceptions found", "program_json": program_json, "adjustments": []}

    all_adjustments = []
    for event in events:
        ev = dict(event)
        all_adjustments.extend(adjust_workout(ev))

    adjusted_program = engine_apply_adjustments(program_json, all_adjustments)

    db.execute(
        text("UPDATE programs SET program_json = CAST(:pj AS jsonb) WHERE id = :id"),
        {"pj": json.dumps(adjusted_program), "id": str(program_id)},
    )

    for adj in all_adjustments:
        db.execute(
            text(
                """
                INSERT INTO adjustments (program_id, exercise_name, field, old_value, new_value, reason)
                VALUES (:program_id, :exercise_name, :field, :old_value, :new_value, :reason)
                """
            ),
            {
                "program_id": program_id,
                "exercise_name": adj["exercise_name"],
                "field": adj["field"],
                "old_value": str(adj.get("old_value")) if adj.get("old_value") is not None else None,
                "new_value": str(adj.get("new_value")) if adj.get("new_value") is not None else None,
                "reason": adj.get("reason", "Adjustment applied"),
            },
        )

    db.commit()

    return {"program_id": program_id, "program_json": adjusted_program, "adjustments": all_adjustments}


@app.post("/sessions/{session_id}/workout-feedback", response_model=WorkoutFeedbackResponse)
def workout_feedback(session_id: UUID, payload: WorkoutFeedbackRequest, db: Session = Depends(get_db)):
    if model is None:
        raise api_error(503, "Model is not ready. Check /ready for details.")

    user_text = payload.text

    session_row = db.execute(
        text("SELECT program_id FROM sessions WHERE id = :id"),
        {"id": str(session_id)},
    ).mappings().first()

    if not session_row:
        raise api_error(404, "Session not found")

    program_id = session_row["program_id"]

    program_row = db.execute(
        text("SELECT program_json FROM programs WHERE id = :id"),
        {"id": str(program_id)},
    ).mappings().first()

    if not program_row:
        raise api_error(404, "Program not found")

    program_json = program_row["program_json"]
    exercises: list[str] = []
    for day in program_json["days"]:
        for ex in day["exercises"]:
            exercises.append(ex["name"])

    session_exercises = "; ".join(exercises)
    events_out = []
    events_for_adjust = []

    for exercise in exercises:
        pred = model.predict(exercise, session_exercises, user_text)
        if pred["event_type"] == "none":
            continue

        db.execute(
            text(
                """
                INSERT INTO exception_events (
                    session_id,
                    exercise_name,
                    event_type,
                    severity,
                    body_area,
                    confidence,
                    source,
                    note
                )
                VALUES (
                    :session_id,
                    :exercise_name,
                    :event_type,
                    :severity,
                    'none',
                    :confidence,
                    'model',
                    :note
                )
                """
            ),
            {
                "session_id": str(session_id),
                "exercise_name": exercise,
                "event_type": pred["event_type"],
                "severity": pred["severity"],
                "confidence": pred["confidence"],
                "note": user_text,
            },
        )

        events_out.append(
            {"exercise": exercise, "event": pred["event_type"], "severity": pred["severity"]}
        )
        events_for_adjust.append(
            {
                "exercise_name": exercise,
                "event_type": pred["event_type"],
                "severity": pred["severity"],
                "body_area": "none",
            }
        )

    db.commit()

    adjustments = []
    for ev in events_for_adjust:
        adjustments.extend(adjust_workout(ev))

    adjusted_program = engine_apply_adjustments(program_json, adjustments)
    db.execute(
        text("UPDATE programs SET program_json = CAST(:pj AS jsonb) WHERE id = :id"),
        {"pj": json.dumps(adjusted_program), "id": str(program_id)},
    )
    db.commit()

    return {"events_detected": events_out, "adjustments": adjustments, "updated_program": adjusted_program}
