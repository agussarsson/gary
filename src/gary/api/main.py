from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from uuid import UUID
import json

from .database import SessionLocal
from .schemas import (
    CreateProgram, DeliverProgram,
    CreateSession, DeliverSession,
    CreateExceptionEvent, DeliverExceptionEvent
)
from .engine import adjust_workout, apply_adjustments

app = FastAPI()

def get_db():
    """
    Establishes DB connection.
    """
    db = SessionLocal()
    try:
        yield db
    except RuntimeError:
        print("RuntimeError: Could not fetch DB.")
    finally:
        db.close()

@app.get("/health")
def health():
    return {"status": "ok"}


# --- PROGRAMS ENDPOINTS ---

@app.post("/programs", response_model=DeliverProgram)
def create_program(payload: CreateProgram, db: Session = Depends(get_db)):
    q = text("""
        INSERT INTO programs (name, program_json)
        VALUES (:name, CAST(:program_json as json))
        RETURNING id, name, program_json;
    """)

    row = db.execute(q, {"name": payload.name, "program_json": json.dumps(payload.program_json)}).mappings().one()
    db.commit()
    return dict(row)

@app.get("/programs/{program_id}", response_model=DeliverProgram)
def get_program(program_id: UUID, db: Session = Depends(get_db)):
    q = text("""
        SELECT id, name, program_json
        FROM programs
        WHERE id = :id
    """)

    row = db.execute(q, {"id": str(program_id)}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Program Not Found.")
    else:
        return dict(row)


# --- SESSIONS ENDPOINTS ---

@app.post("/sessions", response_model=DeliverSession)
def create_session(payload: CreateSession, db: Session = Depends(get_db)):
    if payload.session_date:
        q = text("""
            INSERT INTO sessions (program_id, session_date)
            VALUES (:program_id, :session_date)
            RETURNING id, program_id, session_date, completed;
        """)
        params = {"program_id": str(payload.program_id), "session_date": payload.session_date}
    else:
        q = text("""
            INSERT INTO sessions (program_id)
            VALUES (:program_id)
            RETURNING id, program_id, session_date, completed;
        """)
        params = {"program_id": str(payload.program_id)}

    row = db.execute(q, params).mappings().first()
    if not row:
        raise HTTPException(status_code=400, detail="Failed to create session (check program_id)")
    db.commit()
    return dict(row)

@app.get("/sessions/{session_id}", response_model=DeliverSession)
def get_session(session_id: UUID, db: Session = Depends(get_db)):
    q = text("""
        SELECT id, program_id, session_date, completed
        FROM sessions
        WHERE id = :id
    """)
    row = db.execute(q, {"id": str(session_id)}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return dict(row)


# --- EXCEPTIONS (regarding the workout) ENDPOINTS ---

@app.post("/exceptions", response_model=DeliverExceptionEvent)
def create_exception(payload: CreateExceptionEvent, db: Session = Depends(get_db)):
    q = text("""
        INSERT INTO exception_events (
            session_id, exercise_name, event_type, severity, body_area, note, confidence, source
        )
        VALUES (
            :session_id, :exercise_name, :event_type, :severity, :body_area, :note, :confidence, :source
        )
        RETURNING id, session_id, exercise_name, event_type, severity, body_area, note, confidence, source;
    """)
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
        raise HTTPException(status_code=400, detail="Failed to create exception (check session_id)")
    db.commit()
    return dict(row)

@app.get("/sessions/{session_id}/exceptions", response_model=list[DeliverExceptionEvent])
def list_exceptions(session_id: UUID, db: Session = Depends(get_db)):
    q = text("""
        SELECT id, session_id, exercise_name, event_type, severity, body_area, note, confidence, source
        FROM exception_events
        WHERE session_id = :session_id
        ORDER BY created_at ASC;
    """)
    rows = db.execute(q, {"session_id": str(session_id)}).mappings().all()
    return [dict(r) for r in rows]


# --- WORKOUT ADJUSTMENTS ENDPOINTS ---

@app.post("/sessions/{session_id}/apply-adjustment")
def apply_adjustments(session_id: UUID, db: Session = Depends(get_db)):
    
    session_row = db.execute(
        text("SELECT id, program_id FROM sessions WHERE id = :id"),
        {"id": str(session_id)}
    ).mappings().first()

    if not session_row:
        raise HTTPException(status_code=404, detail="Session Not Found.")
    
    program_id = session_row["program_id"]
    
    program_row = db.execute(
        text("SELECT id, program_json FROM programs WHERE id = :id"),
        {"id": str(program_id)}
    ).mappings().first()

    if not program_row:
        raise HTTPException(status_code=404, detail="Program Not Found.")

    program_json = program_row["program_json"]

    events = db.execute(
        text("""
            SELECT *
            FROM exception_events
            WHERE session_id = :sid
            ORDER BY created_at ASC
        """),
        {"sid": str(session_id)}
    ).mappings().all()

    if not events:
        return {"message": "No exceptions found",
                "program_json": program_json,
                "adjustments": []
                }

    all_adjustments = []
    
    for event in events:
        all_adjustments.extend(adjust_workout(dict(event)))

    adjusted_program = apply_adjustments(program_json, all_adjustments)

    db.execute(
        text("UPDATE programs SET program_json = CAST(:pj AS jsonb) WHERE id = :id"),
        {"pj": json.dumps(adjusted_program), "id": program_id}
    )

    for adj in all_adjustments:
        db.execute(
            text("""
                INSERT INTO adjustments (program_id, exercise_name, field, old_value, new_value, reason)
                VALUES (:program_id, :exercise_name, :field, :old_value, :new_value, :reason)
            """),
            {
                "program_id": program_id,
                "exercise_name": adj["exercise_name"],
                "field": adj["field"],
                "old_value": str(adj.get("old_value")) if adj.get("old_value") is not None else None,
                "new_value": str(adj.get("new_value")) if adj.get("new_value") is not None else None,
                "reason": adj.get("reason", "Adjustment applied"),
            }
        )

    db.commit()

    return {"program_id": program_id, "program_json": adjusted_program, "adjustments": all_adjustments}


