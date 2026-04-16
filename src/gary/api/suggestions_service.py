"""
Suggestion builders: rule-based note parsing, progression streak detection.
"""
from __future__ import annotations

import re
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from .engine import adjust_workout


def heuristic_note_to_events(note: str, exercise_names: list[str]) -> list[dict]:
    """
    Map free-text note to synthetic events for adjust_workout (no ML).
    """
    if not note or not note.strip():
        return []
    low = note.lower()
    events: list[dict] = []
    target_ex = exercise_names[0] if exercise_names else "Exercise"

    if any(w in low for w in ("too heavy", "heavy", "couldn't finish", "hard")):
        events.append(
            {
                "exercise_name": target_ex,
                "event_type": "too_heavy",
                "severity": "medium",
                "body_area": "none",
            }
        )
    elif any(w in low for w in ("too light", "easy", "lightweight", "breeze")):
        events.append(
            {
                "exercise_name": target_ex,
                "event_type": "too_light",
                "severity": "medium",
                "body_area": "none",
            }
        )
    elif any(w in low for w in ("pain", "hurt", "sore", "tweak")):
        events.append(
            {
                "exercise_name": target_ex,
                "event_type": "pain",
                "severity": "low",
                "body_area": "other",
            }
        )

    return events


def events_to_adjustments(events: list[dict]) -> list[dict]:
    out: list[dict] = []
    for ev in events:
        out.extend(adjust_workout(ev))
    return out


def adjustments_to_results(adjustments: list[dict]) -> list[dict]:
    """Serialize for JSON payload storage."""
    return [
        {
            "exercise_name": a["exercise_name"],
            "field": a["field"],
            "multiplier": a.get("multiplier"),
            "reason": a.get("reason"),
            "body_part": a.get("body_part"),
        }
        for a in adjustments
    ]


def last_three_no_note_completions(
    db: Session, program_id: UUID, day_name: str
) -> tuple[bool, list[dict]]:
    rows = db.execute(
        text(
            """
            SELECT id, note, completed_at
            FROM day_completions
            WHERE program_id = :pid AND day_name = :dn
            ORDER BY completed_at DESC
            LIMIT 3
            """
        ),
        {"pid": str(program_id), "dn": day_name},
    ).mappings().all()

    if len(rows) < 3:
        return False, [dict(r) for r in rows]

    for r in rows:
        n = r.get("note")
        if n and str(n).strip():
            return False, [dict(x) for x in rows]

    return True, [dict(r) for r in rows]


def build_progression_payload(program_json: dict, day_name: str) -> list[dict]:
    """Conservative load bump for exercises on a given day."""
    adjustments: list[dict] = []
    for day in program_json.get("days", []):
        if day.get("name") != day_name:
            continue
        for ex in day.get("exercises", []):
            name = ex.get("name")
            load = ex.get("load")
            if name and load is not None:
                adjustments.append(
                    {
                        "exercise_name": name,
                        "field": "load",
                        "multiplier": 1.025,
                        "reason": "Progression: 3 completions without notes",
                        "body_part": None,
                    }
                )
    return adjustments
