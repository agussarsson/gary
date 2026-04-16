"""
Gemini-first program generation with deterministic fallback.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional  # noqa: I001

from .schemas import ProgramGenerateRequest, ProgramJSON, ProgramDay, Exercise

SYSTEM_PROMPT = """You are a strength coaching assistant. Output ONLY valid JSON matching this exact structure (no markdown, no commentary):
{
  "split": "string describing split e.g. PPL",
  "days_per_week": <integer 1-7>,
  "goal": "string",
  "experience_level": "string",
  "notes": "string or null",
  "days": [
    {
      "name": "day name e.g. Push",
      "focus": "short focus string",
      "exercises": [
        {
          "name": "exercise name",
          "sets": <int 1-10>,
          "reps": <int 1-30>,
          "load": <number or null>,
          "progression_rule": "increase_load_if_completed"
        }
      ]
    }
  ]
}
days length MUST equal days_per_week. Each day must have at least one exercise."""


def _fallback_program(req: ProgramGenerateRequest) -> ProgramJSON:
    n = req.days_per_week
    day_templates = [
        ("Push", "Chest shoulders triceps", [
            ("Bench Press", 3, 6, 60.0),
            ("Overhead Press", 3, 8, 35.0),
            ("Tricep Pushdown", 3, 12, 20.0),
        ]),
        ("Pull", "Back biceps", [
            ("Barbell Row", 3, 6, 50.0),
            ("Lat Pulldown", 3, 10, 45.0),
            ("Hammer Curl", 3, 12, 12.0),
        ]),
        ("Legs", "Lower body", [
            ("Back Squat", 3, 6, 80.0),
            ("Romanian Deadlift", 3, 8, 70.0),
            ("Leg Curl", 3, 12, 30.0),
        ]),
    ]
    bodyweight_templates = [
        ("Push", "Upper push endurance", [
            ("Push-Up", 4, 12, None),
            ("Pike Push-Up", 3, 8, None),
            ("Bench Dip", 3, 12, None),
        ]),
        ("Pull", "Upper pull control", [
            ("Inverted Row", 4, 10, None),
            ("Doorway Row", 3, 12, None),
            ("Biceps Curl Iso Hold", 3, 20, None),
        ]),
        ("Legs", "Lower body strength", [
            ("Air Squat", 4, 15, None),
            ("Reverse Lunge", 3, 10, None),
            ("Single-Leg Glute Bridge", 3, 12, None),
        ]),
    ]
    templates = bodyweight_templates if req.workout_style == "body_weight" else day_templates
    days: list[ProgramDay] = []
    for i in range(n):
        name, focus, exs = templates[i % len(templates)]
        exercises = [
            Exercise(
                name=e[0],
                sets=e[1],
                reps=e[2],
                load=e[3],
                progression_rule="increase_load_if_completed",
            )
            for e in exs
        ]
        days.append(ProgramDay(name=f"{name} (Day {i + 1})", focus=focus, exercises=exercises))

    return ProgramJSON(
        split=f"{n}-day split",
        days_per_week=n,
        goal=req.goal,
        experience_level=req.experience_level,
        notes="Generated offline template. Connect Gemini for AI-tailored plans.",
        days=days,
    )


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def generate_with_gemini(req: ProgramGenerateRequest) -> tuple[Optional[ProgramJSON], str]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "no_api_key"

    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return None, "google_generativeai_not_installed"

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    prefs = ", ".join(req.preferences) if req.preferences else "none"
    workout_style = req.workout_style
    user_prompt = f"""{SYSTEM_PROMPT}

User goal: {req.goal}
Days per week: {req.days_per_week}
Experience: {req.experience_level}
Preferences: {prefs}
Workout style: {workout_style}
"""

    try:
        resp = model.generate_content(user_prompt)
        raw = (resp.text or "").strip()
    except Exception as exc:
        return None, f"gemini_error:{exc!s}"

    data = _extract_json(raw)
    if not data:
        return None, "invalid_json"

    try:
        pj = ProgramJSON.model_validate(data)
        return pj, "ok"
    except Exception as exc:
        return None, f"validation_error:{exc!s}"


def generate_program(req: ProgramGenerateRequest) -> tuple[ProgramJSON, str, Optional[str]]:
    """
    Returns (program_json, source, message).
    source is 'gemini' or 'fallback'.
    """
    pj, reason = generate_with_gemini(req)
    if pj is not None:
        return pj, "gemini", None

    fb = _fallback_program(req)
    msg = f"Used deterministic fallback ({reason})."
    return fb, "fallback", msg


def refine_program_json(current: ProgramJSON, feedback: str) -> tuple[Optional[ProgramJSON], str, Optional[str]]:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        fb = current.model_copy(deep=True)
        if fb.notes:
            fb.notes = f"{fb.notes}\nUser feedback: {feedback}"
        else:
            fb.notes = f"User feedback: {feedback}"
        return fb, "fallback", "no_api_key"

    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        fb = current.model_copy(deep=True)
        return fb, "fallback", "google_generativeai_not_installed"

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    prompt = f"""{SYSTEM_PROMPT}

Current program JSON:
{current.model_dump_json()}

User requested changes:
{feedback}

Return the FULL updated program as JSON only."""

    try:
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
    except Exception as exc:
        return None, "error", str(exc)

    data = _extract_json(raw)
    if not data:
        return None, "error", "invalid_json"

    try:
        pj = ProgramJSON.model_validate(data)
        return pj, "gemini", None
    except Exception as exc:
        return None, "error", str(exc)
