import json
import math
import re
from pathlib import Path

_aliases_path = Path(__file__).resolve().parent / "aliases.json"
try:
    with open(_aliases_path, "r", encoding="utf-8") as f:
        ALIASES = json.load(f)
except FileNotFoundError:
    ALIASES = {}


def adjust_workout(event: dict) -> list[dict]:
    """
    Algorithmically decides how the workout should be updated based on the event.
    """
    adjustments: list[dict] = []

    multiplier = 1.0

    event_type = event["event_type"]
    event_severity = event["severity"]
    event_bodyarea = event["body_area"]
    event_exercise = event["exercise_name"]

    if event_type == "too_heavy":
        if event_severity == "high":
            multiplier = 0.75
        elif event_severity == "medium":
            multiplier = 0.85
        else:
            multiplier = 0.92

        adjustments.append(
            {
                "exercise_name": event_exercise,
                "field": "load",
                "multiplier": multiplier,
                "reason": "Too heavy",
                "body_part": event_bodyarea,
            }
        )

    elif event_type == "too_light":
        if event_severity == "high":
            multiplier = 1.20
        elif event_severity == "medium":
            multiplier = 1.12
        else:
            multiplier = 1.08

        adjustments.append(
            {
                "exercise_name": event_exercise,
                "field": "load",
                "multiplier": multiplier,
                "reason": "Too light",
                "body_part": event_bodyarea,
            }
        )

    elif event_type == "pain":
        if event_severity == "high":
            multiplier = 0.5
        elif event_severity == "medium":
            multiplier = 0.65
        else:
            multiplier = 0.8

        adjustments.append(
            {
                "exercise_name": event_exercise,
                "field": "pain",
                "multiplier": multiplier,
                "reason": f"Pain in {event_bodyarea}",
                "body_part": event_bodyarea,
            }
        )

    return adjustments


def apply_adjustments(program_json: dict, adjustments: list[dict]) -> dict:
    """
    Applies load multipliers from adjustments to matching exercises.
    Skips pain field entries (handled elsewhere or future reps logic).
    """
    for adj in adjustments:
        exercise_name = adj["exercise_name"]
        field = adj["field"]
        multiplier = adj.get("multiplier")

        if field != "load" or multiplier is None:
            continue

        for day in program_json["days"]:
            for exercise in day["exercises"]:
                if exercise["name"] != exercise_name:
                    continue
                current = exercise.get("load")
                if current is None:
                    continue
                raw = float(current) * float(multiplier)
                exercise["load"] = round_load_floor(raw)

    return program_json


def round_load_floor(load: float) -> float:
    """
    Gym-friendly floor rounding:
    - < 12.5 -> floor to nearest integer
    - >= 12.5 -> floor to nearest 2.5 step
    """
    value = float(load)
    if value < 12.5:
        return float(math.floor(value))
    return math.floor(value / 2.5) * 2.5


def normalize_text(s: str) -> str:
    lowercase_str = s.lower()
    return re.sub(r"[^a-z0-9]", "", lowercase_str)


def decide_exercise_from_text(user_text: str) -> list:
    if not ALIASES:
        return []
    normalized_text = normalize_text(user_text)
    mentioned = []
    for exercise, aliases in ALIASES.items():
        if any(alias in normalized_text for alias in aliases):
            mentioned.append(exercise)
    return mentioned


def construct_model_input(exercises: list, user_text: str) -> str:
    if not exercises:
        return f"User note: {user_text}"
    ex = exercises[0]
    return f"Exercise: {ex}\nSession exercises: {', '.join(exercises)}\nUser note: {user_text}"
