"""Suggestion helpers and progression rule."""
from unittest.mock import MagicMock

import pytest

from gary.api.suggestions_service import (
    build_progression_payload,
    heuristic_note_to_events,
    last_three_no_note_completions,
)


def test_heuristic_too_heavy():
    events = heuristic_note_to_events("Last set was too heavy", ["Bench Press"])
    assert len(events) == 1
    assert events[0]["event_type"] == "too_heavy"
    assert events[0]["exercise_name"] == "Bench Press"


def test_heuristic_empty():
    assert heuristic_note_to_events("", ["A"]) == []
    assert heuristic_note_to_events("   ", ["A"]) == []


def test_build_progression_payload_load_bump():
    program = {
        "days": [
            {
                "name": "Push",
                "exercises": [
                    {"name": "Bench Press", "load": 100},
                    {"name": "Fly", "load": None},
                ],
            }
        ]
    }
    adjs = build_progression_payload(program, "Push")
    assert len(adjs) == 1
    assert adjs[0]["exercise_name"] == "Bench Press"
    assert adjs[0]["field"] == "load"
    assert adjs[0]["multiplier"] == pytest.approx(1.025)


def test_last_three_no_note_triggers():
    """Three most recent rows with empty notes -> True."""
    mock_db = MagicMock()
    mock_db.execute.return_value.mappings.return_value.all.return_value = [
        {"note": None, "id": "a"},
        {"note": "  ", "id": "b"},
        {"note": None, "id": "c"},
    ]

    from uuid import uuid4

    ok, _out = last_three_no_note_completions(mock_db, uuid4(), "Push")
    assert ok is True


def test_last_three_insufficient_rows():
    mock_db = MagicMock()
    mock_db.execute.return_value.mappings.return_value.all.return_value = []

    from uuid import uuid4

    ok, _rows = last_three_no_note_completions(mock_db, uuid4(), "Push")
    assert ok is False
