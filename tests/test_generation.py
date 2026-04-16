"""Generation: deterministic fallback when Gemini is unavailable."""
import pytest

from gary.api.generation import _fallback_program, generate_program
from gary.api.schemas import ProgramGenerateRequest


def test_fallback_program_shape():
    req = ProgramGenerateRequest(
        goal="Strength",
        days_per_week=3,
        experience_level="beginner",
    )
    pj = _fallback_program(req)
    assert pj.days_per_week == 3
    assert len(pj.days) == 3
    assert all(len(d.exercises) >= 1 for d in pj.days)
    assert all(isinstance(ex.reps, int) for d in pj.days for ex in d.exercises)


def test_generate_program_uses_fallback_without_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    req = ProgramGenerateRequest(
        goal="Test",
        days_per_week=2,
        experience_level="novice",
    )
    pj, source, msg = generate_program(req)
    assert source == "fallback"
    assert pj.days_per_week == 2
    assert msg and "fallback" in msg.lower()


def test_generate_program_respects_env_key_present(monkeypatch: pytest.MonkeyPatch):
    """If a key is set, path attempts Gemini (may still fall back on failure)."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    # Avoid real network: force generate_with_gemini to return None
    import gary.api.generation as gen

    monkeypatch.setattr(
        gen,
        "generate_with_gemini",
        lambda req: (None, "no_api_key_simulated"),
    )
    req = ProgramGenerateRequest(
        goal="Test goal",
        days_per_week=1,
        experience_level="intermediate",
    )
    pj, source, _msg = generate_program(req)
    assert source == "fallback"
    assert len(pj.days) == 1


def test_fallback_respects_body_weight_style():
    req = ProgramGenerateRequest(
        goal="General fitness",
        days_per_week=2,
        experience_level="beginner",
        workout_style="body_weight",
    )
    pj = _fallback_program(req)
    assert all(ex.load is None for d in pj.days for ex in d.exercises)
    assert all(isinstance(ex.reps, int) for d in pj.days for ex in d.exercises)
