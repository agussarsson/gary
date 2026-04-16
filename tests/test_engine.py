from gary.api.engine import apply_adjustments, round_load_floor


def test_round_load_floor_under_12_5():
    assert round_load_floor(11.9) == 11.0
    assert round_load_floor(12.49) == 12.0


def test_round_load_floor_2_5_steps():
    assert round_load_floor(17.93) == 17.5
    assert round_load_floor(19.2) == 17.5
    assert round_load_floor(22.6) == 22.5


def test_apply_adjustments_uses_floor_rounding():
    program = {
        "days": [
            {
                "name": "Pull",
                "exercises": [
                    {
                        "name": "Barbell Row",
                        "sets": 3,
                        "reps": "8-10",
                        "load": 16.01,
                        "progression_rule": "increase_load_if_completed",
                    }
                ],
            }
        ]
    }
    adjusted = apply_adjustments(
        program,
        [
            {
                "exercise_name": "Barbell Row",
                "field": "load",
                "multiplier": 1.12,
            }
        ],
    )
    # 16.01 * 1.12 = 17.9312 -> floor to nearest 2.5 = 17.5
    assert adjusted["days"][0]["exercises"][0]["load"] == 17.5
