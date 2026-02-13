

def adjust_workout(event):
    """
    Algorithmicly decides how the workout should be updated based on the event.

    params:
        event :dict:
            dictionary/json-type of an event
    
    returns:
        adjustments :dict:
            list of adjustments
    """

    adjustments = []

    multiplier = 1

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
            # low
            multiplier = 0.92

        adjustments.append({
            "exercise_name": event_exercise,
            "field": "load",
            "multiplier": multiplier,
            "reason": "Too heavy",
            "body_part": event_bodyarea
        })
    
    elif event_type == "too_light":
        if event_severity == "high":
            multiplier = 1.20
        elif event_severity == "medium":
            multiplier = 1.12
        else:
            # low
            multiplier = 1.08

        adjustments.append({
            "exercise_name": event_exercise,
            "field": "load",
            "multiplier": multiplier,
            "reason": "Too light",
            "body_part": event_bodyarea
        })

    elif event_type == "pain":
        if event_severity == "high":
            multiplier = 0.5
        elif event_severity == "medium":
            multiplier = 0.65
        else:
            # low
            multiplier = 0.8

        adjustments.append({
            "exercise_name": event_exercise,
            "field": "pain",
            "multiplier": multiplier,
            "reason": f"Pain in {event_bodyarea}",
            "body_part": event_bodyarea
        })

    return adjustments

def apply_adjustments(program_json, adjustments):
    """
    Algorithmicly applies adjustments to a program.

    params:
        program_json :dict:
            dictionary/json-type containing the full workout program
    
        adjustments :dict:
            list of adjustments

    returns:
        program_json :dict:
            updated workout program
    """
    for adj in adjustments:
        exercise_name = adj["exercise_name"]
        field = adj["field"]
        multiplier = adj("multiplier")

        for day in program_json["days"]:
            for exercise in day["exercises"]:

                if exercise["name"] == exercise_name:

                    if field == "load" and multiplier is not None:
                        exercise["load"] = round(
                            exercise["load"] * multiplier, 2
                        )

    return program_json