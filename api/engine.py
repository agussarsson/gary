

def adjust_workout(event):

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
            "reason": "Too heavy"
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
            "reason": "Too light"
        })

    elif event_type == "pain":
        # here: could also report severity of pain, but that introduces some considerations
        # such as e.g.: high pain => switch exercise, low pain => lower weight, however, 
        # if pain doesn't go away?
        adjustments.append({
            "exercise_name": event_exercise,
            "field": "exercise_swap",
            "multiplier": None,
            "reason": f"Pain in {event_bodyarea}"
        })

    return adjustments