import torch
import json
from pathlib import Path
from transformers import AutoTokenizer

from gary.ml.model import MultiHeadClassifier


class ExceptionModel:

    def __init__(self, model_dir: str):

        model_dir = Path(model_dir)

        with open(model_dir / "label_map.json") as f:
            labels = json.load(f)

        self.event_types = labels["event_types"]
        self.severities = labels["severities"]

        with open(model_dir / "config.json") as f:
            config = json.load(f)

        base_model = "distilbert-base-uncased"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        self.model = MultiHeadClassifier(
            base_model,
            num_event_types=len(self.event_types),
            num_severities=len(self.severities),
        )

        state_dict = torch.load(model_dir / "model.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, exercise_name, session_exercises, text):

        prompt = (
            f"Exercise: {exercise_name}\n"
            f"Session exercises: {session_exercises}\n"
            f"User note: {text}"
        )

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        with torch.no_grad():
            out = self.model(**enc)

        event_probs = torch.softmax(out["event_logits"], dim=1)
        sev_probs = torch.softmax(out["severity_logits"], dim=1)

        event_id = torch.argmax(event_probs).item()
        sev_id = torch.argmax(sev_probs).item()

        return {
            "event_type": self.event_types[event_id],
            "severity": self.severities[sev_id],
            "confidence": float(event_probs[0][event_id])
        }