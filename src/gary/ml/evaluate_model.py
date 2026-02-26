import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gary
from pathlib import Path

from gary.ml.model import MultiHeadClassifier
from gary.ml.dataset import ExceptionsDataset, LabelMaps
import gary
from pathlib import Path

from sklearn.metrics import f1_score, matthews_corrcoef

import json

import os

EVENT_TYPES = ["none", "too_heavy", "too_light", "pain", "time", "equipment", "form", "other"]
SEVERITIES = ["na", "low", "medium", "high"]

def main():

    base_model = "distilbert-base-uncased"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}.")

    data_path = Path(gary.__file__).parent / "data" / "test.csv"
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    out_dir = PROJECT_ROOT / "artifacts"
    model_dir = out_dir / "best_model.pt"
    
    config = 

    seed = int(os.getenv("SEED", "42"))

    label_maps = LabelMaps(
        event2id={k: i for i, k in enumerate(EVENT_TYPES)},
        sev2id={k: i for i, k in enumerate(SEVERITIES)}
    )

    test_ds = ExceptionsDataset(data_path, base_model, label_maps, max_len=256)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    model = MultiHeadClassifier(base_model, num_event_types=len(EVENT_TYPES), num_severities=len(SEVERITIES))
    model.to(device)

    state_dict = torch.load(model_dir, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    
    all_event_preds = []
    all_event_labels = []

    all_sev_preds = []
    all_sev_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)

            # adjust keys if needed
            event_logits = out["event_logits"]
            sev_logits = out["sev_logits"]

            event_preds = torch.argmax(event_logits, dim=1)
            sev_preds = torch.argmax(sev_logits, dim=1)

            all_event_preds.extend(event_preds.cpu().numpy())
            all_event_labels.extend(batch["event_label"].cpu().numpy())

            all_sev_preds.extend(sev_preds.cpu().numpy())
            all_sev_labels.extend(batch["sev_label"].cpu().numpy())

    event_f1 = f1_score(all_event_labels, all_event_preds, average="macro")
    event_mcc = matthews_corrcoef(all_event_labels, all_event_preds)

    sev_f1 = f1_score(all_sev_labels, all_sev_preds, average="macro")
    sev_mcc = matthews_corrcoef(all_sev_labels, all_sev_preds)

    metrics = {
        "event_macro_f1": event_f1,
        "event_mcc": event_mcc,
        "sev_macro_f1": sev_f1,
        "sev_mcc": sev_mcc
    }

    with open(out_dir, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()