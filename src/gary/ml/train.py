import os
import math
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup

import matplotlib.pyplot as plt

from gary.ml.model import MultiHeadClassifier
from gary.ml.dataset import ExceptionsDataset, LabelMaps
import gary
from pathlib import Path

from datetime import datetime

from sklearn.metrics import f1_score, matthews_corrcoef

import json

EVENT_TYPES = ["none", "too_heavy", "too_light", "pain", "time", "equipment", "form", "other"]
SEVERITIES = ["na", "low", "medium", "high"]


def main():

    base_model = "distilbert-base-uncased"

    data_path = Path(gary.__file__).parent / "data" / "train.csv"
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = PROJECT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    epochs = 3
    lr = 2e-5

    config = {
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}.")

    seed = int(os.getenv("SEED", "42"))

    label_maps = LabelMaps(
        event2id={k: i for i, k in enumerate(EVENT_TYPES)},
        sev2id={k: i for i, k in enumerate(SEVERITIES)}
    )

    ds = ExceptionsDataset(data_path, base_model, label_maps, max_len=256)

    val_frac = 0.15
    val_size = int(val_frac * len(ds))
    train_size = len(ds) - val_size

    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    
    device = torch.device(device)

    model = MultiHeadClassifier(base_model, num_event_types=len(EVENT_TYPES), num_severities=len(SEVERITIES))
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(val_frac * total_steps)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    def eval_loss():
        model.eval()
        losses = []

        all_event_preds, all_event_labels = [], []
        all_sev_preds, all_sev_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                losses.append(out["loss"].item())

                # adjust these keys to match your dataset
                event_labels = batch["event_label"]
                sev_labels   = batch["sev_label"]

                # adjust these keys to match your model output
                event_logits = out["event_logits"]
                sev_logits   = out["sev_logits"]

                event_preds = torch.argmax(event_logits, dim=1)
                sev_preds   = torch.argmax(sev_logits, dim=1)

                all_event_preds.extend(event_preds.cpu().numpy())
                all_event_labels.extend(event_labels.cpu().numpy())

                all_sev_preds.extend(sev_preds.cpu().numpy())
                all_sev_labels.extend(sev_labels.cpu().numpy())

        model.train()

        val_loss = float(sum(losses) / max(1, len(losses)))

        metrics = {
            "val_loss": val_loss,
            "event_macro_f1": f1_score(all_event_labels, all_event_preds, average="macro"),
            "event_mcc": matthews_corrcoef(all_event_labels, all_event_preds),
            "sev_macro_f1": f1_score(all_sev_labels, all_sev_preds, average="macro"),
            "sev_mcc": matthews_corrcoef(all_sev_labels, all_sev_preds),
        }


        return losses, metrics
    
    print_interval = 50
    best = math.inf
    train_losses = []


    for ep in range(1, epochs + 1):
        print(f"Beginning epoch {ep}/{epochs}...")
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            if (step - 1) % print_interval == 0:
                print(f"\rBatch: {step}/{len(train_loader)}")

            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            running += loss.item()


        val_losses, metrics = eval_loss()
        train_avg = running / max(1, len(train_loader))
        train_losses.append(train_avg)
        print(f"epoch={ep} train_loss={train_avg:.4f} val_loss={metrics['val_loss']:.4f}")

        # Save best model

        if metrics["val_loss"] < best:
            torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
            with open(os.path.join(run_dir, "labels.txt"), "w", encoding="utf-8") as f:
                f.write("EVENT_TYPES=" + ",".join(EVENT_TYPES) + "\n")
                f.write("SEVERITIES=" + ",".join(SEVERITIES) + "\n")
            print(f"Saved best to {run_dir}")
            best = metrics["val_loss"]

            plt.plot(train_losses, label="Train-Losses")
            plt.plot(val_losses, label="Val-Losses")
            plt.legend()
            plt.savefig(run_dir / 'losses.png')
            plt.close()

            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f)
            
            with open(run_dir / "metrics.json"):
                json.dump(metrics, f)



if __name__ == "__main__":
    main()
