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

from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

import json


def main():

    label_path = Path(__file__).parent / "label_map.json"
    with open(label_path, "r") as f:
        labels = json.loads(f)
    
    EVENT_TYPES = labels["event_types"]
    SEVERITIES = labels["severities"]

    base_model = "distilbert-base-uncased"

    data_path = Path(gary.__file__).parent / "data" / "train.csv"
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 32
    epochs = 3
    lr = 2e-5

    # config = {
    #     "batch_size": batch_size,
    #     "epochs": epochs,
    #     "lr": lr
    # }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}.")

    seed = int(os.getenv("SEED", "42"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

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
    warmup_steps = int(0.1 * total_steps)
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    # ---------- Evaluation: loss only (for learning curves) ----------
    def eval_epoch_loss(model, val_loader, device):
        model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                total_loss += out["loss"].item()
                batches += 1

        model.train()
        return total_loss / max(1, batches)


    # ---------- Evaluation: metrics ----------
    def evaluate_metrics(model, val_loader, device):
        model.eval()
        all_event_preds, all_event_true = [], []
        all_sev_preds, all_sev_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)

                ev_preds = out["event_logits"].argmax(dim=-1).cpu().numpy()
                sv_preds = out["severity_logits"].argmax(dim=-1).cpu().numpy()

                all_event_preds.extend(ev_preds)
                all_event_true.extend(batch["event_labels"].cpu().numpy())
                all_sev_preds.extend(sv_preds)
                all_sev_true.extend(batch["severity_labels"].cpu().numpy())

        model.train()

        return {
            "event_acc": accuracy_score(all_event_true, all_event_preds),
            "event_f1": f1_score(all_event_true, all_event_preds, average="macro"),
            "sev_acc": accuracy_score(all_sev_true, all_sev_preds),
            "sev_f1": f1_score(all_sev_true, all_sev_preds, average="macro"),
        }


    # ---------- Training Loop ----------
    train_losses = []
    val_losses = []
    metrics_history = []

    best_val = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        for i, batch in enumerate(train_loader):
            if (i + 1) % 100 == 0 or i == 0: print(f"Batch {i+1}/{len(train_loader)}...")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            running += loss.item()

        # --- Epoch metrics ---
        train_epoch_loss = running / len(train_loader)
        val_epoch_loss = eval_epoch_loss(model, val_loader, device)
        metrics = evaluate_metrics(model, val_loader, device)

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        metrics_history.append(metrics)

        print(
            f"epoch={ep} "
            f"train_loss={train_epoch_loss:.4f} "
            f"val_loss={val_epoch_loss:.4f} "
            f"event_f1={metrics['event_f1']:.3f} "
            f"sev_f1={metrics['sev_f1']:.3f}"
        )

        # --- Save best model ---
        if val_epoch_loss < best_val:
            best_val = val_epoch_loss
            torch.save(model.state_dict(), run_dir / "model.pt")
            print("Saved new best model.")


    # ---------- Save metrics ----------
    summary = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "metrics": metrics_history,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f)

    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr
        }, f)


    # ---------- Plot loss curves ----------
    plt.figure()
    plt.plot(train_losses, marker="o", label="Train")
    plt.plot(val_losses, marker="s", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png")
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
