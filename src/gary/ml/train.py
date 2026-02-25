import os
import math
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup

from gary.ml.model import MultiHeadClassifier
from gary.ml.dataset import ExceptionsDataset, LabelMaps
import gary
from pathlib import Path

EVENT_TYPES = ["none", "too_heavy", "too_light", "pain", "time", "equipment", "form", "other"]
SEVERITIES = ["na", "low", "medium", "high"]


def main():
    base_model = "distilbert-base-uncased"
    data_path = Path(gary.__file__).parent / "data" / "train.csv"
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    out_dir = PROJECT_ROOT / "artifacts" / "exception_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size = 16
    epochs = 3
    lr = 2e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}.")

    seed = int(os.getenv("SEED", "42"))


    os.makedirs(out_dir, exist_ok=True)

    label_maps = LabelMaps(
        event2id={k: i for i, k in enumerate(EVENT_TYPES)},
        sev2id={k: i for i, k in enumerate(SEVERITIES)}
    )

    ds = ExceptionsDataset(data_path, base_model, label_maps, max_len=256)

    print(ds)

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
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                losses.append(out["loss"].item())
        model.train()
        return float(sum(losses) / max(1, len(losses)))
    
    best = math.inf
    for ep in range(1, epochs + 1):
        print(f"Beginning epoch {ep}...")
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            running += loss.item()

        val = eval_loss()
        train_avg = running / max(1, len(train_loader))
        print(f"epoch={ep} train_loss={train_avg:.4f} val_loss={val:.4f}")

        # Save best model

        if val < best:
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
            with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as f:
                f.write("EVENT_TYPES=" + ",".join(EVENT_TYPES) + "\n")
                f.write("SEVERITIES=" + ",".join(SEVERITIES) + "\n")
            print(f"Saved best to {out_dir}")


if __name__ == "__main__":
    main()