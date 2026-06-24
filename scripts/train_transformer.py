from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trans_mpc.config import TrainingConfig, TransformerConfig
from trans_mpc.dataset import WindowedRiskDataset, train_val_split
from trans_mpc.transformer_model import RiskTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Train for a few epochs.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the configured full-training epoch count.")
    args = parser.parse_args()

    data = np.load(ROOT / "data" / "dataset.npz")
    windows = data["windows"]
    targets = data["targets"]
    train_cfg = TrainingConfig()
    model_cfg = TransformerConfig()
    x_train, y_train, x_val, y_val = train_val_split(windows, targets, train_cfg.val_fraction, seed=7)

    train_loader = DataLoader(WindowedRiskDataset(x_train, y_train), batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(WindowedRiskDataset(x_val, y_val), batch_size=train_cfg.batch_size)

    model = RiskTransformer(
        input_dim=model_cfg.input_dim,
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        k_v_min=model_cfg.k_v_min,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    epochs = train_cfg.quick_epochs if args.quick else (args.epochs or train_cfg.epochs)
    history = []
    best_val = float("inf")
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer)
        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, loss_fn, None)
        history.append((epoch, train_loss, val_loss))
        print(f"epoch={epoch} train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": model_cfg.__dict__}, ckpt_dir / "best_transformer.pt")

    out = np.asarray(history, dtype=float)
    np.savetxt(ckpt_dir / "training_history.csv", out, delimiter=",", header="epoch,train_loss,val_loss", comments="")
    print(f"saved checkpoint: {ckpt_dir / 'best_transformer.pt'}")


def _run_epoch(model, loader, loss_fn, optimizer):
    total = 0.0
    count = 0
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(x)
        count += len(x)
    return total / max(count, 1)


if __name__ == "__main__":
    main()
