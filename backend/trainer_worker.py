from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    metrics_dir = config.PROJECT_ROOT / "checkpoints" / f"slice_{args.slice}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"

    # Placeholder loop keeps the coordinator/SSE contract live until the real
    # labeled training set and ML stack are installed.
    best_val = float("inf")
    patience_left = 4
    for epoch in range(1, 26):
        train_loss = max(0.05, 1.0 / epoch)
        val_loss = max(0.04, 1.15 / (epoch + 0.5))
        accuracy = min(0.99, 0.55 + epoch * 0.015)
        metrics = {
            "run_id": args.run_id,
            "slice": int(args.slice),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
        }
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        if val_loss < best_val:
            best_val = val_loss
            patience_left = 4
            checkpoint = config.PROJECT_ROOT / "checkpoints" / "videomae_lol_best.pt"
            checkpoint.write_text("placeholder checkpoint; replace with torch state_dict after training\n", encoding="utf-8")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
        time.sleep(0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
