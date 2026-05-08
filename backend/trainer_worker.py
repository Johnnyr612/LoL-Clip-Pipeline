from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

try:
    from . import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend import config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clips-dir", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    metrics_dir = output_dir / f"slice_{args.slice}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    print(
        f"run_id={args.run_id} slice={args.slice} clips_dir={args.clips_dir} labels={args.labels} "
        f"epochs={args.epochs} batch_size={args.batch_size} output_dir={output_dir}",
        flush=True,
    )

    # Placeholder loop keeps the coordinator/SSE contract live until the real
    # labeled training set and ML stack are installed.
    best_val = float("inf")
    patience_left = 4
    for epoch in range(1, args.epochs + 1):
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
        print(json.dumps(metrics), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            patience_left = 4
            checkpoint = output_dir / "videomae_lol_best.pt"
            checkpoint.write_text("placeholder checkpoint; replace with torch state_dict after training\n", encoding="utf-8")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
        time.sleep(0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
