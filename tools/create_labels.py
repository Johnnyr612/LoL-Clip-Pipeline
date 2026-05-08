from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a template labels JSON file for LoL fight boundary training.")
    parser.add_argument("--clips_dir", required=True, type=Path, help="Folder containing training .mp4 files.")
    parser.add_argument("--output", required=True, type=Path, help="Output labels_all.json path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clips_dir = args.clips_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not clips_dir.exists() or not clips_dir.is_dir():
        raise SystemExit(f"clips_dir does not exist or is not a directory: {clips_dir}")

    labels = [
        {"filename": path.name, "fight_start": 0.0, "fight_end": 0.0}
        for path in sorted(clips_dir.glob("*.mp4"), key=lambda item: item.name.lower())
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")

    print(f"Found {len(labels)} .mp4 clips in {clips_dir}", flush=True)
    print(f"Saved label template to {output}", flush=True)
    print("Fill in fight_start and fight_end values before training.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
