from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import config  # noqa: E402
from backend.fight_detector import (  # noqa: E402
    FightDetector,
    _bar_center,
    _champion_bars,
    _combat_health_bars,
    _enemy_bars_near_player,
    _healthbar_engagement_scores,
    _select_player_health_bar,
    boundaries_from_scores,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit fight detection scores and health-bar evidence for a video."
    )
    parser.add_argument("video", type=Path, help="Source .mp4 to inspect.")
    parser.add_argument("--start", type=float, default=0.0, help="First timestamp to print.")
    parser.add_argument("--end", type=float, default=None, help="Last timestamp to print.")
    parser.add_argument("--step", type=float, default=1.0, help="Printed row spacing in seconds.")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Frame sampling rate used for detector reproduction.",
    )
    parser.add_argument(
        "--candidate-min-width",
        type=int,
        default=80,
        help="Extra audit-only red-bar width floor for borderline champion candidates.",
    )
    parser.add_argument(
        "--skip-videomae",
        action="store_true",
        help="Only report health-bar evidence; skip VideoMAE scoring.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    return parser


def _decode_samples(video_path: Path, sample_fps: float) -> tuple[np.ndarray, np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, round(source_fps / sample_fps))
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    frame_index = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_index % step == 0:
            resized = cv2.resize(bgr, (1920, 1080))
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            timestamps.append(frame_index / source_fps)
        frame_index += 1

    cap.release()
    return np.asarray(frames, dtype=np.uint8), np.asarray(timestamps, dtype=np.float32), source_fps


def _near_player(
    bars: list[tuple[int, int, int, int]],
    player_bar: tuple[int, int, int, int] | None,
) -> list[tuple[int, int, int, int]]:
    if player_bar is None:
        return bars
    px, py = _bar_center(player_bar)
    nearby = []
    for bar in bars:
        bx, by = _bar_center(bar)
        if abs(bx - px) <= 420 and abs(by - py) <= 260:
            nearby.append(bar)
    return nearby


def _box_summary(boxes: list[tuple[int, int, int, int]]) -> str:
    return json.dumps([{"x": x, "y": y, "w": w, "h": h} for x, y, w, h in boxes], separators=(",", ":"))


def _score_at(scores: list[float], second: int) -> float | None:
    if 0 <= second < len(scores):
        return float(scores[second])
    return None


def _row_for_timestamp(
    frame: np.ndarray,
    timestamp: float,
    healthbar_scores: list[float],
    combined_scores: list[float],
    candidate_min_width: int,
) -> dict[str, object]:
    red_bars, green_bars = _combat_health_bars(frame)
    player_bar = _select_player_health_bar(green_bars)
    champion_reds = _champion_bars(red_bars)
    nearby_champion_reds = _enemy_bars_near_player(red_bars, player_bar)
    candidate_reds = [bar for bar in red_bars if bar[2] >= candidate_min_width]
    nearby_candidate_reds = _near_player(candidate_reds, player_bar)
    second = int(timestamp)

    return {
        "time": round(float(timestamp), 3),
        "window_second": second,
        "combined_score": _score_at(combined_scores, second),
        "healthbar_score": _score_at(healthbar_scores, second),
        "player_bar": _box_summary([player_bar] if player_bar is not None else []),
        "red_bars": _box_summary(red_bars),
        "champion_red_bars": _box_summary(champion_reds),
        "candidate_red_bars": _box_summary(candidate_reds),
        "nearby_champion_enemy_count": len(nearby_champion_reds),
        "nearby_candidate_enemy_count": len(nearby_candidate_reds),
        "red_widths": json.dumps([bar[2] for bar in red_bars], separators=(",", ":")),
        "champion_red_widths": json.dumps([bar[2] for bar in champion_reds], separators=(",", ":")),
        "candidate_red_widths": json.dumps([bar[2] for bar in candidate_reds], separators=(",", ":")),
    }


def _format_score(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _print_summary(combined_scores: list[float], healthbar_scores: list[float], duration: float) -> None:
    if combined_scores:
        peak = int(np.argmax(np.asarray(combined_scores)))
        print(f"combined_peak_second={peak} combined_peak_score={combined_scores[peak]:.3f}")
        start, end, flags = boundaries_from_scores(combined_scores, duration)
        print(f"detected_fight_start={start:.3f} detected_fight_end={end:.3f} flags={flags}")
    if healthbar_scores:
        health_peak = int(np.argmax(np.asarray(healthbar_scores)))
        print(f"healthbar_peak_second={health_peak} healthbar_peak_score={healthbar_scores[health_peak]:.3f}")
    print(f"fight_threshold={config.FIGHT_CONFIDENCE_THRESHOLD:.3f}")
    print(f"champion_bar_min_width={config.COMBAT_CHAMPION_HEALTHBAR_MIN_WIDTH}px")


def main() -> None:
    args = _build_parser().parse_args()
    frames, timestamps, source_fps = _decode_samples(args.video, args.sample_fps)
    if len(frames) == 0:
        raise SystemExit("No frames decoded.")

    duration = float(timestamps[-1])
    end = args.end if args.end is not None else duration
    healthbar_scores = _healthbar_engagement_scores(frames, timestamps)
    combined_scores: list[float] = []
    if not args.skip_videomae:
        combined_scores = FightDetector().score_windows(frames, timestamps)

    print(f"video={args.video}")
    print(f"source_fps={source_fps:.3f} sampled_frames={len(frames)} sampled_duration={duration:.3f}")
    _print_summary(combined_scores, healthbar_scores, duration)
    print()
    print("time  win  score  hb     nearChamp  nearCand  redWidths  champWidths  candWidths")

    rows: list[dict[str, object]] = []
    for target in np.arange(args.start, end + 0.0001, args.step):
        index = int(np.argmin(np.abs(timestamps - target)))
        row = _row_for_timestamp(
            frames[index],
            float(timestamps[index]),
            healthbar_scores,
            combined_scores,
            args.candidate_min_width,
        )
        rows.append(row)
        print(
            f"{row['time']:>5} "
            f"{row['window_second']:>4} "
            f"{_format_score(row['combined_score']):>6} "
            f"{_format_score(row['healthbar_score']):>6} "
            f"{row['nearby_champion_enemy_count']:>9} "
            f"{row['nearby_candidate_enemy_count']:>8} "
            f"{row['red_widths']:<10} "
            f"{row['champion_red_widths']:<12} "
            f"{row['candidate_red_widths']}"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print()
        print(f"wrote_csv={args.csv}")


if __name__ == "__main__":
    main()
