from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, models
from .caption_gen import CaptionGenerator
from .cropper import AdaptiveCropper
from .encoder import EncoderError, VideoEncoder
from .fight_detector import FightDetector
from .frame_io import FrameDecodeError, decode_video
from .minimap_detector import ChampionResult, MinimapDetector


@dataclass(frozen=True)
class ValidationResult:
    duration: float
    has_audio: bool


class InputValidationError(ValueError):
    pass


def validate_input(path: Path) -> ValidationResult:
    if not path.exists() or path.suffix.lower() != ".mp4":
        raise InputValidationError("Input must be an existing .mp4 file")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return ValidationResult(60.0, False)
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise InputValidationError(result.stderr)
    payload = json.loads(result.stdout)
    duration = float(payload.get("format", {}).get("duration", 0))
    streams = [stream.get("codec_type") for stream in payload.get("streams", [])]
    if duration < 4.0:
        raise InputValidationError("Input duration must be at least 4 seconds")
    if "video" not in streams:
        raise InputValidationError("Input has no video stream")
    return ValidationResult(duration, "audio" in streams)


class ClipPipeline:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self.minimap_detector = MinimapDetector(config.MINIMAP_ICONS_DIR, config.MANIFEST_PATH)
        self.fight_detector = FightDetector()
        self.cropper = AdaptiveCropper()
        self.encoder = VideoEncoder()
        self.captioner = CaptionGenerator()

    async def run(self, source_path: Path, job_id: str | None = None) -> str:
        job_id = job_id or uuid.uuid4().hex
        flags: list[str] = []
        await models.create_job(self.db_path, job_id, source_path)
        try:
            validation = validate_input(source_path)
            if not validation.has_audio:
                flags.append("no_audio")

            await models.update_job(self.db_path, job_id, status="running", stage="stage1_decode", flags=flags)
            bundle = decode_video(source_path, job_id)

            await models.update_job(self.db_path, job_id, stage="stage2_minimap")
            detections = [self.minimap_detector.detect_icons(frame) for frame in bundle.minimap_frames]
            player_positions = [self.minimap_detector.find_white_box(frame) for frame in bundle.minimap_frames]
            participants = self.minimap_detector.aggregate_detections(
                detections,
                bundle.timestamps_mini,
                0,
                validation.duration,
                player_positions,
            )
            flags.extend(participants.flags)
            if self.minimap_detector.minimap_boundary_estimated:
                flags.append("minimap_boundary_estimated")

            await models.update_job(self.db_path, job_id, stage="stage3_fight", flags=flags)
            trim = self.fight_detector.detect(bundle.full_frames, bundle.timestamps_full, validation.duration, bundle.audio_path)
            flags.extend(trim.flags)

            await models.update_job(self.db_path, job_id, stage="stage4_crop", flags=flags)
            player_map_positions = _upsample_positions(player_positions, bundle.timestamps_mini, bundle.timestamps_full)
            enemies = _normalize_enemy_positions(participants.enemies)
            keyframes = self.cropper.compute_keyframes(
                bundle.full_frames,
                bundle.timestamps_full,
                trim.clip_start,
                trim.clip_end,
                player_map_positions,
                enemies,
                participants.fight_type,
            )
            clip_mask = (bundle.timestamps_full >= trim.clip_start) & (bundle.timestamps_full <= trim.clip_end)
            clip_timestamps = bundle.timestamps_full[clip_mask]
            crops = self.cropper.interpolate_to_frames(keyframes, clip_timestamps)

            await models.update_job(self.db_path, job_id, stage="stage5_encode", flags=flags)
            output_path = self.encoder.encode(job_id, source_path, trim.clip_start, trim.clip_end, crops, clip_timestamps)

            await models.update_job(self.db_path, job_id, stage="stage6_caption", output_path=output_path, flags=flags)
            dialog_text = " ".join(segment.text for segment in trim.dialog_segments)
            captions = self.captioner.generate(
                participants.player.champion_name,
                [enemy.champion_name for enemy in participants.enemies],
                participants.fight_type,
                trim.fight_duration,
                dialog_text,
            )
            flags.extend(captions.flags)
            await models.update_job(
                self.db_path,
                job_id,
                status="complete",
                stage="complete",
                flags=flags,
                captions=captions.captions,
                output_path=str(output_path),
                stage_failed=None,
                error_detail=None,
            )
            return job_id
        except (InputValidationError, FrameDecodeError, EncoderError, Exception) as exc:
            await models.update_job(
                self.db_path,
                job_id,
                status="failed",
                stage_failed=await _current_stage(self.db_path, job_id),
                error_detail=str(exc),
                flags=flags,
            )
            return job_id


async def _current_stage(db_path: Path, job_id: str) -> str | None:
    job = await models.get_job(db_path, job_id)
    return job.get("stage") if job else None


def _upsample_positions(
    positions: list[tuple[float, float] | None],
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
) -> list[tuple[float, float] | None]:
    if not positions or len(source_timestamps) == 0:
        return [None for _ in target_timestamps]
    result: list[tuple[float, float] | None] = []
    for timestamp in target_timestamps:
        idx = int(np.argmin(np.abs(source_timestamps - timestamp)))
        result.append(positions[min(idx, len(positions) - 1)])
    return result


def _normalize_enemy_positions(enemies: list[ChampionResult]) -> list[ChampionResult]:
    normalized: list[ChampionResult] = []
    for enemy in enemies:
        x, y = enemy.mean_pos
        if x > 1 or y > 1:
            x = x / 345.0
            y = y / 540.0
        normalized.append(ChampionResult(enemy.champion_name, enemy.confidence, enemy.team, (float(x), float(y)), enemy.is_player))
    return normalized
