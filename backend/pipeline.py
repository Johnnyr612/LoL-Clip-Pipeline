from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import config, models
from .caption_gen import CaptionGenerator
from .cropper import AdaptiveCropper
from .encoder import EncoderError, VideoEncoder
from .fight_detector import FightDetector, add_output_context, apply_dialog_extension, boundaries_from_scores, estimate_visible_enemy_count, finish_on_kill_or_death
from .frame_io import FrameDecodeError, decode_video
from .minimap_detector import ChampionResult, FightParticipants, MinimapDetector
from .models import update_job_progress
from .vision_classifier import VisionFightResult, classify_fight_participants


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
        db_path = self.db_path
        flags: list[str] = []
        current_stage = "queued"
        await models.create_job(db_path, job_id, source_path)
        try:
            validation = validate_input(source_path)
            if not validation.has_audio:
                flags.append("no_audio")

            current_stage = "stage1_decode"
            await models.update_job(db_path, job_id, status="running", stage=current_stage, flags=flags)
            await update_job_progress(
                db_path,
                job_id,
                "stage1_decode",
                10,
                "Validating input file...",
            )
            await update_job_progress(
                db_path,
                job_id,
                "stage1_decode",
                50,
                "Extracting frames from 4K source...",
            )
            bundle = decode_video(source_path, job_id)
            await update_job_progress(
                db_path,
                job_id,
                "stage1_decode",
                100,
                "Frames extracted successfully",
            )

            current_stage = "stage2_minimap"
            await models.update_job(db_path, job_id, stage=current_stage)
            await update_job_progress(
                db_path,
                job_id,
                "stage2_minimap",
                10,
                "Detecting champion icons on minimap...",
            )
            stride = max(1, config.MINIMAP_DETECTION_STRIDE)
            minimap_indices = np.arange(0, len(bundle.minimap_frames), stride)
            detection_frames = bundle.minimap_frames[minimap_indices]
            detection_timestamps = bundle.timestamps_mini[minimap_indices]
            detections = []
            total_detection_frames = max(len(detection_frames), 1)
            for index, frame in enumerate(detection_frames):
                detections.append(self.minimap_detector.detect_icons(frame))
                if index and index % 15 == 0:
                    progress = 10 + int((index / total_detection_frames) * 70)
                    await update_job_progress(
                        db_path,
                        job_id,
                        "stage2_minimap",
                        min(progress, 80),
                        f"Scanning minimap frames {index}/{total_detection_frames}...",
                    )
            player_positions = [self.minimap_detector.find_white_box(frame) for frame in bundle.minimap_frames]
            sampled_player_positions = [
                _position_to_pixels(player_positions[int(i)], bundle.minimap_frames[int(i)].shape)
                for i in minimap_indices
            ]
            if self.minimap_detector.minimap_boundary_estimated:
                flags.append("minimap_boundary_estimated")
            await update_job_progress(
                db_path,
                job_id,
                "stage2_minimap",
                100,
                "Minimap scan complete",
            )

            current_stage = "stage3_fight"
            await models.update_job(db_path, job_id, stage=current_stage, flags=flags)
            await update_job_progress(
                db_path,
                job_id,
                "stage3_fight",
                10,
                "Loading VideoMAE fight detector...",
            )
            scores = self.fight_detector.score_windows(bundle.full_frames, bundle.timestamps_full)
            fight_start, fight_end, fight_flags = boundaries_from_scores(scores, validation.duration)
            await update_job_progress(
                db_path,
                job_id,
                "stage3_fight",
                50,
                f"Fight detected: {fight_start:.1f}s to {fight_end:.1f}s",
            )
            dialog = self.fight_detector.transcribe(bundle.audio_path)
            trim_result = apply_dialog_extension(fight_start, fight_end, validation.duration, dialog)
            trim = finish_on_kill_or_death(
                replace(trim_result, flags=fight_flags + trim_result.flags),
                bundle.full_frames,
                bundle.timestamps_full,
                validation.duration,
            )
            trim = add_output_context(trim, validation.duration)
            player_champion, player_champion_score = _detect_player_champion(
                self.minimap_detector,
                bundle.full_frames,
                bundle.timestamps_full,
                trim.clip_start,
                trim.clip_end,
            )
            trusted_player_champion = player_champion if player_champion_score >= config.HUD_PLAYER_MATCH_CONFIRM else None
            if trusted_player_champion is None:
                flags.append("player_hud_champion_low_confidence")
            participants = self.minimap_detector.aggregate_detections(
                detections,
                detection_timestamps,
                max(0.0, trim.clip_start - config.MINIMAP_CONTEXT_BEFORE_FIGHT_SEC),
                min(validation.duration, trim.clip_end + config.MINIMAP_CONTEXT_AFTER_FIGHT_SEC),
                sampled_player_positions,
                trusted_player_champion,
            )
            visible_enemy_count = estimate_visible_enemy_count(bundle.full_frames, bundle.timestamps_full, trim.fight_start, trim.fight_end)
            if visible_enemy_count is not None:
                participants = _cap_participants_to_visible_enemy_count(participants, visible_enemy_count)
            vision_result = classify_fight_participants(bundle.full_frames, bundle.timestamps_full, trim.clip_start, trim.clip_end)
            if vision_result is not None:
                flags.append("vision_champion_classifier")
                participants = _apply_vision_participants(participants, vision_result, trusted_player_champion, player_champion_score)
            flags.extend(participants.flags)
            await update_job_progress(
                db_path,
                job_id,
                "stage3_fight",
                100,
                f"Fight confirmed: {participants.player.champion_name} {participants.fight_type}",
            )
            flags.extend(trim.flags)

            current_stage = "stage4_crop"
            await models.update_job(db_path, job_id, stage=current_stage, flags=flags)
            await update_job_progress(
                db_path,
                job_id,
                "stage4_crop",
                10,
                "Computing adaptive crop trajectory...",
            )
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
            await update_job_progress(
                db_path,
                job_id,
                "stage4_crop",
                100,
                "Crop trajectory ready",
            )

            current_stage = "stage5_encode"
            await models.update_job(db_path, job_id, stage=current_stage, flags=flags)
            await update_job_progress(
                db_path,
                job_id,
                "stage5_encode",
                10,
                "Trimming source clip...",
            )
            await update_job_progress(
                db_path,
                job_id,
                "stage5_encode",
                50,
                "Encoding 1080x1440 vertical video...",
            )
            output_path = self.encoder.encode(job_id, source_path, trim.clip_start, trim.clip_end, crops, clip_timestamps)
            await update_job_progress(
                db_path,
                job_id,
                "stage5_encode",
                100,
                "Video encoded successfully",
            )

            current_stage = "stage6_caption"
            await models.update_job(db_path, job_id, stage=current_stage, output_path=output_path, flags=flags)
            await update_job_progress(
                db_path,
                job_id,
                "stage6_caption",
                10,
                "Generating TikTok caption...",
            )
            dialog_text = " ".join(segment.text for segment in trim.dialog_segments)
            await update_job_progress(
                db_path,
                job_id,
                "stage6_caption",
                50,
                "Generating Instagram caption...",
            )
            captions = self.captioner.generate(
                participants.player.champion_name,
                [enemy.champion_name for enemy in participants.enemies],
                participants.fight_type,
                trim.fight_duration,
                dialog_text,
                _minimap_champion_context(participants),
            )
            flags.extend(captions.flags)
            await update_job_progress(
                db_path,
                job_id,
                "stage6_caption",
                100,
                "Captions generated successfully",
            )
            await models.update_job(
                db_path,
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
            await update_job_progress(
                db_path,
                job_id,
                current_stage,
                0,
                f"Error: {str(exc)[:200]}",
            )
            await models.update_job(
                db_path,
                job_id,
                status="failed",
                stage_failed=current_stage,
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


def _detect_player_champion(
    detector: MinimapDetector,
    frames: np.ndarray,
    timestamps: np.ndarray,
    clip_start: float,
    clip_end: float,
) -> tuple[str, float]:
    if len(frames) == 0 or len(timestamps) == 0:
        return "unknown", -1.0
    mask = (timestamps >= clip_start) & (timestamps <= clip_end)
    indexes = np.flatnonzero(mask)
    if len(indexes) == 0:
        indexes = np.array([int(np.argmin(np.abs(timestamps - clip_start)))])
    if len(indexes) > 8:
        indexes = indexes[np.linspace(0, len(indexes) - 1, 8, dtype=int)]

    scores: dict[str, list[float]] = {}
    for index in indexes:
        name, score = detector.detect_player_hud_champion(frames[int(index)])
        if name.startswith("unknown"):
            continue
        scores.setdefault(name, []).append(score)
    if not scores:
        return "unknown", -1.0
    name, values = max(scores.items(), key=lambda item: (len(item[1]), float(np.mean(item[1]))))
    return name, float(np.mean(values))


def _cap_participants_to_visible_enemy_count(participants: FightParticipants, enemy_count: int) -> FightParticipants:
    if enemy_count <= 0 or len(participants.enemies) <= enemy_count:
        return participants
    enemies = sorted(participants.enemies, key=lambda enemy: enemy.confidence, reverse=True)[:enemy_count]
    fight_type = f"1v{max(1, min(5, len(enemies)))}"
    return FightParticipants(participants.player, [], enemies, fight_type, [*participants.flags, "enemy_count_capped_by_healthbars"])


def _apply_vision_participants(
    participants: FightParticipants,
    vision_result: VisionFightResult,
    trusted_player_champion: str | None = None,
    trusted_player_score: float = -1.0,
) -> FightParticipants:
    player_name = vision_result.player_champion
    flags = [*participants.flags, "champions_overridden_by_vision"]
    if trusted_player_champion and not trusted_player_champion.startswith("unknown"):
        if vision_result.player_champion == trusted_player_champion:
            player_name = trusted_player_champion
        elif trusted_player_score >= config.HUD_PLAYER_OVERRIDE_VISION_CONFIRM:
            player_name = trusted_player_champion
            flags.append("vision_player_override_ignored")
        else:
            flags.append("hud_player_disagreed_with_vision")

    player = ChampionResult(
        player_name,
        max(vision_result.confidence, participants.player.confidence),
        "ally",
        participants.player.mean_pos,
        True,
    )
    local_enemies = {enemy.champion_name: enemy for enemy in participants.enemies}
    vision_enemies = [name for name in vision_result.enemy_champions if name != player.champion_name]
    enemies = [
        ChampionResult(
            name,
            max(vision_result.confidence, local_enemies.get(name, ChampionResult(name, 0.0, "enemy", (0.5, 0.5))).confidence),
            "enemy",
            local_enemies.get(name, ChampionResult(name, 0.0, "enemy", (0.5, 0.5))).mean_pos,
        )
        for name in vision_enemies
    ]
    fight_type = vision_result.fight_type or f"1v{max(1, len(enemies))}"
    return FightParticipants(player, [], enemies, fight_type, flags)


def _position_to_pixels(position: tuple[float, float] | None, frame_shape: tuple[int, ...]) -> tuple[float, float] | None:
    if position is None:
        return None
    height, width = frame_shape[:2]
    return (float(position[0] * width), float(position[1] * height))


def _minimap_champion_context(participants: FightParticipants) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for champion in [participants.player, *participants.allies, *participants.enemies]:
        name = champion.champion_name
        if not name or name.startswith("unknown") or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names
