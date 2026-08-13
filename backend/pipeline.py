from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from . import config, models
from .cropper import AdaptiveCropper, CropSettings
from .encoder import EncoderError, VideoEncoder, describe_encode_settings
from .fight_detector import (
    FightDetector,
    TrimSettings,
    apply_highlight_trim_settings,
    estimate_combat_screen_x_positions,
    estimate_visible_enemy_count,
)
from .frame_io import FrameDecodeError, decode_video
from .media_probe import MediaProbeError, MediaProfile, probe_media_profile
from .minimap_detector import ChampionResult, FightParticipants, MinimapDetector, _known_player_champion
from .models import update_job_progress
from .team_tracker import TeamTracker, clustered_indices, infer_team_from_border
from .vision_classifier import VisionFightResult, classify_fight_participants


@dataclass(frozen=True)
class ValidationResult:
    duration: float
    has_audio: bool
    media_profile: MediaProfile


@dataclass(frozen=True)
class ProcessingSettings:
    skip_minimap_detection: bool = config.SKIP_MINIMAP_DETECTION


class InputValidationError(ValueError):
    pass


def validate_input(path: Path) -> ValidationResult:
    if not path.exists() or path.suffix.lower() != ".mp4":
        raise InputValidationError("Input must be an existing .mp4 file")
    try:
        media_profile = probe_media_profile(path)
    except MediaProbeError as exc:
        raise InputValidationError(str(exc)) from exc
    if media_profile.duration < 4.0:
        raise InputValidationError("Input duration must be at least 4 seconds")
    if media_profile.video_codec is None:
        raise InputValidationError("Input has no video stream")
    return ValidationResult(media_profile.duration, media_profile.has_audio, media_profile)


class ClipPipeline:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self.minimap_detector = MinimapDetector(config.MINIMAP_ICONS_DIR, config.MANIFEST_PATH)
        self.fight_detector = FightDetector()
        self.cropper = AdaptiveCropper()
        self.encoder = VideoEncoder()

    async def run(
        self,
        source_path: Path,
        job_id: str | None = None,
        trim_settings: TrimSettings | None = None,
        highlight_checkpoint_path: Path | None = None,
        crop_settings: CropSettings | None = None,
        processing_settings: ProcessingSettings | None = None,
    ) -> str:
        job_id = job_id or uuid.uuid4().hex
        db_path = self.db_path
        flags: list[str] = []
        current_stage = "queued"
        crop_settings = crop_settings or CropSettings()
        processing_settings = processing_settings or ProcessingSettings()
        if await models.get_job(db_path, job_id) is None:
            await models.create_job(db_path, job_id, source_path)
        try:
            validation = validate_input(source_path)
            media_debug = {
                "media_profile": {
                    "input": validation.media_profile.to_debug_dict(),
                    "encode_settings": describe_encode_settings(validation.media_profile),
                },
                "highlight_checkpoint": str((highlight_checkpoint_path or config.VIDEOMAE_HIGHLIGHT_CHECKPOINT).resolve()),
                "processing_settings": _processing_settings_to_debug(processing_settings),
            }
            if not validation.has_audio:
                flags.append("no_audio")

            current_stage = "stage1_decode"
            await models.update_job(
                db_path,
                job_id,
                status="running",
                stage=current_stage,
                flags=flags,
                detection_debug=media_debug,
            )
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
            stride = max(1, config.MINIMAP_DETECTION_STRIDE)
            minimap_indices = np.arange(0, len(bundle.minimap_frames), stride)
            detection_frames = bundle.minimap_frames[minimap_indices]
            detection_timestamps = bundle.timestamps_mini[minimap_indices]
            detections = []
            team_tracker_summary: dict | None = None
            if processing_settings.skip_minimap_detection:
                detection_frames = np.empty((0,), dtype=np.uint8)
                detection_timestamps = np.empty((0,), dtype=np.float32)
                await update_job_progress(
                    db_path,
                    job_id,
                    "stage2_minimap",
                    100,
                    "Minimap champion detection skipped",
                )
            else:
                await update_job_progress(
                    db_path,
                    job_id,
                    "stage2_minimap",
                    10,
                    "Detecting champion icons on minimap...",
                )
                team_tracker = TeamTracker()
                total_detection_frames = max(len(detection_frames), 1)
                for index, frame in enumerate(detection_frames):
                    frame_detections = self.minimap_detector.detect_icons(frame)
                    detections.append(frame_detections)
                    detection_boxes = [_detection_box(detection) for detection in frame_detections]
                    clustered = clustered_indices(detection_boxes)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    for detection_index, detection in enumerate(frame_detections):
                        if detection_index in clustered:
                            continue
                        team_vote, border_confidence = infer_team_from_border(frame_bgr, detection_boxes[detection_index])
                        team_tracker.update(detection.champion_name, team_vote, border_confidence, detection.match_score)
                    if index and index % 15 == 0:
                        progress = 10 + int((index / total_detection_frames) * 70)
                        await update_job_progress(
                            db_path,
                            job_id,
                            "stage2_minimap",
                            min(progress, 80),
                            f"Scanning minimap frames {index}/{total_detection_frames}...",
                        )
                team_tracker_summary = team_tracker.summary()
                await models.update_job(db_path, job_id, detection_debug={"team_tracker": team_tracker_summary})
                detections = _apply_tracked_teams(detections, team_tracker)
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
                "Loading VideoMAE highlight editor...",
            )
            trim = self.fight_detector.predict_highlight_trim(
                bundle.full_frames,
                bundle.timestamps_full,
                validation.duration,
                highlight_checkpoint_path,
            )
            raw_trim = trim
            trim = apply_highlight_trim_settings(
                trim,
                bundle.full_frames,
                bundle.timestamps_full,
                validation.duration,
                trim_settings,
            )
            trim_debug = {
                "trim": {
                    "raw_model": _trim_to_debug(raw_trim),
                    "final": _trim_to_debug(trim),
                    "settings": _trim_settings_to_debug(trim_settings or TrimSettings()),
                }
            }
            await update_job_progress(
                db_path,
                job_id,
                "stage3_fight",
                50,
                f"Highlight editor selected final trim: {trim.clip_start:.1f}s to {trim.clip_end:.1f}s",
            )
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
            visible_enemy_count = estimate_visible_enemy_count(bundle.full_frames, bundle.timestamps_full, trim.fight_start, trim.fight_end)
            if processing_settings.skip_minimap_detection:
                participants = _participants_without_minimap(trusted_player_champion, player_champion_score, visible_enemy_count)
            else:
                participants = self.minimap_detector.aggregate_detections(
                    detections,
                    detection_timestamps,
                    max(0.0, trim.clip_start - config.MINIMAP_CONTEXT_BEFORE_FIGHT_SEC),
                    min(validation.duration, trim.clip_end + config.MINIMAP_CONTEXT_AFTER_FIGHT_SEC),
                    trusted_player_champion,
                )
            if visible_enemy_count is not None and not processing_settings.skip_minimap_detection:
                participants = _cap_participants_to_visible_enemy_count(participants, visible_enemy_count)
            vision_result = classify_fight_participants(bundle.full_frames, bundle.timestamps_full, trim.clip_start, trim.clip_end)
            if vision_result is not None:
                flags.append("local_yolo_champion_classifier")
                participants = _apply_vision_participants(participants, vision_result, trusted_player_champion, player_champion_score)
            detection_debug = _write_detection_debug(
                job_id,
                detection_frames,
                detections,
                detection_timestamps,
                trim.clip_start,
                trim.clip_end,
                participants,
                team_tracker_summary,
            )
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
                "Computing dynamic crop trajectory...",
            )
            player_screen_x_positions, threat_screen_x_positions = estimate_combat_screen_x_positions(bundle.full_frames)
            threat_signal_debug = _threat_signal_to_debug(threat_screen_x_positions)
            keyframes = self.cropper.compute_keyframes(
                bundle.full_frames,
                bundle.timestamps_full,
                trim.clip_start,
                trim.clip_end,
                [],
                [],
                participants.fight_type,
                player_screen_x_positions,
                threat_screen_x_positions,
                crop_settings,
            )
            clip_mask = (bundle.timestamps_full >= trim.clip_start) & (bundle.timestamps_full <= trim.clip_end)
            clip_timestamps = bundle.timestamps_full[clip_mask]
            crops = self.cropper.interpolate_to_frames(keyframes, clip_timestamps, crop_settings)
            crop_debug = _crop_path_to_debug(keyframes, crops, clip_timestamps, crop_settings, threat_signal_debug)
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
            output_path = self.encoder.encode(
                job_id,
                source_path,
                trim.clip_start,
                trim.clip_end,
                crops,
                clip_timestamps,
                validation.media_profile,
                crop_settings.transition,
            )
            await update_job_progress(
                db_path,
                job_id,
                "stage5_encode",
                100,
                "Video encoded successfully",
            )

            await models.update_job(
                db_path,
                job_id,
                status="complete",
                stage="complete",
                flags=flags,
                detection_debug={
                    **detection_debug,
                    **media_debug,
                    **trim_debug,
                    "crop_settings": _crop_settings_to_debug(crop_settings),
                    "crop_debug": crop_debug,
                },
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


def _threat_signal_to_debug(healthbar_values: list[float | None]) -> dict:
    healthbar_count = sum(1 for value in healthbar_values if value is not None)
    return {
        "healthbar_samples": healthbar_count,
    }


def _detection_box(detection) -> tuple[int, int, int, int]:
    x, y = detection.circle_center
    radius = int(detection.radius)
    return (int(x - radius), int(y - radius), int(x + radius), int(y + radius))


def _apply_tracked_teams(detections_per_frame: list[list], team_tracker: TeamTracker) -> list[list]:
    tracked: list[list] = []
    for frame_detections in detections_per_frame:
        tracked_frame = []
        for detection in frame_detections:
            team, _confidence = team_tracker.get_team(detection.champion_name)
            tracked_frame.append(replace(detection, team=team) if team is not None else detection)
        tracked.append(tracked_frame)
    return tracked


def _write_detection_debug(
    job_id: str,
    frames: np.ndarray,
    detections_per_frame: list[list],
    timestamps: np.ndarray,
    clip_start: float,
    clip_end: float,
    participants: FightParticipants,
    team_tracker_summary: dict | None = None,
) -> dict:
    debug_dir = config.OUTPUT_DIR / "debug" / job_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    frame_records: list[dict] = []
    if len(frames) == 0 or len(timestamps) == 0:
        return _detection_debug_payload(participants, frame_records, team_tracker_summary)

    indexes = np.flatnonzero((timestamps >= clip_start) & (timestamps <= clip_end))
    if len(indexes) == 0:
        indexes = np.array([int(np.argmin(np.abs(timestamps - clip_start)))])
    max_samples = 12
    if len(indexes) > max_samples:
        indexes = indexes[np.linspace(0, len(indexes) - 1, max_samples, dtype=int)]

    for ordinal, index in enumerate(indexes):
        idx = int(index)
        frame = frames[idx]
        detections = detections_per_frame[idx] if idx < len(detections_per_frame) else []
        overlay = _draw_detection_overlay(frame, detections)
        filename = f"minimap_{ordinal:02d}_{float(timestamps[idx]):06.2f}s.jpg".replace(".", "_", 1)
        path = debug_dir / filename
        cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        frame_records.append(
            {
                "timestamp": round(float(timestamps[idx]), 3),
                "image_url": f"/outputs/debug/{job_id}/{filename}",
                "detections": [_debug_detection(item) for item in detections],
            }
        )

    return _detection_debug_payload(participants, frame_records, team_tracker_summary)


def _trim_to_debug(trim) -> dict:
    return {
        "clip_start": trim.clip_start,
        "clip_end": trim.clip_end,
        "duration": round(max(0.0, trim.clip_end - trim.clip_start), 3),
        "fight_start": trim.fight_start,
        "fight_end": trim.fight_end,
        "fight_duration": trim.fight_duration,
        "pre_fight_lead": round(max(0.0, trim.fight_start - trim.clip_start), 3),
        "flags": list(trim.flags),
    }


def _trim_settings_to_debug(settings: TrimSettings) -> dict:
    return {
        "fight_start_preroll_sec": settings.fight_start_preroll_sec,
        "output_context_padding_sec": settings.output_context_padding_sec,
        "combat_event_end_padding_sec": settings.combat_event_end_padding_sec,
        "max_pre_fight_lead_sec": settings.max_pre_fight_lead_sec,
        "min_clip_duration_sec": settings.min_clip_duration_sec,
        "target_clip_duration_sec": settings.target_clip_duration_sec,
        "max_clip_duration_sec": settings.max_clip_duration_sec,
        "conservative_full_fight_trim": settings.conservative_full_fight_trim,
        "model_only": settings.model_only,
    }


def _crop_settings_to_debug(settings: CropSettings) -> dict:
    return {
        "mode": settings.mode,
        "transition": settings.transition,
    }


def _processing_settings_to_debug(settings: ProcessingSettings) -> dict:
    return {
        "skip_minimap_detection": settings.skip_minimap_detection,
    }


def _crop_path_to_debug(
    keyframes,
    crops: list[tuple[int, int, int, int]],
    timestamps: np.ndarray,
    settings: CropSettings,
    threat_signal_debug: dict | None = None,
) -> dict:
    frame_x = [int(crop[0]) for crop in crops]
    if not frame_x:
        return {
            "mode": settings.mode,
            "transition": settings.transition,
            "frame_count": 0,
            "keyframe_count": len(keyframes),
            "x_min": None,
            "x_max": None,
            "movement_px": 0,
            "position_changes": 0,
            "sample_keyframes": [],
            "threat_signal": threat_signal_debug or {},
        }

    sample_indexes = np.linspace(0, len(keyframes) - 1, min(8, len(keyframes)), dtype=int) if keyframes else []
    return {
        "mode": settings.mode,
        "transition": settings.transition,
        "frame_count": len(frame_x),
        "keyframe_count": len(keyframes),
        "x_min": min(frame_x),
        "x_max": max(frame_x),
        "x_start": frame_x[0],
        "x_end": frame_x[-1],
        "movement_px": max(frame_x) - min(frame_x),
        "position_changes": sum(1 for left, right in zip(frame_x, frame_x[1:]) if left != right),
        "unique_positions": len(set(frame_x)),
        "sample_keyframes": [
            {
                "time": round(float(keyframes[int(index)].timestamp), 3),
                "x": int(keyframes[int(index)].crop_x),
            }
            for index in sample_indexes
        ],
        "sample_frames": _sample_crop_frames(frame_x, timestamps),
        "threat_signal": threat_signal_debug or {},
        "note": "movement_px=0 means this crop mode resolved to a fixed camera path for this clip.",
    }


def _sample_crop_frames(frame_x: list[int], timestamps: np.ndarray) -> list[dict]:
    if not frame_x:
        return []
    sample_indexes = np.linspace(0, len(frame_x) - 1, min(8, len(frame_x)), dtype=int)
    return [
        {
            "time": round(float(timestamps[int(index)]), 3) if len(timestamps) > int(index) else None,
            "x": frame_x[int(index)],
        }
        for index in sample_indexes
    ]


def _detection_debug_payload(participants: FightParticipants, frames: list[dict], team_tracker_summary: dict | None = None) -> dict:
    skipped = "minimap_detection_skipped" in participants.flags
    notes = [
        "Minimap champion detection was skipped for this job.",
        "Participant summary uses main-frame HUD, health-bar, and optional local vision signals only.",
    ] if skipped else [
        "Each crop is the original minimap sample from the final clipped time range.",
        "Boxes and labels show YOLO minimap champion detections used for participant aggregation.",
    ]
    return {
        "summary": {
            "player": participants.player.champion_name,
            "allies": [ally.champion_name for ally in participants.allies],
            "enemies": [enemy.champion_name for enemy in participants.enemies],
            "fight_type": participants.fight_type,
        },
        "team_tracker": team_tracker_summary or {},
        "frames": frames,
        "notes": notes,
    }


def _draw_detection_overlay(frame: np.ndarray, detections: list) -> np.ndarray:
    overlay = frame.copy()
    for detection in detections:
        x, y = detection.circle_center
        radius = int(max(detection.radius, 8))
        color = (235, 64, 64) if detection.team == "enemy" else (70, 135, 245) if detection.team == "ally" else (245, 202, 71)
        cv2.rectangle(overlay, (max(0, x - radius), max(0, y - radius)), (min(overlay.shape[1] - 1, x + radius), min(overlay.shape[0] - 1, y + radius)), color, 2)
        label = f"{detection.champion_name} {detection.match_score:.2f}"
        cv2.putText(overlay, label, (max(0, x - radius), max(14, y - radius - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return overlay


def _debug_detection(detection) -> dict:
    x, y = detection.circle_center
    radius = int(detection.radius)
    return {
        "champion": detection.champion_name,
        "team": detection.team,
        "confidence": round(float(detection.match_score), 3),
        "uncertain": bool(detection.is_uncertain),
        "box": {
            "x1": int(x - radius),
            "y1": int(y - radius),
            "x2": int(x + radius),
            "y2": int(y + radius),
        },
    }


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


def _participants_without_minimap(
    trusted_player_champion: str | None,
    trusted_player_score: float,
    visible_enemy_count: int | None,
) -> FightParticipants:
    player = ChampionResult(
        _known_player_champion(trusted_player_champion, "unknown_champion_0"),
        max(0.0, trusted_player_score),
        "ally",
        (0.5, 0.5),
        True,
    )
    enemy_count = max(1, min(5, int(visible_enemy_count or 1)))
    enemies = [
        ChampionResult(f"unknown_enemy_{index}", 0.0, "enemy", (0.5, 0.5))
        for index in range(enemy_count)
    ]
    flags = ["minimap_detection_skipped"]
    if trusted_player_champion is None:
        flags.append("no_champions_identified")
    if visible_enemy_count is not None:
        flags.append("enemy_count_from_healthbars")
    return FightParticipants(player, [], enemies, f"1v{enemy_count}", flags)


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
