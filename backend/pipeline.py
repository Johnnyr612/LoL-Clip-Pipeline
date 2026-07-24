from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from . import config, models
from .cropper import AdaptiveCropper
from .encoder import EncoderError, VideoEncoder, describe_encode_settings
from .fight_detector import (
    FightDetector,
    TrimSettings,
    add_output_context,
    apply_dialog_extension,
    boundaries_from_scores,
    estimate_combat_screen_x_positions,
    estimate_visible_enemy_count,
    finish_on_kill_or_death,
)
from .frame_io import FrameDecodeError, decode_video
from .media_probe import MediaProbeError, MediaProfile, probe_media_profile
from .minimap_detector import ChampionResult, FightParticipants, MinimapDetector
from .models import update_job_progress
from .team_tracker import TeamTracker, clustered_indices, infer_team_from_border
from .vision_classifier import VisionFightResult, classify_fight_participants


@dataclass(frozen=True)
class ValidationResult:
    duration: float
    has_audio: bool
    media_profile: MediaProfile


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

    async def run(self, source_path: Path, job_id: str | None = None, trim_settings: TrimSettings | None = None) -> str:
        settings = trim_settings or TrimSettings()
        job_id = job_id or uuid.uuid4().hex
        db_path = self.db_path
        flags: list[str] = []
        current_stage = "queued"
        if await models.get_job(db_path, job_id) is None:
            await models.create_job(db_path, job_id, source_path)
        try:
            validation = validate_input(source_path)
            media_debug = {
                "media_profile": {
                    "input": validation.media_profile.to_debug_dict(),
                    "encode_settings": describe_encode_settings(validation.media_profile),
                }
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
            fight_start, fight_end, fight_flags = boundaries_from_scores(scores, validation.duration, settings)
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
                settings,
            )
            trim = add_output_context(trim, validation.duration, settings)
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
                flags.append("local_yolo_champion_classifier")
                participants = _apply_vision_participants(participants, vision_result, trusted_player_champion, player_champion_score)
            detection_debug = _write_detection_debug(
                job_id,
                detection_frames,
                detections,
                detection_timestamps,
                sampled_player_positions,
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
                "Computing adaptive crop trajectory...",
            )
            player_map_positions = _upsample_positions(player_positions, bundle.timestamps_mini, bundle.timestamps_full)
            player_screen_x_positions, threat_screen_x_positions = estimate_combat_screen_x_positions(bundle.full_frames)
            enemies = _normalize_enemy_positions(participants.enemies)
            keyframes = self.cropper.compute_keyframes(
                bundle.full_frames,
                bundle.timestamps_full,
                trim.clip_start,
                trim.clip_end,
                player_map_positions,
                enemies,
                participants.fight_type,
                player_screen_x_positions,
                threat_screen_x_positions,
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
            output_path = self.encoder.encode(
                job_id,
                source_path,
                trim.clip_start,
                trim.clip_end,
                crops,
                clip_timestamps,
                validation.media_profile,
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
                detection_debug={**detection_debug, **media_debug},
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
    player_positions: list[tuple[float, float] | None],
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
        player_pos = player_positions[idx] if idx < len(player_positions) else None
        overlay = _draw_detection_overlay(frame, detections, player_pos)
        filename = f"minimap_{ordinal:02d}_{float(timestamps[idx]):06.2f}s.jpg".replace(".", "_", 1)
        path = debug_dir / filename
        cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        frame_records.append(
            {
                "timestamp": round(float(timestamps[idx]), 3),
                "image_url": f"/outputs/debug/{job_id}/{filename}",
                "white_box": _debug_point(player_pos),
                "detections": [_debug_detection(item) for item in detections],
            }
        )

    return _detection_debug_payload(participants, frame_records, team_tracker_summary)


def _detection_debug_payload(participants: FightParticipants, frames: list[dict], team_tracker_summary: dict | None = None) -> dict:
    return {
        "summary": {
            "player": participants.player.champion_name,
            "allies": [ally.champion_name for ally in participants.allies],
            "enemies": [enemy.champion_name for enemy in participants.enemies],
            "fight_type": participants.fight_type,
        },
        "team_tracker": team_tracker_summary or {},
        "frames": frames,
        "notes": [
            "The white minimap camera box is treated as the recording/player anchor.",
            "Each crop is the original minimap sample from the final clipped time range.",
            "Boxes and labels show YOLO minimap champion detections used for participant aggregation.",
        ],
    }


def _draw_detection_overlay(frame: np.ndarray, detections: list, player_pos: tuple[float, float] | None) -> np.ndarray:
    overlay = frame.copy()
    for detection in detections:
        x, y = detection.circle_center
        radius = int(max(detection.radius, 8))
        color = (235, 64, 64) if detection.team == "enemy" else (70, 135, 245) if detection.team == "ally" else (245, 202, 71)
        cv2.rectangle(overlay, (max(0, x - radius), max(0, y - radius)), (min(overlay.shape[1] - 1, x + radius), min(overlay.shape[0] - 1, y + radius)), color, 2)
        label = f"{detection.champion_name} {detection.match_score:.2f}"
        cv2.putText(overlay, label, (max(0, x - radius), max(14, y - radius - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    if player_pos is not None:
        px, py = int(round(player_pos[0])), int(round(player_pos[1]))
        cv2.drawMarker(overlay, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(overlay, "white box anchor", (max(0, px - 44), max(14, py - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _debug_point(point: tuple[float, float] | None) -> dict | None:
    if point is None:
        return None
    return {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}


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
