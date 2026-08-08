from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Real environment variables win, so a one-off PowerShell override can
        # still take precedence over values saved in the local .env file.
        os.environ.setdefault(key, value)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


_load_local_env(PROJECT_ROOT / ".env")


MINIMAP_ICONS_DIR = PROJECT_ROOT / "data" / "minimap_icons"
MANIFEST_PATH = MINIMAP_ICONS_DIR / "champions_manifest.json"

MINIMAP_CROP_X_PCT = 0.82
MINIMAP_CROP_Y_PCT = 0.75

HOUGH_MIN_RADIUS = 8
HOUGH_MAX_RADIUS = 20
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 18
HOUGH_MIN_DIST = 20

TEMPLATE_MATCH_CONFIRM = 0.55
TEMPLATE_MATCH_UNCERTAIN = 0.35
MINIMAP_DETECTION_STRIDE = 4
MINIMAP_TOP_TEMPLATE_CANDIDATES = 8
MINIMAP_MAX_CIRCLES_PER_FRAME = 12
MINIMAP_CONTEXT_BEFORE_FIGHT_SEC = 2.0
MINIMAP_CONTEXT_AFTER_FIGHT_SEC = 1.5
MINIMAP_FIGHT_RADIUS_PX = 125.0
MINIMAP_OVERLAP_DISTANCE_MULTIPLIER = 1.55
MINIMAP_YOLO_ENABLED = os.environ.get("LOL_CLIP_MINIMAP_YOLO_ENABLED", "1").strip() != "0"
_MINIMAP_YOLO_WEIGHTS = os.environ.get("LOL_CLIP_MINIMAP_YOLO_WEIGHTS", "").strip()
MINIMAP_YOLO_WEIGHTS = (
    Path(_MINIMAP_YOLO_WEIGHTS).expanduser()
    if _MINIMAP_YOLO_WEIGHTS
    else PROJECT_ROOT / "checkpoints" / "minimap_yolov8s_best.pt"
)
MINIMAP_YOLO_CONFIDENCE = float(os.environ.get("LOL_CLIP_MINIMAP_YOLO_CONFIDENCE", "0.35"))
MINIMAP_YOLO_CONFIRM = float(os.environ.get("LOL_CLIP_MINIMAP_YOLO_CONFIRM", "0.45"))
MINIMAP_YOLO_DEVICE = os.environ.get("LOL_CLIP_MINIMAP_YOLO_DEVICE", "").strip() or None
MINIMAP_YOLO_MAX_DETECTIONS = int(os.environ.get("LOL_CLIP_MINIMAP_YOLO_MAX_DETECTIONS", "12"))
HUD_PLAYER_MATCH_CONFIRM = 0.65
HUD_PLAYER_OVERRIDE_VISION_CONFIRM = 0.92
ICON_PHASH_CONFIRM_MAX_DISTANCE = 22
ICON_PHASH_SUPPORT_MAX_DISTANCE = 24
ICON_PHASH_MIN_GRAY_SCORE = 0.25
CONSERVATIVE_FULL_FIGHT_TRIM = os.environ.get("LOL_CLIP_CONSERVATIVE_FULL_FIGHT_TRIM", "0").strip() != "0"
COMBAT_EVENT_SEARCH_AFTER_FIGHT_SEC = 40.0
COMBAT_EVENT_END_PADDING_SEC = float(os.environ.get("LOL_CLIP_END_PADDING_SEC", "3.0"))
COMBAT_EVENT_MIN_VISIBLE_ENEMY_FRAMES = 3
# Enemy bars must be absent this many consecutive samples (~0.5s each) before
# a kill is declared. Too low and an enemy stepping into fog mid-fight ends
# the clip early.
COMBAT_EVENT_MISSING_FRAMES = int(os.environ.get("LOL_CLIP_EVENT_MISSING_FRAMES", "4"))
# Minimum clip length. Kills/deaths before this point are ignored as end
# triggers, so keep it short enough that real fight endings can end the clip.
COMBAT_EVENT_MIN_CLIP_DURATION_SEC = float(os.environ.get("LOL_CLIP_MIN_CLIP_SEC", "20.0"))
# Fallback length when no kill/death is confirmed (and floor when
# CONSERVATIVE_FULL_FIGHT_TRIM is enabled).
COMBAT_EVENT_TARGET_CLIP_DURATION_SEC = float(os.environ.get("LOL_CLIP_TARGET_CLIP_SEC", "35.0"))
OUTPUT_CONTEXT_PADDING_SEC = float(os.environ.get("LOL_CLIP_OUTPUT_PADDING_SEC", "1.5"))
COMBAT_HEALTHBAR_MIN_WIDTH = int(os.environ.get("LOL_CLIP_HEALTHBAR_MIN_WIDTH", "25"))
COMBAT_HEALTHBAR_MAX_WIDTH = 170
# Champion health strips are thicker than minion strips at 1920x1080
# (~9-11px vs ~3-5px). Width is only a low noise floor because visible fill
# shrinks with remaining health and adjacent minion bars can merge.
COMBAT_CHAMPION_HEALTHBAR_MIN_WIDTH = int(os.environ.get("LOL_CLIP_CHAMPION_BAR_MIN_WIDTH", "25"))
COMBAT_CHAMPION_HEALTHBAR_MIN_HEIGHT = int(os.environ.get("LOL_CLIP_CHAMPION_BAR_MIN_HEIGHT", "8"))
# Crop steering trusts thick red bars. Width and area are kept as modest noise
# rejection only, so low-health champions can still steer the crop.
COMBAT_CAMERA_THREAT_HEALTHBAR_MIN_WIDTH = int(os.environ.get("LOL_CLIP_CAMERA_THREAT_BAR_MIN_WIDTH", "25"))
COMBAT_CAMERA_THREAT_HEALTHBAR_MIN_HEIGHT = int(os.environ.get("LOL_CLIP_CAMERA_THREAT_BAR_MIN_HEIGHT", "8"))
COMBAT_CAMERA_THREAT_HEALTHBAR_MIN_AREA = int(os.environ.get("LOL_CLIP_CAMERA_THREAT_BAR_MIN_AREA", "300"))
COMBAT_CAMERA_THREAT_SUPPORT_RADIUS_FRAMES = int(os.environ.get("LOL_CLIP_CAMERA_THREAT_SUPPORT_RADIUS_FRAMES", "3"))
COMBAT_CAMERA_THREAT_MIN_SUPPORT_SAMPLES = int(os.environ.get("LOL_CLIP_CAMERA_THREAT_MIN_SUPPORT_SAMPLES", "3"))
COMBAT_CAMERA_THREAT_SUPPORT_TOLERANCE_PX = int(os.environ.get("LOL_CLIP_CAMERA_THREAT_SUPPORT_TOLERANCE_PX", "170"))
COMBAT_HEALTHBAR_IGNORE_LEFT_X_PCT = 0.16
COMBAT_HEALTHBAR_IGNORE_LEFT_Y_MAX_PCT = 0.72

FIGHT_CONFIDENCE_THRESHOLD = 0.50
# Lower threshold used only when walking LEFT from the score peak: the 16s
# scoring windows ramp up gradually as the fight fills them, so the fight's
# opening seconds score below the main threshold.
FIGHT_ONSET_THRESHOLD = float(os.environ.get("LOL_CLIP_FIGHT_ONSET_THRESHOLD", "0.35"))
# Allow this many consecutive sub-threshold seconds during the boundary walk
# so a brief mid-fight lull doesn't cut the detected fight short.
FIGHT_BOUNDARY_GAP_TOLERANCE_SEC = int(os.environ.get("LOL_CLIP_FIGHT_GAP_TOLERANCE", "3"))
# Pull the detected fight start back by this many seconds to include the
# approach/poke phase that windowed scoring inherently misses.
FIGHT_START_PREROLL_SEC = float(os.environ.get("LOL_CLIP_FIGHT_START_PREROLL", "1.5"))
# Hard cap on non-fight lead-in: the clip never starts more than this many
# seconds before the detected fight start, no matter what dialog extension or
# padding would otherwise add.
MAX_PRE_FIGHT_LEAD_SEC = float(os.environ.get("LOL_CLIP_MAX_PRE_FIGHT_LEAD_SEC", "2.5"))
FIGHT_MIN_DURATION = float(os.environ.get("LOL_CLIP_FIGHT_MIN_DURATION", "4.0"))
FIGHT_MAX_DURATION = float(os.environ.get("LOL_CLIP_FIGHT_MAX_DURATION", "35.0"))
FIGHT_MERGE_GAP_SEC = 1.5
DIALOG_EXTENSION_WINDOW = 5.0
DIALOG_PADDING = 0.5
MAX_CLIP_DURATION = 60.0

CROP_W = 810
CROP_H = 1080
CROP_Y = 0
PLAYER_SAFE_LEFT_PX = 220
PLAYER_SAFE_RIGHT_PX = 590
PLAYER_CENTER_DEADZONE_PX = 45
PLAYER_THIRDS_DEADZONE_PX = 45
PLAYER_THIRDS_LOOK_ROOM_PX = _int_env("LOL_CLIP_THIRDS_LOOK_ROOM_PX", 20)
PLAYER_COMPOSITION = os.environ.get("LOL_CLIP_PLAYER_COMPOSITION", "thirds").strip().lower()
THREAT_FRAME_MARGIN_PX = 70
MINIMAP_UI_AVOID_MARGIN_PX = 30
# Crop mode:
#   "dynamic"  - default. For locked camera: starts centered, waits for a
#                persistent visible enemy side, then reframes with thirds
#                composition while limiting view changes.
#   "static"   - one fixed crop for the whole clip, zero movement.
CROP_MODE = os.environ.get("LOL_CLIP_CROP_MODE", "dynamic").strip().lower()
CROP_MODE = CROP_MODE if CROP_MODE in {"dynamic", "static"} else "dynamic"
# Fixed crop x for static mode and the base position for dynamic mode. Default
# centers the 810px crop in the 1920px frame, where a locked camera holds the
# champion.
STATIC_CROP_X = int(os.environ.get("LOL_CLIP_STATIC_CROP_X", str((1920 - 810) // 2)))
# How the crop moves between positions:
#   "cut" - default. Snaps instantly between held positions, like an editor's
#           camera cut. No sliding.
#   "pan" - eases between positions at a limited speed.
CROP_TRANSITION = os.environ.get("LOL_CLIP_CROP_TRANSITION", "cut").strip().lower()
# Dynamic mode: how far from the champion the threat must sit (px) before that
# flank counts as a left/right fight side.
DYNAMIC_THREAT_SIDE_TRIGGER_PX = int(
    os.environ.get("LOL_CLIP_DYNAMIC_THREAT_SIDE_TRIGGER_PX", os.environ.get("LOL_CLIP_HYBRID_SIDE_TRIGGER_PX", "60"))
)
# Dynamic mode: how long the same enemy side must persist before the camera
# reframes. Missing threats recenter immediately after filtering.
DYNAMIC_THREAT_HOLD_SEC = float(
    os.environ.get("LOL_CLIP_DYNAMIC_THREAT_HOLD_SEC", os.environ.get("LOL_CLIP_HYBRID_HOLD_SEC", "1.0"))
)
# Dynamic mode: hard budget on non-center enemy reframes per clip.
DYNAMIC_MAX_VIEW_CHANGES = int(
    os.environ.get("LOL_CLIP_DYNAMIC_MAX_VIEW_CHANGES", os.environ.get("LOL_CLIP_HYBRID_MAX_VIEW_CHANGES", "3"))
)

MAX_CROP_KEYFRAMES = 61
KEYFRAME_INTERVAL_SEC = 1.0
MAX_PAN_SPEED_PX_PER_SEC = 240
PAN_DEADBAND_PX = 24
CROP_START_SEED_KEYFRAMES = 3
PLAYER_SX_MEDIAN_WINDOW_SEC = 0.75
CROP_EXPR_MAX_POINTS = 48
LOW_FLOW_THRESHOLD = 500
BLEND_1V1 = (0.92, 0.08, 0.0)
BLEND_1VN = (0.90, 0.10, 0.0)
BLEND_NVN = (0.88, 0.12, 0.0)

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1440
OUTPUT_FPS = 60
FFMPEG_MATCH_SOURCE_ENCODING = os.environ.get("LOL_CLIP_MATCH_SOURCE_ENCODING", "1").strip() != "0"
FFMPEG_VIDEO_ENCODER = os.environ.get("LOL_CLIP_VIDEO_ENCODER", "libx264").strip() or "libx264"
FFMPEG_CRF = int(os.environ.get("LOL_CLIP_FFMPEG_CRF", "18"))
FFMPEG_PRESET = os.environ.get("LOL_CLIP_FFMPEG_PRESET", "slow").strip() or "slow"
FFMPEG_NVENC_PRESET = os.environ.get("LOL_CLIP_NVENC_PRESET", "p5").strip() or "p5"
FFMPEG_NVENC_RC = os.environ.get("LOL_CLIP_NVENC_RC", "vbr").strip() or "vbr"
FFMPEG_NVENC_CQ = os.environ.get("LOL_CLIP_NVENC_CQ", "").strip()
FFMPEG_VIDEO_BITRATE = os.environ.get("LOL_CLIP_VIDEO_BITRATE", "").strip()
FFMPEG_VIDEO_MAXRATE = os.environ.get("LOL_CLIP_VIDEO_MAXRATE", "").strip()
FFMPEG_VIDEO_BUFSIZE = os.environ.get("LOL_CLIP_VIDEO_BUFSIZE", "").strip()
FFMPEG_SOURCE_BITRATE_MULTIPLIER = _float_env("LOL_CLIP_SOURCE_BITRATE_MULTIPLIER", 1.15)
FFMPEG_AUDIO_BITRATE = os.environ.get("LOL_CLIP_AUDIO_BITRATE", "320k").strip() or "320k"
CROP_QUANTIZE_THRESHOLD = 5

VIDEOMAE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "videomae_lol_best.pt"
VIDEOMAE_HIGHLIGHT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "videomae_lol_highlight_editor.pt"
HIGHLIGHT_CONTEXT_SECONDS = int(os.environ.get("LOL_CLIP_HIGHLIGHT_CONTEXT_SECONDS", "60"))
HIGHLIGHT_INPUT_FRAMES = int(os.environ.get("LOL_CLIP_HIGHLIGHT_INPUT_FRAMES", "16"))
HIGHLIGHT_INCLUDE_THRESHOLD = float(os.environ.get("LOL_CLIP_HIGHLIGHT_INCLUDE_THRESHOLD", "0.50"))
HIGHLIGHT_MIN_CLIP_DURATION_SEC = float(os.environ.get("LOL_CLIP_HIGHLIGHT_MIN_CLIP_SEC", "8.0"))
HIGHLIGHT_MAX_CLIP_DURATION_SEC = float(os.environ.get("LOL_CLIP_HIGHLIGHT_MAX_CLIP_SEC", "60.0"))
PLAYER_NAME = "Aaplay44"
VISION_CLASSIFIER_TIMEOUT_SEC = 45
VISION_CLASSIFIER_MIN_CONFIDENCE = 0.60
_YOLO_WEIGHTS = os.environ.get("LOL_CLIP_YOLO_WEIGHTS", "").strip()
YOLO_DETECTOR_WEIGHTS = Path(_YOLO_WEIGHTS).expanduser() if _YOLO_WEIGHTS else None
YOLO_DETECTOR_CONFIDENCE = float(os.environ.get("LOL_CLIP_YOLO_CONFIDENCE", "0.35"))
YOLO_DETECTOR_DEVICE = os.environ.get("LOL_CLIP_YOLO_DEVICE", "").strip() or None
YOLO_DETECTOR_MAX_FRAMES = int(os.environ.get("LOL_CLIP_YOLO_MAX_FRAMES", "5"))

_APPDATA_BASE = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
APPDATA_DIR = _APPDATA_BASE / "LoLClipApp"
TEMP_DIR = APPDATA_DIR / "temp"
LOG_DIR = APPDATA_DIR / "logs"
OUTPUT_DIR = Path(os.environ.get("LOL_CLIP_OUTPUT_DIR", r"D:\LoLClipOutputVids")).expanduser()
DB_PATH = APPDATA_DIR / "lol_clip_app.sqlite3"

TIKTOK_CLIENT_KEY = os.environ.get("TIKTOK_CLIENT_KEY", "").strip()
TIKTOK_CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "").strip()
TIKTOK_REDIRECT_URI = os.environ.get("TIKTOK_REDIRECT_URI", "http://127.0.0.1:8000/tiktok/callback").strip()
TIKTOK_AUTH_SUCCESS_URL = os.environ.get("TIKTOK_AUTH_SUCCESS_URL", "http://127.0.0.1:5173").strip()
TIKTOK_DEFAULT_SCOPES = "user.info.basic,video.upload"
TIKTOK_DIRECT_SCOPES = "user.info.basic,video.publish"
