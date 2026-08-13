export type JobRecord = {
  id: string;
  status: string;
  stage: string | null;
  progress: number;
  status_message: string;
  stage_failed: string | null;
  flags: string;
  error_detail?: string | null;
  source_path?: string | null;
  output_path?: string | null;
  detection_debug?: string;
  created_at: string;
  updated_at: string;
  history_only?: boolean;
};

export type OutputFile = {
  filename: string;
  path: string;
  url: string;
  size: number;
  modified_at: string;
};

export type HighlightCheckpoint = {
  filename: string;
  path: string;
  size: number;
  modified_at: string;
  active: boolean;
};

export type TrainingMetric = {
  status?: string;
  epoch?: number;
  train_loss?: number;
  val_loss?: number;
  accuracy?: number;
};

export type TikTokStatus = {
  configured: boolean;
  connected: boolean;
  open_id?: string | null;
  scope?: string;
  expires_at?: number | null;
  refresh_expires_at?: number | null;
};

export type DetectionDebug = {
  media_profile?: MediaProfileDebug;
  processing_settings?: {
    skip_minimap_detection?: boolean;
  };
  crop_settings?: {
    mode?: string;
    transition?: string;
  };
  crop_debug?: {
    mode?: string;
    transition?: string;
    frame_count?: number;
    keyframe_count?: number;
    x_min?: number | null;
    x_max?: number | null;
    x_start?: number | null;
    x_end?: number | null;
    movement_px?: number;
    position_changes?: number;
    unique_positions?: number;
    sample_keyframes?: Array<{ time: number; x: number }>;
    sample_frames?: Array<{ time: number | null; x: number }>;
    threat_signal?: {
      healthbar_samples?: number;
    };
    note?: string;
  };
  trim?: {
    settings?: Record<string, unknown>;
    final?: {
      clip_start?: number;
      clip_end?: number;
      duration?: number;
      fight_start?: number;
      fight_end?: number;
    };
  };
  summary?: {
    player?: string;
    allies?: string[];
    enemies?: string[];
    fight_type?: string;
  };
  notes?: string[];
  frames?: DetectionDebugFrame[];
};

export type MediaProfileDebug = {
  input?: {
    duration?: number | null;
    has_audio?: boolean;
    video_codec?: string | null;
    audio_codec?: string | null;
    width?: number | null;
    height?: number | null;
    fps?: number | null;
    fps_rate?: string | null;
    video_bitrate?: number | null;
    audio_bitrate?: number | null;
    total_bitrate?: number | null;
  };
  encode_settings?: {
    match_source_encoding?: boolean;
    encoder?: string | null;
    fps?: string | null;
    target_video_bitrate?: string | null;
    maxrate?: string | null;
    bufsize?: string | null;
    source_bitrate_multiplier?: number | null;
    crf?: number | null;
    preset?: string | null;
    rate_control?: string | null;
    audio_codec?: string | null;
    audio_bitrate?: string | null;
    pixel_format?: string | null;
    output_width?: number | null;
    output_height?: number | null;
  };
};

export type DetectionDebugFrame = {
  timestamp: number;
  image_url: string;
  detections: DetectionDebugResult[];
};

export type DetectionDebugResult = {
  champion: string;
  team: "ally" | "enemy" | "unknown";
  confidence: number;
  uncertain: boolean;
  box: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
};
