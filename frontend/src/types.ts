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
  captions?: string;
  detection_debug?: string;
  created_at: string;
  updated_at: string;
};

export type CaptionPayload = {
  caption: string;
  hashtags: string[];
  hook_line: string;
};

export type TrainingMetric = {
  status?: string;
  epoch?: number;
  train_loss?: number;
  val_loss?: number;
  accuracy?: number;
};

export type DetectionDebug = {
  summary?: {
    player?: string;
    allies?: string[];
    enemies?: string[];
    fight_type?: string;
  };
  notes?: string[];
  frames?: DetectionDebugFrame[];
};

export type DetectionDebugFrame = {
  timestamp: number;
  image_url: string;
  white_box?: { x: number; y: number } | null;
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
