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
