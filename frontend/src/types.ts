export type JobRecord = {
  id: string;
  status: string;
  stage: string | null;
  flags: string;
  error_detail?: string | null;
  output_path?: string | null;
  captions?: string;
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
