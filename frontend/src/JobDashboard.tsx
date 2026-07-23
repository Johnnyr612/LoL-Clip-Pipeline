import React from "react";
import type { JobRecord, OutputFile } from "./types";

const stages = [
  "queued",
  "stage1_decode",
  "stage2_minimap",
  "stage3_fight",
  "stage4_crop",
  "stage5_encode",
  "complete"
] as const;

const stageNames: Record<string, string> = {
  queued: "Waiting to Start",
  stage1_decode: "Frame Extraction",
  stage2_minimap: "Champion Detection",
  stage3_fight: "Fight Detection",
  stage4_crop: "Crop Trajectory",
  stage5_encode: "Video Encoding",
  complete: "Complete"
};

type Props = {
  selectedJob: JobRecord | null;
  onSelectJob: (job: JobRecord | null) => void;
};

type TrimSettingsState = {
  fight_start_preroll_sec: number;
  output_context_padding_sec: number;
  combat_event_end_padding_sec: number;
  max_pre_fight_lead_sec: number;
  min_clip_duration_sec: number;
};

const defaultTrimSettings: TrimSettingsState = {
  fight_start_preroll_sec: 1.5,
  output_context_padding_sec: 1.5,
  combat_event_end_padding_sec: 3.0,
  max_pre_fight_lead_sec: 2.5,
  min_clip_duration_sec: 20.0
};

const trimPresets: Record<string, TrimSettingsState> = {
  Tight: {
    fight_start_preroll_sec: 0.8,
    output_context_padding_sec: 0.5,
    combat_event_end_padding_sec: 2.0,
    max_pre_fight_lead_sec: 1.2,
    min_clip_duration_sec: 14.0
  },
  Balanced: defaultTrimSettings,
  Cinematic: {
    fight_start_preroll_sec: 2.5,
    output_context_padding_sec: 2.0,
    combat_event_end_padding_sec: 4.0,
    max_pre_fight_lead_sec: 4.0,
    min_clip_duration_sec: 24.0
  }
};

function formatElapsed(totalSeconds: number) {
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  if (mins === 0) return `${secs}s`;
  return `${mins}m ${secs.toString().padStart(2, "0")}s`;
}

function durationFromJob(job: JobRecord | null) {
  if (!job?.created_at || !job.updated_at) return 0;
  const created = Date.parse(job.created_at);
  const updated = Date.parse(job.updated_at);
  if (Number.isNaN(created) || Number.isNaN(updated) || updated < created) return 0;
  return Math.floor((updated - created) / 1000);
}

function normalizeSourcePath(value: string) {
  let next = value.trim();
  while (next.length >= 2) {
    const first = next[0];
    const last = next[next.length - 1];
    if (!((first === `"` && last === `"`) || (first === `'` && last === `'`))) break;
    next = next.slice(1, -1).trim();
  }
  return next;
}

function displaySource(value?: string | null) {
  if (!value) return "Unknown source";
  return value.split(/[\\/]/).filter(Boolean).pop() ?? value;
}

function activeJobLabel(job: JobRecord | null) {
  if (!job) return "idle";
  if (job.history_only) return "previous output";
  if (job.status === "queued") return "waiting";
  if (job.status === "running") return "running";
  if (job.status === "complete") return "complete";
  if (job.status === "failed") return "failed";
  return job.status;
}

function formatFileSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatModified(value: string) {
  const date = Date.parse(value);
  if (Number.isNaN(date)) return "";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(date);
}

function outputFileToJob(output: OutputFile): JobRecord {
  return {
    id: `output:${output.filename}`,
    status: "complete",
    stage: "complete",
    progress: 100,
    status_message: "Previous output",
    stage_failed: null,
    flags: "[]",
    source_path: output.path,
    output_path: output.path,
    detection_debug: "{}",
    created_at: output.modified_at,
    updated_at: output.modified_at,
    history_only: true
  };
}

async function fetchJsonOr<T>(url: string, fallback: T): Promise<T> {
  try {
    const response = await fetch(url);
    if (!response.ok) return fallback;
    return await response.json();
  } catch {
    return fallback;
  }
}

export function JobDashboard({ selectedJob, onSelectJob }: Props) {
  const [sourcePath, setSourcePath] = React.useState("");
  const [jobs, setJobs] = React.useState<JobRecord[]>([]);
  const [outputs, setOutputs] = React.useState<OutputFile[]>([]);
  const [outputDir, setOutputDir] = React.useState("");
  const [outputsOpen, setOutputsOpen] = React.useState(true);
  const [advancedOpen, setAdvancedOpen] = React.useState(false);
  const [trimSettings, setTrimSettings] = React.useState<TrimSettingsState>(defaultTrimSettings);
  const [busy, setBusy] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState("");
  const [elapsed, setElapsed] = React.useState(0);
  const startTimeRef = React.useRef<number | null>(null);
  const timingJobIdRef = React.useRef("");
  const selectedJobId = selectedJob?.id ?? "";
  const runningJob = jobs.find((item) => item.status === "running" || item.status === "queued");
  const latestJob = jobs[0] ?? null;
  const progressJob = selectedJob?.history_only ? (runningJob ?? latestJob) : selectedJob;

  const refreshJobs = React.useCallback(
    async (preferredJobId?: string) => {
      const [jobsPayload, outputsPayload] = await Promise.all([
        fetchJsonOr("/jobs", { jobs: [] }),
        fetchJsonOr("/output-files", { files: [], output_dir: "" })
      ]);
      const nextJobs = Array.isArray(jobsPayload.jobs) ? jobsPayload.jobs : [];
      const nextOutputs = Array.isArray(outputsPayload.files) ? outputsPayload.files : [];
      setJobs(nextJobs);
      setOutputs(nextOutputs);
      setOutputDir(outputsPayload.output_dir ?? "");

      const targetId = preferredJobId ?? selectedJobId;
      const nextSelected = targetId && !targetId.startsWith("output:")
        ? nextJobs.find((item: JobRecord) => item.id === targetId)
        : null;
      if (nextSelected) {
        onSelectJob(nextSelected);
      }
    },
    [onSelectJob, selectedJobId]
  );

  React.useEffect(() => {
    void refreshJobs();
    const timer = window.setInterval(() => void refreshJobs(), 2000);
    return () => window.clearInterval(timer);
  }, [refreshJobs]);

  React.useEffect(() => {
    if (!progressJob) {
      startTimeRef.current = null;
      timingJobIdRef.current = "";
      setElapsed(0);
      return;
    }

    const isActive = progressJob.status !== "complete" && progressJob.status !== "failed";
    const existingDuration = durationFromJob(progressJob);
    if (progressJob.id !== timingJobIdRef.current) {
      timingJobIdRef.current = progressJob.id;
      startTimeRef.current = isActive ? Date.now() - existingDuration * 1000 : null;
      setElapsed(existingDuration);
      return;
    }

    if (isActive && !startTimeRef.current) {
      startTimeRef.current = Date.now() - existingDuration * 1000;
    }
    if (!isActive) {
      startTimeRef.current = null;
      setElapsed(existingDuration);
    }
  }, [progressJob]);

  React.useEffect(() => {
    const timer = window.setInterval(() => {
      if (startTimeRef.current && progressJob?.status !== "complete" && progressJob?.status !== "failed") {
        setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [progressJob?.status]);

  async function start(pathOverride?: string) {
    const path = normalizeSourcePath(pathOverride ?? sourcePath);
    setBusy(true);
    setErrorMessage("");
    try {
      const response = await fetch("/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_path: path, trim_settings: trimSettings })
      });
      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        if (response.status === 422) {
          setErrorMessage(`Invalid file: ${payload.detail ?? "Unable to validate file"}`);
        } else if (response.status === 503) {
          setErrorMessage("Server not ready, try again");
        } else {
          setErrorMessage(payload.detail ?? "Unable to start job");
        }
        return;
      }

      if (!payload.job_id) {
        setErrorMessage("Server did not return a job id");
        return;
      }

      setSourcePath(path);
      startTimeRef.current = Date.now();
      setElapsed(0);
      const next = await fetch(`/jobs/${payload.job_id}`).then((r) => r.json());
      onSelectJob(next);
      await refreshJobs(payload.job_id);
    } finally {
      setBusy(false);
    }
  }

  const job = progressJob;
  const activeStage = job?.stage ?? "queued";
  const failedStage = job?.stage_failed ?? activeStage;
  const activeIndex = stages.indexOf(activeStage as (typeof stages)[number]);
  const failedIndex = stages.indexOf(failedStage as (typeof stages)[number]);
  const displayElapsed = elapsed || durationFromJob(job);

  return (
    <aside className="border border-lane bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <h2 className="text-base font-semibold">Jobs</h2>
      <div className="mt-4 grid gap-3">
        <label className="grid gap-1 text-sm font-medium">
          Source MP4 path
          <input
            className="border border-lane bg-white px-3 py-2 text-sm outline-none focus:border-accent dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
            value={sourcePath}
            onChange={(event) => setSourcePath(event.target.value)}
            placeholder="C:/Videos/clip.mp4"
          />
          <span className="text-xs font-normal leading-5 text-slate-500 dark:text-slate-400">
            Enter the full path to your Medal.tv clip
            <br />
            {"Example: D:\\Medal\\Clips\\League of Legends\\clip.mp4"}
          </span>
        </label>
        <button
          className="bg-accent px-3 py-2 text-sm font-semibold text-white disabled:opacity-50"
          disabled={!sourcePath || busy}
          onClick={() => void start()}
          type="button"
        >
          {busy ? "Starting" : "Start"}
        </button>
        {errorMessage ? <p className="border border-danger p-3 text-sm text-danger dark:bg-red-950/40">{errorMessage}</p> : null}
      </div>

      <div className="mt-4 border-t border-lane pt-4 dark:border-slate-800">
        <div className="flex items-center justify-between gap-3">
          <button
            aria-expanded={advancedOpen}
            className="text-left text-sm font-semibold"
            onClick={() => setAdvancedOpen((value) => !value)}
            type="button"
          >
            Advanced Trim Settings
          </button>
          <button
            className="shrink-0 border border-lane px-2 py-1 text-xs font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300"
            onClick={() => setAdvancedOpen((value) => !value)}
            type="button"
          >
            {advancedOpen ? "Hide" : "Show"}
          </button>
        </div>
        {advancedOpen ? (
          <div className="mt-3 grid gap-4 text-sm">
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(trimPresets).map(([name, preset]) => (
                <button
                  className="border border-lane px-2 py-1.5 text-xs font-semibold hover:border-accent dark:border-slate-700"
                  key={name}
                  onClick={() => setTrimSettings(preset)}
                  type="button"
                >
                  {name}
                </button>
              ))}
            </div>
            <TrimSlider
              label="Incoming rewind"
              max={4}
              min={0}
              onChange={(value) => setTrimSettings((settings) => ({ ...settings, fight_start_preroll_sec: value }))}
              step={0.1}
              value={trimSettings.fight_start_preroll_sec}
            />
            <TrimSlider
              label="Max pre-fight lead"
              max={6}
              min={0}
              onChange={(value) => setTrimSettings((settings) => ({ ...settings, max_pre_fight_lead_sec: value }))}
              step={0.1}
              value={trimSettings.max_pre_fight_lead_sec}
            />
            <TrimSlider
              label="Final context"
              max={5}
              min={0}
              onChange={(value) => setTrimSettings((settings) => ({ ...settings, output_context_padding_sec: value }))}
              step={0.1}
              value={trimSettings.output_context_padding_sec}
            />
            <TrimSlider
              label="Post-fight hold"
              max={8}
              min={0}
              onChange={(value) => setTrimSettings((settings) => ({ ...settings, combat_event_end_padding_sec: value }))}
              step={0.1}
              value={trimSettings.combat_event_end_padding_sec}
            />
            <TrimSlider
              label="Minimum clip"
              max={35}
              min={8}
              onChange={(value) => setTrimSettings((settings) => ({ ...settings, min_clip_duration_sec: value }))}
              step={1}
              value={trimSettings.min_clip_duration_sec}
            />
          </div>
        ) : null}
      </div>

      <div className="mt-5 border-t border-lane pt-4 dark:border-slate-800">
        <div className="flex items-center justify-between gap-3">
          <button
            aria-expanded={outputsOpen}
            className="text-left text-sm font-semibold"
            onClick={() => setOutputsOpen((value) => !value)}
            type="button"
          >
            Previous Outputs
          </button>
          <button
            className="shrink-0 border border-lane px-2 py-1 text-xs font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300"
            onClick={() => setOutputsOpen((value) => !value)}
            type="button"
          >
            {outputsOpen ? "Hide" : `Show ${outputs.length}`}
          </button>
        </div>
        {outputsOpen && outputDir ? <p className="mt-1 truncate text-xs text-slate-500 dark:text-slate-400">{outputDir}</p> : null}
        {outputsOpen ? <div className="mt-3 grid max-h-80 gap-2 overflow-auto pr-1">
          {outputs.length ? (
            outputs.map((output) => {
              const historyJob = outputFileToJob(output);
              const isSelected = selectedJob?.history_only && selectedJob.output_path === output.path;
              return (
                <button
                  key={output.path}
                  className={`grid gap-1 border px-3 py-2 text-left text-sm ${
                    isSelected
                      ? "border-accent bg-blue-50 dark:bg-blue-950/30"
                      : "border-lane hover:border-accent dark:border-slate-800"
                  }`}
                  onClick={() => onSelectJob(historyJob)}
                  type="button"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="min-w-0 truncate font-medium">{output.filename}</span>
                    <span className="shrink-0 text-xs text-slate-500 dark:text-slate-400">{formatFileSize(output.size)}</span>
                  </div>
                  <span className="truncate text-xs text-slate-500 dark:text-slate-400">{formatModified(output.modified_at)}</span>
                </button>
              );
            })
          ) : (
            <p className="border border-dashed border-lane p-3 text-sm text-slate-500 dark:border-slate-800 dark:text-slate-400">
              No completed clips found.
            </p>
          )}
        </div> : null}
      </div>

      <div className="mt-5 border-t border-lane pt-4 dark:border-slate-800">
        <h3 className="mb-3 text-sm font-semibold">Current Job Progress</h3>
        {selectedJob?.history_only ? (
          <p className="mb-3 text-xs text-slate-500 dark:text-slate-400">
            Viewing previous output: {displaySource(selectedJob.output_path)}
          </p>
        ) : null}
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="break-all font-medium">{job?.id ?? "No job selected"}</span>
          <span className="shrink-0 text-slate-600 dark:text-slate-400">{activeJobLabel(job)}</span>
        </div>
        <div className="mt-4 grid gap-3">
          {stages.map((stage, index) => {
            const isFailed = job?.status === "failed" && stage === failedStage;
            const isComplete = job?.status === "complete" || (activeIndex > index && !isFailed) || (failedIndex > index && job?.status === "failed");
            const isCurrent = Boolean(job) && job?.status !== "complete" && job?.status !== "failed" && stage === activeStage;
            const icon = isFailed ? "x" : isComplete ? "+" : isCurrent ? "*" : "-";
            const iconClass = isFailed
              ? "text-danger"
              : isComplete
                ? "text-success"
                : isCurrent
                  ? "animate-pulse text-accent"
                  : "text-slate-400 dark:text-slate-600";

            return (
              <div key={stage} className="grid grid-cols-[20px_1fr] gap-2 text-sm">
                <span className={`pt-0.5 font-semibold ${iconClass}`}>{icon}</span>
                <div className="min-w-0">
                  <span className={isCurrent ? "font-semibold" : "text-slate-700 dark:text-slate-300"}>{stageNames[stage]}</span>
                  {isCurrent ? (
                    <div className="mt-2">
                      <div className="h-2 w-full overflow-hidden bg-lane dark:bg-slate-800">
                        <div className="h-full bg-accent" style={{ width: `${Math.max(0, Math.min(100, job?.progress ?? 0))}%` }} />
                      </div>
                      <div className="mt-1 flex items-start justify-between gap-2 text-xs text-slate-600 dark:text-slate-400">
                        <span>{job?.status_message}</span>
                        <span className="shrink-0">{job?.progress ?? 0}%</span>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            );
          })}
        </div>

        {job?.status === "queued" ? <p className="mt-4 text-sm text-slate-600 dark:text-slate-400">Waiting for {formatElapsed(displayElapsed)}</p> : null}
        {job?.status === "running" ? <p className="mt-4 text-sm text-slate-600 dark:text-slate-400">Running for {formatElapsed(displayElapsed)}</p> : null}
        {job?.status === "complete" ? <p className="mt-4 text-sm text-slate-600 dark:text-slate-400">Completed in {formatElapsed(displayElapsed)}</p> : null}
        {job?.status === "failed" ? <p className="mt-4 text-sm text-slate-600 dark:text-slate-400">Failed after {formatElapsed(displayElapsed)}</p> : null}

        {job?.status === "failed" ? (
          <div className="mt-4 border border-danger p-3 text-sm dark:bg-red-950/30">
            <p className="font-semibold text-danger">Pipeline Failed</p>
            <p className="mt-1 text-slate-700 dark:text-slate-300">Failed at: {stageNames[failedStage] ?? failedStage}</p>
            <p className="mt-2 whitespace-pre-wrap break-words text-danger">{job.error_detail}</p>
            <button
              className="mt-3 border border-danger px-3 py-2 text-sm font-semibold text-danger disabled:opacity-50"
              disabled={busy || !(job.source_path || sourcePath)}
              onClick={() => void start(job.source_path ?? sourcePath)}
              type="button"
            >
              Retry
            </button>
          </div>
        ) : null}
      </div>
    </aside>
  );
}

function TrimSlider({
  label,
  max,
  min,
  onChange,
  step,
  value
}: {
  label: string;
  max: number;
  min: number;
  onChange: (value: number) => void;
  step: number;
  value: number;
}) {
  return (
    <label className="grid gap-1">
      <span className="flex items-center justify-between gap-3">
        <span className="font-medium">{label}</span>
        <span className="text-xs text-slate-500 dark:text-slate-400">{value.toFixed(step >= 1 ? 0 : 1)}s</span>
      </span>
      <input
        className="accent-blue-600"
        max={max}
        min={min}
        onChange={(event) => onChange(Number(event.target.value))}
        step={step}
        type="range"
        value={value}
      />
    </label>
  );
}
