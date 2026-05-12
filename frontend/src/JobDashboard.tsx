import React from "react";
import type { JobRecord } from "./types";

const stages = [
  "stage1_decode",
  "stage2_minimap",
  "stage3_fight",
  "stage4_crop",
  "stage5_encode",
  "stage6_caption",
  "complete"
] as const;

const stageNames: Record<string, string> = {
  stage1_decode: "Frame Extraction",
  stage2_minimap: "Champion Detection",
  stage3_fight: "Fight Detection",
  stage4_crop: "Crop Trajectory",
  stage5_encode: "Video Encoding",
  stage6_caption: "Caption Generation",
  complete: "Complete",
  queued: "Queued"
};

type Props = {
  selectedJob: JobRecord | null;
  onSelectJob: (job: JobRecord | null) => void;
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

export function JobDashboard({ selectedJob, onSelectJob }: Props) {
  const [sourcePath, setSourcePath] = React.useState("");
  const [jobId, setJobId] = React.useState("");
  const [job, setJob] = React.useState<JobRecord | null>(selectedJob);
  const [busy, setBusy] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState("");
  const [elapsed, setElapsed] = React.useState(0);
  const startTimeRef = React.useRef<number | null>(null);

  React.useEffect(() => {
    setJob(selectedJob);
  }, [selectedJob]);

  React.useEffect(() => {
    if (!jobId) return;

    async function pollJob() {
      const next = await fetch(`/jobs/${jobId}`).then((r) => (r.ok ? r.json() : null));
      if (next) {
        setJob(next);
        onSelectJob(next);
      }
    }

    void pollJob();
    const timer = window.setInterval(pollJob, 2000);
    return () => window.clearInterval(timer);
  }, [jobId, onSelectJob]);

  React.useEffect(() => {
    if (!jobId) return;
    startTimeRef.current = Date.now();
    setElapsed(0);
  }, [jobId]);

  React.useEffect(() => {
    const timer = window.setInterval(() => {
      if (startTimeRef.current && job?.status !== "complete" && job?.status !== "failed") {
        setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [job?.status]);

  async function start(pathOverride?: string) {
    const path = pathOverride ?? sourcePath;
    setBusy(true);
    setErrorMessage("");
    try {
      const response = await fetch("/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_path: path })
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
      setJobId(payload.job_id);
      startTimeRef.current = Date.now();
      setElapsed(0);
      const next = await fetch(`/jobs/${payload.job_id}`).then((r) => r.json());
      setJob(next);
      onSelectJob(next);
    } finally {
      setBusy(false);
    }
  }

  const activeStage = job?.stage ?? "queued";
  const failedStage = job?.stage_failed ?? activeStage;
  const activeIndex = stages.indexOf(activeStage as (typeof stages)[number]);
  const failedIndex = stages.indexOf(failedStage as (typeof stages)[number]);
  const displayElapsed = elapsed || durationFromJob(job);
  const isRunning = Boolean(job && job.status !== "complete" && job.status !== "failed");

  return (
    <aside className="border border-lane bg-white p-4">
      <h2 className="text-base font-semibold">Jobs</h2>
      <div className="mt-4 grid gap-3">
        <label className="grid gap-1 text-sm font-medium">
          Source MP4 path
          <input
            className="border border-lane px-3 py-2 text-sm outline-none focus:border-accent"
            value={sourcePath}
            onChange={(event) => setSourcePath(event.target.value)}
            placeholder="C:/Videos/clip.mp4"
          />
          <span className="text-xs font-normal leading-5 text-slate-500">
            Enter the full path to your Medal.tv clip
            <br />
            {"Example: D:\\Medal\\Clips\\League of Legends\\clip.mp4"}
          </span>
        </label>
        <button
          className="bg-accent px-3 py-2 text-sm font-semibold text-white disabled:opacity-50"
          disabled={!sourcePath || busy}
          onClick={() => void start()}
        >
          {busy ? "Starting" : "Start"}
        </button>
        {errorMessage ? <p className="border border-danger p-3 text-sm text-danger">{errorMessage}</p> : null}
      </div>

      <div className="mt-5 border-t border-lane pt-4">
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="break-all font-medium">{job?.id ?? "No job selected"}</span>
          <span className="shrink-0 text-slate-600">{job?.status ?? "idle"}</span>
        </div>
        <div className="mt-4 grid gap-3">
          {stages.map((stage, index) => {
            const isFailed = job?.status === "failed" && stage === failedStage;
            const isComplete = job?.status === "complete" || (activeIndex > index && !isFailed) || (failedIndex > index && job?.status === "failed");
            const isCurrent = job?.status !== "complete" && job?.status !== "failed" && stage === activeStage;
            const icon = isFailed ? "✗" : isComplete ? "✓" : isCurrent ? "●" : "○";
            const iconClass = isFailed
              ? "text-danger"
              : isComplete
                ? "text-success"
                : isCurrent
                  ? "animate-pulse text-accent"
                  : "text-slate-400";

            return (
              <div key={stage} className="grid grid-cols-[20px_1fr] gap-2 text-sm">
                <span className={`pt-0.5 font-semibold ${iconClass}`}>{icon}</span>
                <div className="min-w-0">
                  <span className={isCurrent ? "font-semibold" : "text-slate-700"}>{stageNames[stage]}</span>
                  {isCurrent ? (
                    <div className="mt-2">
                      <div className="h-2 w-full overflow-hidden bg-lane">
                        <div className="h-full bg-accent" style={{ width: `${Math.max(0, Math.min(100, job?.progress ?? 0))}%` }} />
                      </div>
                      <div className="mt-1 flex items-start justify-between gap-2 text-xs text-slate-600">
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

        {job && isRunning ? <p className="mt-4 text-sm text-slate-600">Running for {formatElapsed(displayElapsed)}</p> : null}
        {job?.status === "complete" ? <p className="mt-4 text-sm text-slate-600">Completed in {formatElapsed(displayElapsed)}</p> : null}
        {job?.status === "failed" ? <p className="mt-4 text-sm text-slate-600">Failed after {formatElapsed(displayElapsed)}</p> : null}

        {job?.status === "failed" ? (
          <div className="mt-4 border border-danger p-3 text-sm">
            <p className="font-semibold text-danger">Pipeline Failed</p>
            <p className="mt-1 text-slate-700">Failed at: {stageNames[failedStage] ?? failedStage}</p>
            <p className="mt-2 whitespace-pre-wrap break-words text-danger">{job.error_detail}</p>
            <button
              className="mt-3 border border-danger px-3 py-2 text-sm font-semibold text-danger disabled:opacity-50"
              disabled={busy || !(job.source_path || sourcePath)}
              onClick={() => void start(job.source_path ?? sourcePath)}
            >
              Retry
            </button>
          </div>
        ) : null}
      </div>
    </aside>
  );
}
