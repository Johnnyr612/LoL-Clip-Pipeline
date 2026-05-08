import React from "react";
import type { JobRecord } from "./types";

const stages = ["stage1_decode", "stage2_minimap", "stage3_fight", "stage4_crop", "stage5_encode", "stage6_caption", "complete"];

type Props = {
  selectedJob: JobRecord | null;
  onSelectJob: (job: JobRecord | null) => void;
};

export function JobDashboard({ selectedJob, onSelectJob }: Props) {
  const [sourcePath, setSourcePath] = React.useState("");
  const [jobId, setJobId] = React.useState("");
  const [job, setJob] = React.useState<JobRecord | null>(selectedJob);
  const [busy, setBusy] = React.useState(false);

  React.useEffect(() => {
    setJob(selectedJob);
  }, [selectedJob]);

  React.useEffect(() => {
    if (!jobId) return;
    const timer = window.setInterval(async () => {
      const next = await fetch(`/jobs/${jobId}`).then((r) => (r.ok ? r.json() : null));
      if (next) {
        setJob(next);
        onSelectJob(next);
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [jobId, onSelectJob]);

  async function start() {
    setBusy(true);
    try {
      const response = await fetch("/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_path: sourcePath })
      });
      const payload = await response.json();
      setJobId(payload.job_id);
      const next = await fetch(`/jobs/${payload.job_id}`).then((r) => r.json());
      setJob(next);
      onSelectJob(next);
    } finally {
      setBusy(false);
    }
  }

  const stageIndex = Math.max(0, stages.indexOf(job?.stage ?? ""));

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
        </label>
        <button
          className="bg-accent px-3 py-2 text-sm font-semibold text-white disabled:opacity-50"
          disabled={!sourcePath || busy}
          onClick={start}
        >
          {busy ? "Starting" : "Start"}
        </button>
      </div>

      <div className="mt-5 border-t border-lane pt-4">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">{job?.id ?? "No job selected"}</span>
          <span className="text-slate-600">{job?.status ?? "idle"}</span>
        </div>
        <div className="mt-4 grid gap-2">
          {stages.map((stage, index) => (
            <div key={stage} className="grid grid-cols-[18px_1fr] items-center gap-2 text-sm">
              <span className={index <= stageIndex ? "h-2 w-2 bg-accent" : "h-2 w-2 bg-lane"} />
              <span className={index === stageIndex ? "font-semibold" : "text-slate-600"}>{stage.replace("stage", "stage ")}</span>
            </div>
          ))}
        </div>
        {job?.error_detail ? <p className="mt-4 border border-danger p-3 text-sm text-danger">{job.error_detail}</p> : null}
      </div>
    </aside>
  );
}
