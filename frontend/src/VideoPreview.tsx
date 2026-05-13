import type { JobRecord } from "./types";

export function VideoPreview({ job }: { job: JobRecord | null }) {
  const sourceUrl = job?.id && job.source_path ? `/jobs/${job.id}/source` : "";
  const outputUrl = job?.output_path ? `/outputs/${job.output_path.split(/[\\/]/).pop()}` : "";
  const sourceName = job?.source_path?.split(/[\\/]/).pop() ?? "";
  return (
    <section className="min-w-0 overflow-hidden border border-lane bg-white p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold">Preview</h2>
        <span className="text-sm text-slate-600">{job?.stage ?? "idle"}</span>
      </div>
      <div className="mt-4 grid min-w-0 gap-4 md:grid-cols-[minmax(0,1fr)_210px]">
        <div className="min-w-0">
          <div className="aspect-video bg-ink">
            {sourceUrl ? <video className="h-full w-full object-contain" src={sourceUrl} controls preload="metadata" /> : <div className="grid h-full place-items-center text-sm text-white/70">original</div>}
          </div>
          <p className="mt-2 truncate text-xs text-slate-600">{sourceName || "Original source file"}</p>
        </div>
        <div className="mx-auto">
          <div className="aspect-[3/4] h-[280px] bg-ink">
            {outputUrl ? <video className="h-full w-full object-cover" src={outputUrl} controls preload="metadata" /> : <div className="grid h-full place-items-center text-sm text-white/70">output</div>}
          </div>
          <p className="mt-2 text-center text-xs text-slate-600">Generated clip</p>
        </div>
      </div>
    </section>
  );
}
