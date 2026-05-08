import type { JobRecord } from "./types";

export function VideoPreview({ job }: { job: JobRecord | null }) {
  const outputUrl = job?.output_path ? `/outputs/${job.output_path.split(/[\\/]/).pop()}` : "";
  return (
    <section className="border border-lane bg-white p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold">Preview</h2>
        <span className="text-sm text-slate-600">{job?.stage ?? "idle"}</span>
      </div>
      <div className="mt-4 grid gap-4 md:grid-cols-[1fr_260px]">
        <div className="aspect-video bg-ink">
          <div className="grid h-full place-items-center text-sm text-white/70">original</div>
        </div>
        <div className="mx-auto aspect-[3/4] h-[360px] bg-ink">
          {outputUrl ? <video className="h-full w-full object-cover" src={outputUrl} controls /> : <div className="grid h-full place-items-center text-sm text-white/70">output</div>}
        </div>
      </div>
    </section>
  );
}
