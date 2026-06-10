import React from "react";
import type { CaptionPayload, JobRecord } from "./types";

export function CaptionEditor({ job }: { job: JobRecord | null }) {
  const captions = safeJson<Record<string, CaptionPayload>>(job?.captions, {});
  const active = captions.default;
  const [text, setText] = React.useState("");
  const outputUrl = outputUrlFromPath(job?.output_path);

  React.useEffect(() => {
    setText(active?.caption ?? "");
  }, [active?.caption]);

  const description = [text, (active?.hashtags ?? []).join(" ")].filter(Boolean).join("\n\n");

  return (
    <section className="border border-lane bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold">Output</h2>
        {outputUrl ? (
          <div className="flex shrink-0 gap-2 text-sm">
            <a className="border border-lane px-3 py-1.5 font-semibold text-slate-700" href={outputUrl} target="_blank" rel="noreferrer">
              Open
            </a>
            <a className="bg-accent px-3 py-1.5 font-semibold text-white" href={outputUrl} download>
              Download
            </a>
          </div>
        ) : null}
      </div>
      {outputUrl ? (
        <video className="mt-4 aspect-[3/4] max-h-[640px] w-full bg-black object-contain" src={outputUrl} controls playsInline />
      ) : (
        <div className="mt-4 flex aspect-[3/4] max-h-[640px] w-full items-center justify-center border border-lane bg-slate-50 text-sm text-slate-500">
          No completed clip selected
        </div>
      )}

      <h2 className="mt-5 text-base font-semibold">Description</h2>
      <textarea
        className="mt-4 min-h-40 w-full resize-y border border-lane p-3 text-sm outline-none focus:border-accent"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="description"
      />
      <div className="mt-3 flex items-center justify-between text-sm text-slate-600">
        <span>{description.length} chars</span>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {(active?.hashtags ?? []).map((tag) => (
          <span className="border border-lane px-2 py-1 text-xs" key={tag}>
            {tag}
          </span>
        ))}
      </div>
    </section>
  );
}

function safeJson<T>(value: string | undefined | null, fallback: T): T {
  try {
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function outputUrlFromPath(value: string | undefined | null) {
  if (!value) return "";
  const filename = value.split(/[\\/]/).pop();
  return filename ? `/outputs/${encodeURIComponent(filename)}` : "";
}
