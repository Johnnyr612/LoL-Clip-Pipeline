import React from "react";
import type { CaptionPayload, JobRecord } from "./types";

export function CaptionEditor({ job }: { job: JobRecord | null }) {
  const captions = safeJson<Record<string, CaptionPayload>>(job?.captions, {});
  const active = captions.default;
  const [text, setText] = React.useState("");

  React.useEffect(() => {
    setText(active?.caption ?? "");
  }, [active?.caption]);

  const description = [text, (active?.hashtags ?? []).join(" ")].filter(Boolean).join("\n\n");

  return (
    <section className="border border-lane bg-white p-4">
      <h2 className="text-base font-semibold">Description</h2>
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
