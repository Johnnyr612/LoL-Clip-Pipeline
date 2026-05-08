import React from "react";
import type { CaptionPayload, JobRecord } from "./types";

export function CaptionEditor({ job }: { job: JobRecord | null }) {
  const captions = safeJson<Record<string, CaptionPayload>>(job?.captions, {});
  const [platform, setPlatform] = React.useState<"tiktok" | "instagram">("tiktok");
  const active = captions[platform];
  const [text, setText] = React.useState("");

  React.useEffect(() => {
    setText(active?.caption ?? "");
  }, [active?.caption]);

  return (
    <section className="border border-lane bg-white p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold">Captions</h2>
        <div className="grid grid-cols-2 border border-lane text-sm">
          {(["tiktok", "instagram"] as const).map((item) => (
            <button key={item} className={platform === item ? "bg-ink px-3 py-1 text-white" : "px-3 py-1"} onClick={() => setPlatform(item)}>
              {item}
            </button>
          ))}
        </div>
      </div>
      <textarea
        className="mt-4 min-h-40 w-full resize-y border border-lane p-3 text-sm outline-none focus:border-accent"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="caption"
      />
      <div className="mt-3 flex items-center justify-between text-sm text-slate-600">
        <span>{text.length} chars</span>
        <button className="border border-lane px-3 py-1 font-medium text-ink">Post</button>
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
