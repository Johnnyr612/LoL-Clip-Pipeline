import React from "react";
import type { CaptionPayload, JobRecord } from "./types";

export function CaptionEditor({ job }: { job: JobRecord | null }) {
  const captions = safeJson<Record<string, CaptionPayload>>(job?.captions, {});
  const [platform, setPlatform] = React.useState<"tiktok" | "instagram">("tiktok");
  const active = captions[platform];
  const [text, setText] = React.useState("");
  const [busyAction, setBusyAction] = React.useState<"post" | "draft" | "instagram" | null>(null);
  const [message, setMessage] = React.useState("");

  React.useEffect(() => {
    setText(active?.caption ?? "");
    setMessage("");
  }, [active?.caption]);

  const description = [text, (active?.hashtags ?? []).join(" ")].filter(Boolean).join("\n\n");

  async function sendTikTok(mode: "post" | "draft") {
    if (!job?.id) return;
    setBusyAction(mode);
    setMessage("");
    try {
      const response = await fetch("/upload/tiktok", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: job.id, mode, description })
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.status !== "ok") {
        setMessage(payload.detail ?? "Unable to send to TikTok");
        return;
      }
      if (mode === "draft") {
        await navigator.clipboard?.writeText(description).catch(() => undefined);
        setMessage("Sent to TikTok inbox. Open TikTok to finish the draft; the description was copied for pasting if TikTok does not prefill it.");
      } else {
        setMessage("Posted to TikTok.");
      }
    } finally {
      setBusyAction(null);
    }
  }

  async function sendInstagram() {
    if (!job?.id) return;
    setBusyAction("instagram");
    setMessage("");
    try {
      const response = await fetch("/upload/instagram", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: job.id, description })
      });
      const payload = await response.json().catch(() => ({}));
      setMessage(!response.ok || payload.status !== "ok" ? payload.detail ?? "Unable to post to Instagram" : "Posted to Instagram.");
    } finally {
      setBusyAction(null);
    }
  }

  return (
    <section className="border border-lane bg-white p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold">Description</h2>
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
        placeholder="description"
      />
      <div className="mt-3 flex items-center justify-between text-sm text-slate-600">
        <span>{description.length} chars</span>
        <div className="flex flex-wrap justify-end gap-2">
          {platform === "tiktok" ? (
            <>
              <button className="border border-lane px-3 py-1 font-medium text-ink disabled:opacity-50" disabled={!job?.id || busyAction !== null} onClick={() => void sendTikTok("draft")}>
                {busyAction === "draft" ? "Saving" : "Save draft"}
              </button>
              <button className="border border-lane px-3 py-1 font-medium text-ink disabled:opacity-50" disabled={!job?.id || busyAction !== null} onClick={() => void sendTikTok("post")}>
                {busyAction === "post" ? "Posting" : "Post"}
              </button>
            </>
          ) : (
            <button className="border border-lane px-3 py-1 font-medium text-ink disabled:opacity-50" disabled={!job?.id || busyAction !== null} onClick={() => void sendInstagram()}>
              {busyAction === "instagram" ? "Posting" : "Post"}
            </button>
          )}
        </div>
      </div>
      {message ? <p className="mt-3 border border-lane p-2 text-sm text-slate-700">{message}</p> : null}
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
