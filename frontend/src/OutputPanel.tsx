import React from "react";
import type { DetectionDebug, DetectionDebugResult, JobRecord, TikTokStatus } from "./types";

export function OutputPanel({ job }: { job: JobRecord | null }) {
  const detectionDebug = safeJson<DetectionDebug>(job?.detection_debug, {});
  const outputUrl = outputUrlFromPath(job?.output_path);
  const [tiktokStatus, setTikTokStatus] = React.useState<TikTokStatus | null>(null);
  const [mode, setMode] = React.useState<"inbox" | "direct">("inbox");
  const [title, setTitle] = React.useState("");
  const [privacyLevel, setPrivacyLevel] = React.useState("SELF_ONLY");
  const [disableComment, setDisableComment] = React.useState(false);
  const [disableDuet, setDisableDuet] = React.useState(false);
  const [disableStitch, setDisableStitch] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [message, setMessage] = React.useState("");

  React.useEffect(() => {
    void refreshTikTokStatus();
  }, []);

  async function refreshTikTokStatus() {
    const status = await fetch("/tiktok/status").then((response) => response.json());
    setTikTokStatus(status);
  }

  async function publishToTikTok() {
    if (!job?.id) return;
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch(`/tiktok/jobs/${job.id}/publish`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode,
          title,
          privacy_level: privacyLevel,
          disable_comment: disableComment,
          disable_duet: disableDuet,
          disable_stitch: disableStitch,
          // Clips are gameplay footage generated from recorded video, not AI-generated media.
          is_aigc: false
        })
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        setMessage(payload.detail ?? "TikTok upload failed");
        return;
      }
      setMessage(mode === "inbox" ? `Sent to TikTok inbox: ${payload.publish_id}` : `Direct post initialized: ${payload.publish_id}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="surface-panel p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="section-kicker">Rendered clip</p>
          <h2 className="section-title mt-1">Output</h2>
          {job?.output_path ? <p className="mt-1 break-all text-xs text-slate-500 dark:text-slate-400">{job.output_path}</p> : null}
        </div>
        {outputUrl ? (
          <div className="flex shrink-0 gap-2 text-sm">
            <a className="button min-h-8 py-1.5" href={outputUrl} target="_blank" rel="noreferrer">
              Open
            </a>
            <a className="button-primary min-h-8 py-1.5" href={outputUrl} download>
              Download
            </a>
          </div>
        ) : null}
      </div>
      {outputUrl ? (
        <div className="mt-4 rounded-lg border border-slate-200 bg-slate-950 p-3 shadow-inner dark:border-slate-800">
          <video
            key={outputUrl}
            className="mx-auto aspect-[3/4] max-h-[640px] w-full rounded-md bg-black object-contain"
            src={outputUrl}
            controls
            playsInline
            preload="metadata"
          />
        </div>
      ) : (
        <div className="surface-muted mt-4 flex aspect-[3/4] max-h-[640px] w-full items-center justify-center p-4 text-center text-sm text-slate-500 dark:text-slate-400">
          Select a completed job or previous output to preview the vertical clip.
        </div>
      )}

      <TikTokPanel
        busy={busy}
        connected={Boolean(tiktokStatus?.connected)}
        configured={Boolean(tiktokStatus?.configured)}
        scope={tiktokStatus?.scope ?? ""}
        disableComment={disableComment}
        disableDuet={disableDuet}
        disableStitch={disableStitch}
        hasOutput={Boolean(job?.id && !job.history_only && job.status === "complete" && outputUrl)}
        message={message}
        mode={mode}
        onConnect={() => {
          window.location.href = `/tiktok/auth?mode=${mode}`;
        }}
        onDisconnect={async () => {
          await fetch("/tiktok/disconnect", { method: "POST" });
          await refreshTikTokStatus();
        }}
        onPublish={() => void publishToTikTok()}
        privacyLevel={privacyLevel}
        setDisableComment={setDisableComment}
        setDisableDuet={setDisableDuet}
        setDisableStitch={setDisableStitch}
        setMode={setMode}
        setPrivacyLevel={setPrivacyLevel}
        setTitle={setTitle}
        title={title}
      />

      <ChampionDetectionPanel detectionDebug={detectionDebug} />
    </section>
  );
}

function TikTokPanel({
  busy,
  connected,
  configured,
  disableComment,
  disableDuet,
  disableStitch,
  hasOutput,
  message,
  mode,
  onConnect,
  onDisconnect,
  onPublish,
  privacyLevel,
  setDisableComment,
  setDisableDuet,
  setDisableStitch,
  setMode,
  setPrivacyLevel,
  setTitle,
  scope,
  title
}: {
  busy: boolean;
  connected: boolean;
  configured: boolean;
  disableComment: boolean;
  disableDuet: boolean;
  disableStitch: boolean;
  hasOutput: boolean;
  message: string;
  mode: "inbox" | "direct";
  onConnect: () => void;
  onDisconnect: () => void;
  onPublish: () => void;
  privacyLevel: string;
  setDisableComment: (value: boolean) => void;
  setDisableDuet: (value: boolean) => void;
  setDisableStitch: (value: boolean) => void;
  setMode: (value: "inbox" | "direct") => void;
  setPrivacyLevel: (value: string) => void;
  setTitle: (value: string) => void;
  scope: string;
  title: string;
}) {
  const requiredScope = mode === "direct" ? "video.publish" : "video.upload";
  const hasRequiredScope = scope.split(",").map((item) => item.trim()).includes(requiredScope);
  return (
    <section className="divider mt-6 pt-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="section-kicker">Publishing</p>
          <h2 className="section-title mt-1">TikTok</h2>
        </div>
        <div className="flex flex-wrap gap-2 text-sm">
          {connected ? (
            <button className="button min-h-8 py-1.5" onClick={onDisconnect} type="button">
              Disconnect
            </button>
          ) : (
            <button className="button-primary min-h-8 py-1.5" disabled={!configured} onClick={onConnect} type="button">
              Connect TikTok
            </button>
          )}
          {connected && !hasRequiredScope ? (
            <button className="button min-h-8 py-1.5" onClick={onConnect} type="button">
              Authorize {mode === "direct" ? "Direct Post" : "Inbox Upload"}
            </button>
          ) : null}
          <button className="button-primary min-h-8 py-1.5" disabled={!configured || !connected || !hasRequiredScope || !hasOutput || busy} onClick={onPublish} type="button">
            {busy ? "Sending" : mode === "inbox" ? "Send to Inbox" : "Direct Post"}
          </button>
        </div>
      </div>

      {!configured ? <p className="mt-3 rounded-md border border-danger bg-red-50 p-3 text-sm text-danger dark:bg-red-950/30">Set TikTok client key, secret, and redirect URI before connecting.</p> : null}

      <div className="mt-4 grid gap-3 text-sm">
        <label className="field-label">
          Mode
          <select className="input-field" value={mode} onChange={(event) => setMode(event.target.value as "inbox" | "direct")}>
            <option value="inbox">Upload to TikTok inbox</option>
            <option value="direct">Direct Post</option>
          </select>
        </label>

        {mode === "direct" ? (
          <div className="grid gap-3">
            <label className="field-label">
              Post title
              <input className="input-field" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Optional TikTok caption/title" />
            </label>
            <label className="field-label">
              Privacy
              <select className="input-field" value={privacyLevel} onChange={(event) => setPrivacyLevel(event.target.value)}>
                <option value="SELF_ONLY">Self only</option>
                <option value="MUTUAL_FOLLOW_FRIENDS">Mutual friends</option>
                <option value="FOLLOWER_OF_CREATOR">Followers</option>
                <option value="PUBLIC_TO_EVERYONE">Public</option>
              </select>
            </label>
            <div className="flex flex-wrap gap-3 rounded-md border border-lane bg-slate-50 p-3 text-slate-700 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-300">
              <label className="flex items-center gap-2"><input checked={disableComment} onChange={(event) => setDisableComment(event.target.checked)} type="checkbox" /> Comments off</label>
              <label className="flex items-center gap-2"><input checked={disableDuet} onChange={(event) => setDisableDuet(event.target.checked)} type="checkbox" /> Duets off</label>
              <label className="flex items-center gap-2"><input checked={disableStitch} onChange={(event) => setDisableStitch(event.target.checked)} type="checkbox" /> Stitches off</label>
            </div>
          </div>
        ) : (
          <p className="surface-muted p-3 text-slate-600 dark:text-slate-400">Inbox upload sends the clip to TikTok so the creator can finish editing and posting there.</p>
        )}
      </div>

      {message ? <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">{message}</p> : null}
    </section>
  );
}

function ChampionDetectionPanel({ detectionDebug }: { detectionDebug: DetectionDebug }) {
  const frames = detectionDebug.frames ?? [];
  const summary = detectionDebug.summary;
  return (
    <section className="divider mt-6 pt-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="section-kicker">Debug review</p>
          <h2 className="section-title mt-1">Champion Detection</h2>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600 dark:text-slate-400">
            The pipeline treats the white minimap camera box as the recording anchor. These are sampled minimap crops from the final clipped time range with YOLO champion detections overlaid.
          </p>
        </div>
        {summary ? (
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="chip chip-neutral">Player: {summary.player ?? "unknown"}</span>
            <span className="chip chip-active">Fight: {summary.fight_type ?? "unknown"}</span>
          </div>
        ) : null}
      </div>

      {frames.length ? (
        <div className="mt-4 grid gap-4 xl:grid-cols-2">
          {frames.map((frame) => (
            <article key={`${frame.timestamp}-${frame.image_url}`} className="surface-muted p-3">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold">{frame.timestamp.toFixed(2)}s minimap crop</h3>
                <span className="chip chip-neutral min-h-6 px-2 py-0.5">{frame.detections.length} detections</span>
              </div>
              <img className="mt-3 w-full rounded-md border border-lane bg-black object-contain dark:border-slate-800" src={frame.image_url} alt={`Minimap champion detections at ${frame.timestamp.toFixed(2)} seconds`} />
              <div className="mt-3 grid gap-2">
                {frame.white_box ? (
                  <p className="text-xs text-slate-600 dark:text-slate-400">White box anchor: x {frame.white_box.x}, y {frame.white_box.y}</p>
                ) : (
                  <p className="text-xs text-slate-500 dark:text-slate-400">White box anchor not visible in this sample.</p>
                )}
                <DetectionList detections={frame.detections} />
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className="surface-muted mt-4 p-4 text-sm text-slate-500 dark:text-slate-400">
          No minimap detection debug frames are available for this job yet. Run a new clip after this update to generate crops and YOLO detection results.
        </div>
      )}
    </section>
  );
}

function DetectionList({ detections }: { detections: DetectionDebugResult[] }) {
  if (!detections.length) {
    return <p className="text-xs text-slate-500 dark:text-slate-400">YOLO returned no champion detections for this crop.</p>;
  }
  return (
    <div className="flex flex-wrap gap-2">
      {detections.map((detection) => (
        <span
          className={`chip ${teamClass(detection.team)}`}
          key={`${detection.champion}-${detection.team}-${detection.confidence}-${detection.box.x1}-${detection.box.y1}`}
        >
          {detection.champion} - {detection.team} - {(detection.confidence * 100).toFixed(0)}%
          {detection.uncertain ? " - uncertain" : ""}
        </span>
      ))}
    </div>
  );
}

function teamClass(team: DetectionDebugResult["team"]) {
  if (team === "enemy") return "chip-danger";
  if (team === "ally") return "chip-active";
  return "chip-warning";
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
