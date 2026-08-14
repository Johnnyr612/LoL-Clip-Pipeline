import React from "react";
import type { DetectionDebug, DetectionDebugResult, JobRecord, TikTokStatus } from "./types";

export function OutputPanel({ job }: { job: JobRecord | null }) {
  const detectionDebug = safeJson<DetectionDebug>(job?.detection_debug, {});
  const outputUrl = outputUrlFromPath(job?.output_path);
  const [tiktokStatus, setTikTokStatus] = React.useState<TikTokStatus | null>(null);
  const [mode, setMode] = React.useState<"inbox" | "direct">("inbox");
  const [description, setDescription] = React.useState("");
  const [privacyLevel, setPrivacyLevel] = React.useState("SELF_ONLY");
  const [disableComment, setDisableComment] = React.useState(false);
  const [disableDuet, setDisableDuet] = React.useState(false);
  const [disableStitch, setDisableStitch] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [message, setMessage] = React.useState("");
  const [publishState, setPublishState] = React.useState<TikTokPublishState | null>(null);

  React.useEffect(() => {
    void refreshTikTokStatus();
  }, []);

  React.useEffect(() => {
    if (!job?.tiktok_publish_id) {
      setPublishState(null);
      return;
    }
    setPublishState({
      publish_id: job.tiktok_publish_id,
      mode: job.tiktok_publish_mode ?? "inbox",
      status: job.tiktok_publish_status ?? "initialized",
      fail_reason: job.tiktok_publish_fail_reason ?? null
    });
  }, [job?.id, job?.tiktok_publish_id, job?.tiktok_publish_mode, job?.tiktok_publish_status, job?.tiktok_publish_fail_reason]);

  const publishTerminal = isTikTokTerminalStatus(publishState?.status);

  React.useEffect(() => {
    if (!publishState?.publish_id || publishState.mode !== "inbox" || publishTerminal) return;
    const publishId = publishState.publish_id;
    const controller = new AbortController();
    let stopped = false;

    async function pollPublishStatus() {
      try {
        const response = await fetch(`/tiktok/publish/${encodeURIComponent(publishId)}/status`, {
          signal: controller.signal
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || stopped) return;
        setPublishState((current) => {
          if (!current || current.publish_id !== publishId) return current;
          return {
            ...current,
            status: payload.status ?? current.status,
            fail_reason: payload.fail_reason ?? null
          };
        });
      } catch (error) {
        if (!controller.signal.aborted) {
          console.warn("TikTok status polling failed", error);
        }
      }
    }

    void pollPublishStatus();
    const interval = window.setInterval(() => void pollPublishStatus(), 10000);
    return () => {
      stopped = true;
      controller.abort();
      window.clearInterval(interval);
    };
  }, [publishState?.publish_id, publishState?.mode, publishTerminal]);

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
          title: description,
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
      setPublishState({
        publish_id: payload.publish_id,
        mode: payload.mode ?? mode,
        status: payload.status ?? "initialized",
        fail_reason: null
      });
      setMessage(mode === "inbox" ? "TikTok upload started. Waiting for inbox confirmation..." : `Direct post initialized: ${payload.publish_id}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="surface-panel p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="section-kicker">Rendered clip</p>
          <h2 className="section-title mt-1">Description</h2>
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
      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(280px,360px)]">
        <DescriptionEditor description={description} onChange={setDescription} />
        <PhonePostPreview
          description={description}
          detectionDebug={detectionDebug}
          outputUrl={outputUrl}
        />
      </div>

      <MediaProfilePanel detectionDebug={detectionDebug} />
      <CropPlanPanel detectionDebug={detectionDebug} />

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
        publishState={publishState}
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
        description={description}
      />

      <ChampionDetectionPanel detectionDebug={detectionDebug} />
    </section>
  );
}

type TikTokPublishState = {
  publish_id: string;
  mode: string;
  status: string;
  fail_reason?: string | null;
};

function isTikTokTerminalStatus(status: string | null | undefined) {
  return status === "SEND_TO_USER_INBOX" || status === "FAILED";
}

function DescriptionEditor({
  description,
  onChange
}: {
  description: string;
  onChange: (value: string) => void;
}) {
  return (
    <section className="surface-muted min-h-[420px] p-4">
      <label className="field-label h-full">
        Description
        <textarea
          className="textarea-field min-h-[280px] flex-1 resize-y"
          maxLength={2200}
          onChange={(event) => onChange(event.target.value)}
          placeholder="League of Legends highlight"
          value={description}
        />
        <span className="text-xs font-normal text-slate-500 dark:text-slate-400">
          {description.length}/2200
        </span>
      </label>
    </section>
  );
}

function PhonePostPreview({
  description,
  detectionDebug,
  outputUrl
}: {
  description: string;
  detectionDebug: DetectionDebug;
  outputUrl: string;
}) {
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const [currentTime, setCurrentTime] = React.useState(0);
  const [duration, setDuration] = React.useState(0);
  const [isPlaying, setIsPlaying] = React.useState(false);
  const player = detectionDebug.summary?.player;
  const username = player ? `@${slugifyHandle(player)}` : "@lolclip";
  const captionText = description.trim() || "League of Legends highlight";

  function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  }

  function seekTo(value: string) {
    const video = videoRef.current;
    if (!video) return;
    const nextTime = Number(value);
    video.currentTime = nextTime;
    setCurrentTime(nextTime);
  }

  return (
    <div className="flex justify-center xl:justify-end">
      <div className="phone-shell" aria-label="iPhone TikTok-style preview">
        <div className="phone-screen">
          {outputUrl ? (
            <video
              ref={videoRef}
              key={`phone-${outputUrl}`}
              className="phone-video"
              src={outputUrl}
              controls
              onLoadedMetadata={(event) => setDuration(event.currentTarget.duration || 0)}
              onPause={() => setIsPlaying(false)}
              onPlay={() => setIsPlaying(true)}
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
              playsInline
              preload="metadata"
            />
          ) : (
            <div className="phone-empty">No output</div>
          )}
          <div className="phone-status">
            <span>9:41</span>
            <span className="phone-status-icons">5G 100%</span>
          </div>
          <div className="phone-tabs">
            <span>Following</span>
            <strong>For You</strong>
          </div>
          {outputUrl ? (
            <div className="phone-control-bar">
              <button className="phone-play-button" onClick={togglePlayback} type="button">
                {isPlaying ? "Pause" : "Play"}
              </button>
              <input
                aria-label="Preview timeline"
                max={duration || 0}
                min="0"
                onChange={(event) => seekTo(event.target.value)}
                step="0.01"
                type="range"
                value={Math.min(currentTime, duration || currentTime)}
              />
              <span>{formatVideoTime(currentTime)}</span>
            </div>
          ) : null}
          <div className="phone-action-rail" aria-hidden="true">
            <span className="phone-avatar">{player?.slice(0, 1).toUpperCase() ?? "L"}</span>
            <span className="phone-action">♥</span>
            <span className="phone-action">⌕</span>
            <span className="phone-action">↗</span>
            <span className="phone-disc">♪</span>
          </div>
          <div className="phone-caption">
            <strong>{username}</strong>
            <span>{captionText}</span>
            <small>Original sound - {username}</small>
          </div>
          <div className="phone-bottom-nav" aria-hidden="true">
            <span>⌂</span>
            <span>⊕</span>
            <span>▣</span>
            <span>♡</span>
            <span>◉</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function MediaProfilePanel({ detectionDebug }: { detectionDebug: DetectionDebug }) {
  const input = detectionDebug.media_profile?.input;
  const output = detectionDebug.media_profile?.encode_settings;
  if (!input && !output) return null;

  return (
    <section className="divider mt-6 pt-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="section-kicker">Media profile</p>
          <h2 className="section-title mt-1">Input & Output Match</h2>
        </div>
        {output?.match_source_encoding ? <span className="chip chip-active">Source matching on</span> : null}
      </div>
      <div className="mt-4 grid gap-3 text-sm md:grid-cols-2">
        <div className="surface-muted p-3">
          <h3 className="text-sm font-semibold">Input MP4</h3>
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            <span className="chip chip-neutral">{input?.video_codec?.toUpperCase() ?? "unknown codec"}</span>
            <span className="chip chip-neutral">{input?.width ?? "?"}x{input?.height ?? "?"}</span>
            <span className="chip chip-neutral">{formatFps(input?.fps)} FPS</span>
            <span className="chip chip-neutral">{formatBitrate(input?.video_bitrate ?? input?.total_bitrate)}</span>
            <span className="chip chip-neutral">Audio: {input?.has_audio ? (input.audio_codec?.toUpperCase() ?? "yes") : "none"}</span>
          </div>
        </div>
        <div className="surface-muted p-3">
          <h3 className="text-sm font-semibold">Output Encode</h3>
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            <span className="chip chip-active">{output?.encoder ?? "encoder"}</span>
            <span className="chip chip-neutral">{output?.output_width ?? "?"}x{output?.output_height ?? "?"}</span>
            <span className="chip chip-neutral">{output?.fps ?? "?"} FPS</span>
            <span className="chip chip-neutral">{output?.target_video_bitrate ?? (output?.crf ? `CRF ${output.crf}` : "auto bitrate")}</span>
            <span className="chip chip-neutral">Preset: {output?.preset ?? "default"}</span>
            {output?.source_bitrate_multiplier ? <span className="chip chip-neutral">Video: {output.source_bitrate_multiplier.toFixed(2)}x source</span> : null}
            <span className="chip chip-neutral">Audio: {output?.audio_codec === "copy" ? "copy" : (output?.audio_bitrate ?? "AAC")}</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function CropPlanPanel({ detectionDebug }: { detectionDebug: DetectionDebug }) {
  const settings = detectionDebug.crop_settings;
  const debug = detectionDebug.crop_debug;
  const trim = detectionDebug.trim?.final;
  if (!settings && !debug && !trim) return null;

  const mode = titleCase(debug?.mode ?? settings?.mode ?? "unknown");
  const transition = debug?.transition ?? settings?.transition ?? "";
  const transitionLabel = transition === "pan" ? "Smooth" : transition === "cut" ? "Jump" : titleCase(transition || "unknown");
  const movement = debug?.movement_px;
  const threatSignal = debug?.threat_signal;
  const samples = debug?.sample_keyframes?.length ? debug.sample_keyframes : debug?.sample_frames ?? [];

  return (
    <section className="divider mt-6 pt-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="section-kicker">Crop plan</p>
          <h2 className="section-title mt-1">View Settings</h2>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="chip chip-active">{mode}</span>
          <span className="chip chip-neutral">{transitionLabel}</span>
        </div>
      </div>

      <div className="mt-4 grid gap-3 text-sm md:grid-cols-2">
        <div className="surface-muted p-3">
          <h3 className="text-sm font-semibold">Selected View</h3>
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            <span className="chip chip-active">Crop: {mode}</span>
            <span className="chip chip-neutral">Motion: {transitionLabel}</span>
            {trim?.duration ? <span className="chip chip-neutral">Clip: {trim.duration.toFixed(1)}s</span> : null}
          </div>
        </div>
        <div className="surface-muted p-3">
          <h3 className="text-sm font-semibold">Camera Path</h3>
          {debug ? (
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              <span className={movement ? "chip chip-active" : "chip chip-warning"}>Move: {movement ?? 0}px</span>
              <span className="chip chip-neutral">x {debug.x_min ?? "?"} to {debug.x_max ?? "?"}</span>
              <span className="chip chip-neutral">{debug.keyframe_count ?? 0} keyframes</span>
              <span className="chip chip-neutral">{debug.position_changes ?? 0} changes</span>
            </div>
          ) : (
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">Run a new job after this update to record crop movement stats.</p>
          )}
        </div>
      </div>

      {debug?.movement_px === 0 ? (
        <p className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-800 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-200">
          This output resolved to a fixed crop path, so Jump and Smooth will look identical for this clip.
        </p>
      ) : null}

      {threatSignal ? (
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          <span className="chip chip-neutral" title="Frames where a thick, stable red health-bar sample steered the crop.">
            Healthbar samples: {threatSignal.healthbar_samples ?? 0}
          </span>
        </div>
      ) : null}

      {samples.length ? (
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          {samples.map((sample, index) => (
            <span className="chip chip-neutral" key={`${sample.time}-${sample.x}-${index}`}>
              {sample.time === null ? "?" : `${sample.time.toFixed(1)}s`}: x {sample.x}
            </span>
          ))}
        </div>
      ) : null}
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
  publishState,
  scope,
  description
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
  publishState: TikTokPublishState | null;
  scope: string;
  description: string;
}) {
  const requiredScope = mode === "direct" ? "video.publish" : "video.upload";
  const hasRequiredScope = scope.split(",").map((item) => item.trim()).includes(requiredScope);
  const statusText = tikTokPublishStatusText(publishState, busy);
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
            <p className="surface-muted p-3 text-sm text-slate-600 dark:text-slate-400">
              Direct Post uses the Description text above as the TikTok title.
              {description.trim() ? "" : " Add a description before publishing if you want a caption."}
            </p>
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
          <div className="grid gap-3">
            <p className="surface-muted p-3 text-slate-600 dark:text-slate-400">Inbox upload sends the clip to TikTok so the creator can finish editing and posting there.</p>
            {statusText ? (
              <div className={publishState?.status === "FAILED" ? "rounded-md border border-danger bg-red-50 p-3 text-sm text-danger dark:bg-red-950/30" : "surface-muted p-3 text-sm text-slate-700 dark:text-slate-300"}>
                <p className="font-semibold">{statusText.title}</p>
                {statusText.body ? <p className="mt-1 leading-6">{statusText.body}</p> : null}
              </div>
            ) : null}
            {publishState?.status === "SEND_TO_USER_INBOX" ? (
              <div className="rounded-md border border-emerald-300 bg-emerald-50 p-4 text-sm leading-6 text-emerald-900 dark:border-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-100">
                <p className="font-semibold">In your TikTok inbox</p>
                <p className="mt-1">
                  Draft sent to @aaaplay44&apos;s TikTok inbox. Open the TikTok app - tap the inbox notification - finish the post and set visibility to Public.
                </p>
              </div>
            ) : null}
          </div>
        )}
      </div>

      {message ? <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">{message}</p> : null}
    </section>
  );
}

function tikTokPublishStatusText(state: TikTokPublishState | null, busy: boolean) {
  if (busy) {
    return { title: "Uploading to TikTok", body: "Sending the finished MP4 to the TikTok inbox upload endpoint." };
  }
  if (!state || state.mode !== "inbox") return null;
  if (state.status === "SEND_TO_USER_INBOX") {
    return { title: "In your TikTok inbox", body: "" };
  }
  if (state.status === "FAILED") {
    return { title: `Failed: ${state.fail_reason || "TikTok did not provide a reason"}`, body: "Fix the issue and send the clip again." };
  }
  const label = state.status === "initialized" ? "Upload accepted" : titleCase(state.status.toLowerCase().replace(/_/g, " "));
  return { title: "Processing on TikTok", body: `${label}. Waiting for TikTok to finish sending the draft to the account inbox.` };
}

function ChampionDetectionPanel({ detectionDebug }: { detectionDebug: DetectionDebug }) {
  const frames = detectionDebug.frames ?? [];
  const summary = detectionDebug.summary;
  const minimapSkipped = Boolean(detectionDebug.processing_settings?.skip_minimap_detection);
  return (
    <section className="divider mt-6 pt-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="section-kicker">Debug review</p>
          <h2 className="section-title mt-1">Champion Detection</h2>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600 dark:text-slate-400">
            {minimapSkipped
              ? "Minimap champion detection was skipped for this job. The summary uses main-frame HUD, health-bar, and optional local vision signals only."
              : "These are sampled minimap crops from the final clipped time range with YOLO champion detections overlaid."}
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
                <DetectionList detections={frame.detections} />
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className="surface-muted mt-4 p-4 text-sm text-slate-500 dark:text-slate-400">
          {minimapSkipped
            ? "No minimap debug frames were generated because minimap detection was skipped."
            : "No minimap detection debug frames are available for this job yet. Run a new clip after this update to generate crops and YOLO detection results."}
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

function formatBitrate(value: number | null | undefined) {
  if (!value || value <= 0) return "unknown bitrate";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value % 1_000_000 === 0 ? 0 : 1)} Mbps`;
  return `${Math.round(value / 1000)} kbps`;
}

function formatFps(value: number | null | undefined) {
  if (!value || value <= 0) return "?";
  const rounded = Math.round(value);
  return Math.abs(value - rounded) < 0.05 ? String(rounded) : value.toFixed(2);
}

function formatVideoTime(value: number) {
  if (!Number.isFinite(value) || value < 0) return "0:00";
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function titleCase(value: string) {
  if (!value) return "";
  return value.slice(0, 1).toUpperCase() + value.slice(1);
}

function slugifyHandle(value: string) {
  const handle = value.toLowerCase().replace(/[^a-z0-9]+/g, "").slice(0, 20);
  return handle || "lolclip";
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
