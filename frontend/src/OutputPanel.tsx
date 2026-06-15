import type { DetectionDebug, DetectionDebugResult, JobRecord } from "./types";

export function OutputPanel({ job }: { job: JobRecord | null }) {
  const detectionDebug = safeJson<DetectionDebug>(job?.detection_debug, {});
  const outputUrl = outputUrlFromPath(job?.output_path);

  return (
    <section className="border border-lane bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold">Output</h2>
        {outputUrl ? (
          <div className="flex shrink-0 gap-2 text-sm">
            <a className="border border-lane px-3 py-1.5 font-semibold text-slate-700 dark:border-slate-700 dark:text-slate-200" href={outputUrl} target="_blank" rel="noreferrer">
              Open
            </a>
            <a className="bg-accent px-3 py-1.5 font-semibold text-white" href={outputUrl} download>
              Download
            </a>
          </div>
        ) : null}
      </div>
      {outputUrl ? (
        <video className="mt-4 aspect-[3/4] max-h-[640px] w-full bg-black object-contain" src={outputUrl} controls playsInline preload="metadata" />
      ) : (
        <div className="mt-4 flex aspect-[3/4] max-h-[640px] w-full items-center justify-center border border-lane bg-slate-50 text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-400">
          No completed clip selected
        </div>
      )}

      <ChampionDetectionPanel detectionDebug={detectionDebug} />
    </section>
  );
}

function ChampionDetectionPanel({ detectionDebug }: { detectionDebug: DetectionDebug }) {
  const frames = detectionDebug.frames ?? [];
  const summary = detectionDebug.summary;
  return (
    <section className="mt-6 border-t border-lane pt-5 dark:border-slate-800">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold">Champion Detection</h2>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600 dark:text-slate-400">
            The pipeline treats the white minimap camera box as the recording anchor. These are sampled minimap crops from the final clipped time range with YOLO champion detections overlaid.
          </p>
        </div>
        {summary ? (
          <div className="grid gap-1 text-right text-xs text-slate-600 dark:text-slate-400">
            <span>Player: <strong className="text-slate-900 dark:text-slate-100">{summary.player ?? "unknown"}</strong></span>
            <span>Fight: <strong className="text-slate-900 dark:text-slate-100">{summary.fight_type ?? "unknown"}</strong></span>
          </div>
        ) : null}
      </div>

      {frames.length ? (
        <div className="mt-4 grid gap-4 xl:grid-cols-2">
          {frames.map((frame) => (
            <article key={`${frame.timestamp}-${frame.image_url}`} className="border border-lane bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold">{frame.timestamp.toFixed(2)}s minimap crop</h3>
                <span className="text-xs text-slate-500 dark:text-slate-400">{frame.detections.length} detections</span>
              </div>
              <img className="mt-3 w-full border border-lane bg-black object-contain dark:border-slate-800" src={frame.image_url} alt={`Minimap champion detections at ${frame.timestamp.toFixed(2)} seconds`} />
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
        <div className="mt-4 border border-lane bg-slate-50 p-4 text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-400">
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
          className={`border px-2 py-1 text-xs ${teamClass(detection.team)}`}
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
  if (team === "enemy") return "border-red-300 bg-red-50 text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-200";
  if (team === "ally") return "border-blue-300 bg-blue-50 text-blue-700 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-200";
  return "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200";
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
