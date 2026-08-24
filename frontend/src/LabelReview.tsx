import React from "react";

type LabelPayload = {
  summary: LabelSummary;
  raw_file_inventory?: RawFileInventory;
  records: LabelRecord[];
  unmatched_edits: string[];
  duplicate_raw_stems: string[];
};

type LabelSummary = {
  total: number;
  approved: number;
  needs_review: number;
  high_confidence: number;
  unmatched: number;
  skipped: number;
};

type RawFileInventory = {
  summary: {
    total: number;
    used_for_training: number;
    approved_for_training?: number;
    new_holdout_candidates: number;
    review_queue: number;
    skipped: number;
    posted_to_tiktok?: number;
    sent_to_inbox?: number;
    missing_dirs: number;
  };
  files: RawTrainingFile[];
  missing_dirs: string[];
  duplicate_stems: string[];
};

type RawTrainingFile = {
  filename: string;
  path: string;
  source_dir: string;
  size: number;
  modified_at: string;
  status:
    | "used_for_training"
    | "approved_for_training"
    | "new_holdout_candidate"
    | "review_queue"
    | "skipped"
    | "posted_to_tiktok"
    | "sent_to_inbox";
  used_for_training: boolean;
  approved_for_training?: boolean;
  record_index: number | null;
};

type LabelRecord = {
  filename: string;
  raw_path: string;
  edit_path: string;
  edit_filename: string;
  raw_duration: number;
  edit_duration: number;
  clip_start: number;
  clip_end: number;
  fight_start: number;
  fight_end: number;
  fight_segments?: number[][];
  segments?: {
    fight?: number[] | number[][];
    bridge?: number[][];
    pre_fight_context?: number[];
    post_fight_context?: number[];
  };
  match: {
    method: string;
    score: number | null;
    confidence: string;
  };
  needs_review: boolean;
  review_note: string;
  skipped?: boolean;
  review_status?: string;
  detector_flags?: string[];
  raw_exists?: boolean;
  edit_exists?: boolean;
};

type AddRawFileResult = {
  record_index: number;
  payload: LabelPayload;
};

type DeleteRecordResult = {
  deleted_index: number;
  payload: LabelPayload;
};

type FightSegmentField = {
  start: string;
  end: string;
};

type EditableFields = {
  clip_start: string;
  clip_end: string;
  fight_segments: FightSegmentField[];
  review_note: string;
};

function fmt(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) return "0.000";
  return value.toFixed(3);
}

function formatFileSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatModified(value: string) {
  const date = Date.parse(value);
  if (Number.isNaN(date)) return "";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(date);
}

function formatSegments(segments: Array<{ start: number; end: number }>) {
  if (segments.length === 0) return "none";
  return segments.map((segment, index) => (
    `Fight ${index + 1}: ${fmt(segment.start)}s-${fmt(segment.end)}s`
  )).join(" | ");
}

function videoUrl(path: string) {
  return `/training/video?path=${encodeURIComponent(path)}`;
}


function normalizeFightSegments(record: LabelRecord): number[][] {
  const raw = record.fight_segments ?? record.segments?.fight;
  if (Array.isArray(raw) && raw.length > 0) {
    if (Array.isArray(raw[0])) {
      return (raw as number[][])
        .filter((item) => item.length >= 2)
        .map((item) => [Number(item[0]), Number(item[1])])
        .filter(([start, end]) => Number.isFinite(start) && Number.isFinite(end) && end > start);
    }
    if (raw.length >= 2) {
      const single = raw as number[];
      return [[Number(single[0]), Number(single[1])]];
    }
  }
  return [[record.fight_start, record.fight_end]];
}

function segmentFieldsFromRecord(record: LabelRecord): FightSegmentField[] {
  return normalizeFightSegments(record).map(([start, end]) => ({
    start: fmt(start),
    end: fmt(end)
  }));
}
function fieldsFromRecord(record: LabelRecord): EditableFields {
  return {
    clip_start: fmt(record.clip_start),
    clip_end: fmt(record.clip_end),
    fight_segments: segmentFieldsFromRecord(record),
    review_note: record.review_note ?? ""
  };
}

function numberFromField(value: string) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}


type ReviewStatus = "approved" | "check" | "review" | "skip";

function reviewStatus(record: LabelRecord): ReviewStatus {
  if (record.skipped || record.review_status === "skip") return "skip";
  if (!record.needs_review) return "approved";
  if (record.match.confidence !== "high") return "check";
  return "review";
}

function statusLabel(status: ReviewStatus) {
  if (status === "approved") return "Train ready";
  if (status === "check") return "Check";
  if (status === "skip") return "Skip";
  return "Review";
}

function statusClass(status: ReviewStatus, active: boolean) {
  const activeClass = active ? " ring-2 ring-accent" : "";
  const base = "rounded-md transition hover:shadow-sm";
  if (status === "approved") return `${base} border-emerald-500 bg-emerald-50 text-emerald-950 dark:bg-emerald-950/30 dark:text-emerald-100${activeClass}`;
  if (status === "check") return `${base} border-amber-500 bg-amber-50 text-amber-950 dark:bg-amber-950/30 dark:text-amber-100${activeClass}`;
  if (status === "skip") return `${base} border-slate-400 bg-slate-200 text-slate-700 opacity-80 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-300${activeClass}`;
  return `${base} border-sky-500 bg-sky-50 text-sky-950 dark:bg-sky-950/30 dark:text-sky-100${activeClass}`;
}

function rawFileStatusLabel(status: RawTrainingFile["status"]) {
  if (status === "used_for_training" || status === "approved_for_training") return "Approved for training";
  if (status === "posted_to_tiktok") return "Posted to TikTok";
  if (status === "sent_to_inbox") return "Sent to inbox";
  if (status === "review_queue") return "Model review";
  if (status === "skipped") return "Skipped";
  return "New raw clip";
}

function rawFileStatusClass(status: RawTrainingFile["status"]) {
  if (status === "used_for_training" || status === "approved_for_training") return "chip chip-success";
  if (status === "posted_to_tiktok" || status === "sent_to_inbox") return "chip chip-success";
  if (status === "review_queue") return "chip chip-warning";
  if (status === "skipped") return "chip chip-neutral";
  return "chip chip-active";
}

function visiblePositionFor(payload: LabelPayload, recordIndex: number, needsReviewOnly: boolean) {
  const indexes = payload.records
    .map((item, index) => ({ item, index }))
    .filter(({ item }) => !needsReviewOnly || (item.needs_review && !item.skipped))
    .map(({ index }) => index);
  return Math.max(0, indexes.indexOf(recordIndex));
}

function clampPct(value: number) {
  return Math.max(0, Math.min(100, value));
}

export function LabelReview() {
  const [payload, setPayload] = React.useState<LabelPayload | null>(null);
  const [visiblePosition, setVisiblePosition] = React.useState(0);
  const [showNeedsReviewOnly, setShowNeedsReviewOnly] = React.useState(true);
  const [rawFilesOpen, setRawFilesOpen] = React.useState(false);
  const [fields, setFields] = React.useState<EditableFields | null>(null);
  const [status, setStatus] = React.useState("");
  const [error, setError] = React.useState("");
  const [rawActionPath, setRawActionPath] = React.useState("");
  const [detectingFight, setDetectingFight] = React.useState(false);
  const [deletingRecord, setDeletingRecord] = React.useState(false);
  const rawVideoRef = React.useRef<HTMLVideoElement | null>(null);

  React.useEffect(() => {
    void load();
  }, []);

  const visibleIndexes = React.useMemo(() => {
    if (!payload) return [];
    return payload.records
      .map((record, index) => ({ record, index }))
      .filter(({ record }) => !showNeedsReviewOnly || (record.needs_review && !record.skipped))
      .map(({ index }) => index);
  }, [payload, showNeedsReviewOnly]);

  const activePosition = Math.max(0, Math.min(visiblePosition, Math.max(0, visibleIndexes.length - 1)));
  const activeIndex = visibleIndexes[activePosition] ?? 0;
  const record = payload?.records[activeIndex] ?? null;

  React.useEffect(() => {
    if (record) setFields(fieldsFromRecord(record));
  }, [record]);

  async function load() {
    setError("");
    const response = await fetch("/training/label-review");
    const next = await response.json().catch(() => null);
    if (!response.ok || !next) {
      setError(next?.detail ?? "Unable to load label review data");
      return;
    }
    setPayload(next);
    setVisiblePosition(0);
  }

  async function refreshRawFiles() {
    setStatus("Refreshing private raw file inventory...");
    setError("");
    const response = await fetch("/training/label-review/refresh-files", {
      method: "POST"
    });
    const next = await response.json().catch(() => null);
    if (!response.ok || !next) {
      setError(next?.detail ?? "Unable to refresh raw files");
      setStatus("");
      return;
    }
    setPayload(next);
    setStatus("Raw file inventory refreshed.");
  }

  async function addRawFileToReview(path: string) {
    setRawActionPath(path);
    setStatus("Finding a postable clip with current VideoMAE weights...");
    setError("");
    const response = await fetch("/training/label-review/raw-files", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path })
    });
    const result = await response.json().catch(() => null) as AddRawFileResult | null;
    setRawActionPath("");
    if (!response.ok || !result?.payload) {
      setError((result as any)?.detail ?? "Unable to run model review for this raw clip");
      setStatus("");
      return;
    }
    setPayload(result.payload);
    setShowNeedsReviewOnly(true);
    setVisiblePosition(visiblePositionFor(result.payload, result.record_index, true));
    setStatus("Added raw clip with VideoMAE annotations for review.");
  }

  function updateField(key: "clip_start" | "clip_end" | "review_note", value: string) {
    setFields((current) => (current ? { ...current, [key]: value } : current));
  }

  function updateFightSegment(index: number, key: keyof FightSegmentField, value: string) {
    setFields((current) => {
      if (!current) return current;
      const fightSegments = current.fight_segments.map((item, itemIndex) => (
        itemIndex === index ? { ...item, [key]: value } : item
      ));
      return { ...current, fight_segments: fightSegments };
    });
  }

  function addFightSegment() {
    setFields((current) => {
      if (!current) return current;
      const last = current.fight_segments[current.fight_segments.length - 1];
      const start = last ? last.end : current.clip_start;
      const end = fmt(Math.min(numberFromField(start) + 3, numberFromField(current.clip_end)));
      return { ...current, fight_segments: [...current.fight_segments, { start, end }] };
    });
  }

  function removeFightSegment(index: number) {
    setFields((current) => {
      if (!current || current.fight_segments.length <= 1) return current;
      return { ...current, fight_segments: current.fight_segments.filter((_, itemIndex) => itemIndex !== index) };
    });
  }

  function move(delta: number) {
    setVisiblePosition((current) => Math.max(0, Math.min(visibleIndexes.length - 1, current + delta)));
  }
  function selectRecord(index: number) {
    const nextPosition = visibleIndexes.indexOf(index);
    if (nextPosition >= 0) {
      setVisiblePosition(nextPosition);
      return;
    }
    setShowNeedsReviewOnly(false);
    setVisiblePosition(index);
  }

  function toggleNeedsReviewOnly(checked: boolean) {
    if (!payload) {
      setShowNeedsReviewOnly(checked);
      setVisiblePosition(0);
      return;
    }
    const nextVisibleIndexes = payload.records
      .map((item, index) => ({ item, index }))
      .filter(({ item }) => !checked || (item.needs_review && !item.skipped))
      .map(({ index }) => index);
    const currentRecordPosition = nextVisibleIndexes.indexOf(activeIndex);
    setShowNeedsReviewOnly(checked);
    setVisiblePosition(currentRecordPosition >= 0 ? currentRecordPosition : 0);
  }

  function previewRawVideo(second: number, preroll = 0) {
    const video = rawVideoRef.current;
    if (!video) return;
    const target = Math.max(0, second - preroll);
    const safeTarget = Number.isFinite(video.duration) ? Math.min(target, Math.max(0, video.duration - 0.1)) : target;
    video.pause();
    const playAfterSeek = () => {
      void video.play();
    };
    video.addEventListener("seeked", playAfterSeek, { once: true });
    video.currentTime = safeTarget;
    window.setTimeout(() => {
      if (video.paused) void video.play();
    }, 150);
  }

  async function skipRecord() {
    if (!record) return;
    setStatus("Skipping label...");
    setError("");
    const response = await fetch(`/training/label-review/records/${activeIndex}/skip`, {
      method: "POST"
    });
    const result = await response.json().catch(() => null);
    if (!response.ok || !result) {
      setError(result?.detail ?? "Unable to skip label");
      setStatus("");
      return;
    }
    setPayload((current) => {
      if (!current) return current;
      const records = [...current.records];
      records[activeIndex] = result.record;
      return { ...current, records, summary: result.summary };
    });
    setStatus("Skipped label and regenerated trainer labels without it.");
    setVisiblePosition((current) => {
      const nextLength = showNeedsReviewOnly ? Math.max(visibleIndexes.length - 1, 0) : visibleIndexes.length;
      return Math.max(0, Math.min(nextLength - 1, current));
    });
  }

  async function deleteRecord() {
    if (!record) return;
    const firstConfirm = window.confirm(
      `Delete this label entry?\n\n${record.filename}\n\nThis removes it from Label Review and regenerates trainer labels. The raw MP4 file is not deleted.`
    );
    if (!firstConfirm) return;
    const secondConfirm = window.confirm(
      "Are you sure? This permanently removes the whole annotation record from fight_label_candidates.json and from regenerated videomae_labels.json."
    );
    if (!secondConfirm) return;

    setDeletingRecord(true);
    setStatus("Deleting label entry and regenerating trainer labels...");
    setError("");
    const response = await fetch(`/training/label-review/records/${activeIndex}`, {
      method: "DELETE"
    });
    const result = await response.json().catch(() => null) as DeleteRecordResult | null;
    setDeletingRecord(false);
    if (!response.ok || !result?.payload) {
      setError((result as any)?.detail ?? "Unable to delete label entry");
      setStatus("");
      return;
    }
    setPayload(result.payload);
    setFields(null);
    setVisiblePosition((current) => {
      const nextVisibleIndexes = result.payload.records
        .map((item, index) => ({ item, index }))
        .filter(({ item }) => !showNeedsReviewOnly || (item.needs_review && !item.skipped))
        .map(({ index }) => index);
      return Math.max(0, Math.min(current, Math.max(0, nextVisibleIndexes.length - 1)));
    });
    setStatus("Deleted label entry and regenerated trainer labels.");
  }

  async function save(approved: boolean) {
    if (!fields || !record) return;
    setStatus(approved ? "Approving label..." : "Saving label...");
    setError("");
    const response = await fetch(`/training/label-review/records/${activeIndex}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clip_start: numberFromField(fields.clip_start),
        clip_end: numberFromField(fields.clip_end),
        fight_start: numberFromField(fields.fight_segments[0]?.start ?? fields.clip_start),
        fight_end: numberFromField(fields.fight_segments[fields.fight_segments.length - 1]?.end ?? fields.clip_end),
        fight_segments: fields.fight_segments.map((item) => [
          numberFromField(item.start),
          numberFromField(item.end)
        ]),
        approved,
        review_note: fields.review_note
      })
    });
    const result = await response.json().catch(() => null);
    if (!response.ok || !result) {
      setError(result?.detail ?? "Unable to save label");
      setStatus("");
      return;
    }
    setPayload((current) => {
      if (!current) return current;
      const records = [...current.records];
      records[activeIndex] = result.record;
      return { ...current, records, summary: result.summary };
    });
    setStatus(approved ? "Approved for the next training run and trainer labels regenerated." : "Saved and trainer labels regenerated.");
    if (approved) {
      setVisiblePosition((current) => {
        const nextLength = showNeedsReviewOnly ? Math.max(visibleIndexes.length - 1, 0) : visibleIndexes.length;
        const nextPosition = showNeedsReviewOnly ? current : current + 1;
        return Math.max(0, Math.min(nextLength - 1, nextPosition));
      });
    }
  }

  if (error && !payload) {
    return (
      <section className="surface-panel border-danger p-4 text-sm text-danger">
        {error}
      </section>
    );
  }

  async function detectFightClip() {
    if (!record) return;
    setDetectingFight(true);
    setStatus("Finding postable clip with current VideoMAE weights...");
    setError("");
    const response = await fetch(`/training/label-review/records/${activeIndex}/detect-fight`, {
      method: "POST"
    });
    const result = await response.json().catch(() => null);
    setDetectingFight(false);
    if (!response.ok || !result) {
      setError(result?.detail ?? "Unable to find clip");
      setStatus("");
      return;
    }
    setPayload((current) => {
      if (!current) return current;
      const records = [...current.records];
      records[activeIndex] = result.record;
      return { ...current, records, summary: result.summary };
    });
    setFields(fieldsFromRecord(result.record));
    const detectedSegments = normalizeFightSegments(result.record).map(([start, end]) => ({ start, end }));
    setStatus(
      `Found clip ${fmt(result.record.clip_start)}s-${fmt(result.record.clip_end)}s. ${formatSegments(detectedSegments)}. Adjust before approving.`
    );
    window.setTimeout(() => previewRawVideo(result.record.clip_start), 100);
  }

  if (!payload || !record || !fields) {
    return (
      <section className="surface-panel p-4 text-sm text-slate-600 dark:text-slate-400">
        Loading label review data...
      </section>
    );
  }

  const duration = Math.max(0.001, record.raw_duration);
  const clipLeft = clampPct((numberFromField(fields.clip_start) / duration) * 100);
  const clipWidth = clampPct(((numberFromField(fields.clip_end) - numberFromField(fields.clip_start)) / duration) * 100);
  const fightSegmentValues = fields.fight_segments.map((item) => ({
    start: numberFromField(item.start),
    end: numberFromField(item.end)
  }));
  const currentClipStart = numberFromField(fields.clip_start);
  const currentClipEnd = numberFromField(fields.clip_end);
  const currentClipDuration = Math.max(0, currentClipEnd - currentClipStart);
  const firstFightStart = fightSegmentValues[0]?.start ?? numberFromField(fields.clip_start);
  const lastFightEnd = fightSegmentValues[fightSegmentValues.length - 1]?.end ?? numberFromField(fields.clip_end);
  const rawInventory = payload.raw_file_inventory;
  const newRawFiles = rawInventory?.summary.new_holdout_candidates ?? 0;
  const approvedForTraining = rawInventory?.summary.approved_for_training ?? rawInventory?.summary.used_for_training ?? 0;
  const hasEditedReference = Boolean(record.edit_path);
  const hasRawVideo = record.raw_exists !== false;
  const rawMissingMessage = "Raw source file not found. Restore this MP4 at the stored path or add the current copy from Raw Video Inventory.";

  return (
    <section className="grid gap-4">
      <div className="surface-panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="section-kicker">Model training data</p>
            <h2 className="section-title mt-1">Training Label Review</h2>
            <div className="mt-3 flex flex-wrap gap-2">
              <span className="chip chip-success">{payload.summary.approved} approved for training</span>
              <span className="chip chip-neutral">{payload.summary.total} total</span>
              <span className="chip chip-warning">{payload.summary.needs_review} need review</span>
              <span className="chip chip-neutral">{payload.summary.skipped ?? 0} skipped</span>
              {rawInventory ? <span className="chip chip-active">{newRawFiles} new raw</span> : null}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button className="button" onClick={() => void refreshRawFiles()} type="button">
              Refresh Raw Files
            </button>
            <label className="flex items-center gap-2 rounded-md border border-lane bg-slate-50 px-3 py-2 text-sm font-medium dark:border-slate-800 dark:bg-slate-950">
              <input
                type="checkbox"
                checked={showNeedsReviewOnly}
                onChange={(event) => toggleNeedsReviewOnly(event.target.checked)}
              />
              Needs review only
            </label>
          </div>
        </div>
      </div>

      {rawInventory ? (
        <div className="surface-panel p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="section-kicker">Private local files</p>
              <h3 className="text-sm font-semibold">Raw Video Inventory</h3>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="chip chip-active">{rawInventory.summary.new_holdout_candidates} new raw</span>
              <span className="chip chip-success">{approvedForTraining} approved for training</span>
              <span className="chip chip-warning">{rawInventory.summary.review_queue} model review</span>
              <span className="chip chip-neutral">{rawInventory.summary.total} scanned</span>
              <button className="button min-h-8 px-2 py-1 text-xs" onClick={() => setRawFilesOpen((value) => !value)} type="button">
                {rawFilesOpen ? "Collapse" : "Expand"}
              </button>
            </div>
          </div>
          {rawFilesOpen ? (
            <div>
              {rawInventory.missing_dirs.length > 0 ? (
                <p className="mb-3 rounded-md border border-danger bg-red-50 p-3 text-sm text-danger dark:bg-red-950/30">
                  {rawInventory.missing_dirs.length} configured raw folder could not be found.
                </p>
              ) : null}
              <div className="max-h-56 overflow-y-auto pr-1">
                <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                  {rawInventory.files.map((item) => (
                    <div
                      key={item.path}
                      className="rounded-md border border-lane bg-slate-50 p-2 text-xs transition hover:shadow-sm dark:border-slate-800 dark:bg-slate-950"
                    >
                      <div className="mb-1 flex items-center justify-between gap-2">
                        <span className={rawFileStatusClass(item.status)}>{rawFileStatusLabel(item.status)}</span>
                        <span className="shrink-0 text-slate-500 dark:text-slate-400">{formatFileSize(item.size)}</span>
                      </div>
                      <p className="truncate font-semibold" title={item.filename}>{item.filename}</p>
                      <p className="mt-1 truncate text-slate-500 dark:text-slate-400" title={item.path}>
                        {formatModified(item.modified_at)} | {item.path}
                      </p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {item.status === "new_holdout_candidate" ? (
                          <button
                            className="button-primary min-h-8 px-2 py-1 text-xs"
                            disabled={rawActionPath === item.path}
                            onClick={() => void addRawFileToReview(item.path)}
                            type="button"
                          >
                            {rawActionPath === item.path ? "Finding..." : "Find Clip"}
                          </button>
                        ) : null}
                        {typeof item.record_index === "number" ? (
                          <button
                            className="button min-h-8 px-2 py-1 text-xs"
                            onClick={() => selectRecord(item.record_index as number)}
                            type="button"
                          >
                            Open
                          </button>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-[280px_minmax(0,1fr)_360px]">
        <aside className="surface-panel p-3">
          <div className="mb-3 flex items-center justify-between gap-2">
            <h3 className="text-sm font-semibold">Review Queue</h3>
            <span className="chip chip-neutral min-h-6 px-2 py-0.5">{payload.records.length}</span>
          </div>
          <div className="max-h-[70vh] overflow-y-auto pr-1">
            <div className="grid gap-2">
              {payload.records.map((item, index) => {
                const state = reviewStatus(item);
                const active = index === activeIndex;
                return (
                  <button
                    key={`${index}:${item.edit_filename}`}
                    className={`border p-2 text-left text-xs ${statusClass(state, active)}`}
                    onClick={() => selectRecord(index)}
                    type="button"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold">#{index + 1}</span>
                      <span className="shrink-0 text-[11px] font-semibold uppercase tracking-normal">{statusLabel(state)}</span>
                    </div>
                    <p className="mt-1 truncate font-medium" title={item.filename}>{item.filename}</p>
                    <p className="mt-0.5 truncate opacity-80" title={item.edit_filename || "Model-generated from raw clip"}>
                      {item.edit_filename || "raw model annotation"}
                    </p>
                    <p className="mt-1 opacity-80">{item.match.confidence} | {fmt(item.clip_start)}s</p>
                  </button>
                );
              })}
            </div>
          </div>
        </aside>
        <div className="grid gap-4">
          <div className="surface-panel p-3">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3 text-sm">
              <div className="min-w-0">
                <p className="break-all font-semibold">{record.filename}</p>
                <p className="break-all text-xs text-slate-500 dark:text-slate-400">
                  Edit: {record.edit_filename || "none - model-generated from raw clip"}
                </p>
                <p className="break-all text-xs text-slate-500 dark:text-slate-400">Raw: {record.raw_path}</p>
              </div>
              <span className="chip chip-neutral shrink-0">
                {activePosition + 1} / {visibleIndexes.length} | record {activeIndex + 1}
              </span>
            </div>
            {hasRawVideo ? (
              <video
                key={`${activeIndex}:${record.raw_path}`}
                ref={rawVideoRef}
                className="aspect-video w-full rounded-md bg-black"
                controls
                preload="metadata"
                src={videoUrl(record.raw_path)}
              />
            ) : (
              <div className="grid aspect-video w-full place-items-center rounded-md border border-danger bg-red-50 p-4 text-center text-sm text-danger dark:bg-red-950/30">
                {rawMissingMessage}
              </div>
            )}
            <div className="mt-3 grid gap-2">
              <div className="relative h-9 overflow-hidden rounded-md bg-slate-200 dark:bg-slate-800">
                <div className="absolute top-0 h-9 bg-sky-400/50" style={{ left: `${clipLeft}%`, width: `${clipWidth}%` }} />
                {fightSegmentValues.map((segment, index) => (
                  <div
                    key={`${index}:${segment.start}:${segment.end}`}
                    className="absolute top-1 h-7 rounded-sm bg-red-500/70"
                    style={{
                      left: `${clampPct((segment.start / duration) * 100)}%`,
                      width: `${clampPct(((segment.end - segment.start) / duration) * 100)}%`
                    }}
                  />
                ))}
              </div>
              <div className="flex flex-wrap gap-2">
                <button className="button" disabled={!hasRawVideo} onClick={() => previewRawVideo(numberFromField(fields.clip_start))}>
                  Play Clip Start
                </button>
                <button className="button" disabled={!hasRawVideo} onClick={() => previewRawVideo(firstFightStart)}>
                  Play Fight Start
                </button>
                <button className="button" disabled={!hasRawVideo} onClick={() => previewRawVideo(lastFightEnd, 3)}>
                  Play Fight End
                </button>
              </div>
            </div>
          </div>

          {hasEditedReference ? (
            <div className="surface-panel p-3">
              <p className="section-kicker mb-1">Comparison clip</p>
              <p className="mb-3 text-sm font-semibold">Edited Reference</p>
              <video
                key={`${activeIndex}:${record.edit_path}`}
                className="aspect-video w-full rounded-md bg-black"
                controls
                preload="metadata"
                src={videoUrl(record.edit_path)}
              />
              <p className="mt-2 break-all text-xs text-slate-500 dark:text-slate-400">{record.edit_filename}</p>
            </div>
          ) : null}
        </div>

        <aside className="surface-panel self-start p-4 xl:sticky xl:top-24">
          <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-3">
            {(["clip_start", "clip_end"] as const).map((key) => (
              <label key={key} className="field-label">
                {key.replace("_", " ")}
                <input
                  className="input-field"
                  inputMode="decimal"
                  value={fields[key]}
                  onChange={(event) => updateField(key, event.target.value)}
                />
              </label>
            ))}
          </div>

          <div className="divider mt-4 pt-4">
            <div className="mb-3 flex items-center justify-between gap-2">
              <h3 className="text-sm font-semibold">Fight Segments</h3>
              <button
                className="button min-h-8 px-2 py-1 text-xs"
                onClick={addFightSegment}
                type="button"
              >
                Add
              </button>
            </div>
            <div className="grid gap-3">
              {fields.fight_segments.map((segment, index) => (
                <div key={index} className="rounded-md border border-lane bg-slate-50 p-2 dark:border-slate-700 dark:bg-slate-950">
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold">Fight {index + 1}</span>
                    <button
                      className="button-danger min-h-8 px-2 py-1 text-xs"
                      disabled={fields.fight_segments.length <= 1}
                      onClick={() => removeFightSegment(index)}
                      type="button"
                    >
                      Remove
                    </button>
                  </div>
                  <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-2">
                    <label className="grid gap-1 text-xs font-medium text-slate-700 dark:text-slate-300">
                      start
                      <input
                        className="input-field px-2"
                        inputMode="decimal"
                        value={segment.start}
                        onChange={(event) => updateFightSegment(index, "start", event.target.value)}
                      />
                    </label>
                    <label className="grid gap-1 text-xs font-medium text-slate-700 dark:text-slate-300">
                      end
                      <input
                        className="input-field px-2"
                        inputMode="decimal"
                        value={segment.end}
                        onChange={(event) => updateFightSegment(index, "end", event.target.value)}
                      />
                    </label>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <dl className="mt-4 grid gap-2 rounded-md border border-lane bg-slate-50 p-3 text-sm dark:border-slate-800 dark:bg-slate-950">
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500 dark:text-slate-400">Clip start</dt>
              <dd className="text-right font-semibold">{fmt(currentClipStart)}s</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500 dark:text-slate-400">Clip end</dt>
              <dd className="text-right font-semibold">{fmt(currentClipEnd)}s</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500 dark:text-slate-400">Clip duration</dt>
              <dd className="text-right">{fmt(currentClipDuration)}s</dd>
            </div>
            <div className="grid gap-1">
              <dt className="text-slate-500 dark:text-slate-400">Fight segments</dt>
              <dd className="grid gap-1 text-right font-semibold">
                {fightSegmentValues.map((segment, index) => (
                  <span key={`${index}:${segment.start}:${segment.end}`}>
                    {index + 1}: {fmt(segment.start)}s-{fmt(segment.end)}s
                  </span>
                ))}
              </dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500 dark:text-slate-400">Match</dt>
              <dd className="text-right font-semibold">{record.match.confidence}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500 dark:text-slate-400">Method</dt>
              <dd className="text-right">{record.match.method}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500 dark:text-slate-400">Score</dt>
              <dd className="text-right">{record.match.score ?? "n/a"}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-500 dark:text-slate-400">Raw duration</dt>
              <dd className="text-right">{fmt(record.raw_duration)}s</dd>
            </div>
          </dl>

          <label className="field-label mt-4">
            Review note
            <textarea
              className="textarea-field"
              value={fields.review_note}
              onChange={(event) => updateField("review_note", event.target.value)}
            />
          </label>

          {error ? <p className="mt-3 rounded-md border border-danger bg-red-50 p-3 text-sm text-danger dark:bg-red-950/30">{error}</p> : null}
          {status ? <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">{status}</p> : null}

          <div className="mt-4 grid grid-cols-2 gap-2">
            <button className="button" onClick={() => move(-1)}>
              Previous
            </button>
            <button className="button" onClick={() => move(1)}>
              Next
            </button>
            <button className="button-primary col-span-2" disabled={detectingFight || deletingRecord || !hasRawVideo} onClick={() => void detectFightClip()}>
              {detectingFight ? "Finding..." : "Find Clip"}
            </button>
            <button className="button-warning" disabled={deletingRecord} onClick={() => void skipRecord()}>
              Skip
            </button>
            <button className="button" disabled={deletingRecord} onClick={() => void save(false)}>
              Save
            </button>
            <button className="button-danger" disabled={deletingRecord} onClick={() => void deleteRecord()}>
              {deletingRecord ? "Deleting..." : "Delete Entry"}
            </button>
            <button className="button-primary" disabled={deletingRecord} onClick={() => void save(true)}>
              Approve
            </button>
          </div>
        </aside>
      </div>
    </section>
  );
}


