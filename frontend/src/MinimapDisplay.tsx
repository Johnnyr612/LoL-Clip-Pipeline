import type { JobRecord } from "./types";

export function MinimapDisplay({ job }: { job: JobRecord | null }) {
  const flags = safeJson<string[]>(job?.flags, []);
  return (
    <section className="border border-lane bg-white p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold">Minimap</h2>
        <span className="text-sm text-slate-600">{flags.length} flags</span>
      </div>
      <div className="mt-4 aspect-square max-h-80 border border-lane bg-panel">
        <div className="grid h-full place-items-center text-sm text-slate-500">minimap preview</div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {flags.map((flag) => (
          <span key={flag} className="border border-lane px-2 py-1 text-xs text-slate-700">
            {flag}
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
