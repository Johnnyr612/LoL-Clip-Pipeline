import React from "react";

type FolderInfo = {
  key: string;
  label: string;
  path: string;
  kind: string;
  exists: boolean;
};

type SettingsPayload = {
  folders?: FolderInfo[];
};

type Props = {
  inputFolders: string[];
  setInputFolders: (folders: string[]) => void;
};

export function SettingsPanel({ inputFolders, setInputFolders }: Props) {
  const [folders, setFolders] = React.useState<FolderInfo[]>([]);
  const [newInputFolder, setNewInputFolder] = React.useState("");
  const [message, setMessage] = React.useState("");

  React.useEffect(() => {
    void refreshFolders();
  }, []);

  async function refreshFolders() {
    const payload = await fetchJsonOr<SettingsPayload>("/settings/folders", { folders: [] });
    setFolders(Array.isArray(payload.folders) ? payload.folders : []);
  }

  async function openFolder(path: string) {
    setMessage("");
    const response = await fetch("/settings/open-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path })
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      setMessage(payload.detail ?? "Unable to open folder");
      return;
    }
    setMessage(`Opened ${payload.path ?? path}`);
  }

  function addInputFolder() {
    const path = normalizePath(newInputFolder);
    if (!path || inputFolders.includes(path)) {
      setNewInputFolder("");
      return;
    }
    setInputFolders([path, ...inputFolders]);
    setNewInputFolder("");
  }

  function removeInputFolder(path: string) {
    setInputFolders(inputFolders.filter((item) => item !== path));
  }

  return (
    <section className="grid gap-5">
      <div className="surface-panel p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="section-kicker">Local folders</p>
            <h2 className="section-title mt-1">Settings</h2>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
              Open folders used for source clips, rendered outputs, checkpoints, logs, and temp files.
            </p>
          </div>
          <button className="button min-h-8 py-1.5" onClick={() => void refreshFolders()} type="button">
            Refresh
          </button>
        </div>

        {message ? <p className="mt-4 rounded-md border border-lane bg-slate-50 p-3 text-sm text-slate-600 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-300">{message}</p> : null}

        <div className="mt-4 grid gap-3 lg:grid-cols-2">
          {folders.map((folder) => (
            <FolderCard
              exists={folder.exists}
              key={folder.key}
              kind={folder.kind}
              label={folder.label}
              onOpen={() => void openFolder(folder.path)}
              path={folder.path}
            />
          ))}
        </div>
      </div>

      <div className="surface-panel p-4">
        <div>
          <p className="section-kicker">Input folders</p>
          <h2 className="section-title mt-1">Source Clip Locations</h2>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
          <input
            className="input-field"
            onChange={(event) => setNewInputFolder(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                addInputFolder();
              }
            }}
            placeholder="D:/Medal/Clips/League of Legends"
            value={newInputFolder}
          />
          <button className="button-primary" onClick={addInputFolder} type="button">
            Add Folder
          </button>
        </div>
        <p className="mt-2 text-xs leading-5 text-slate-500 dark:text-slate-400">
          Browser security does not expose arbitrary folder paths from a picker, so save the local path here and open it from the app.
        </p>

        <div className="mt-4 grid gap-3">
          {inputFolders.length ? (
            inputFolders.map((path) => (
              <FolderCard
                exists
                key={path}
                kind="saved"
                label="Input folder"
                onOpen={() => void openFolder(path)}
                onRemove={() => removeInputFolder(path)}
                path={path}
              />
            ))
          ) : (
            <p className="rounded-md border border-dashed border-lane p-4 text-sm text-slate-500 dark:border-slate-800 dark:text-slate-400">
              No input folders saved yet.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

function FolderCard({
  exists,
  kind,
  label,
  onOpen,
  onRemove,
  path
}: {
  exists: boolean;
  kind: string;
  label: string;
  onOpen: () => void;
  onRemove?: () => void;
  path: string;
}) {
  return (
    <article className="surface-muted grid gap-3 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold">{label}</h3>
          <p className="mt-1 break-all text-xs leading-5 text-slate-500 dark:text-slate-400">{path}</p>
        </div>
        <span className={exists ? "chip chip-success shrink-0" : "chip chip-warning shrink-0"}>{exists ? kind : "missing"}</span>
      </div>
      <div className="flex flex-wrap gap-2">
        <button className="button-primary min-h-8 py-1.5" disabled={!exists} onClick={onOpen} type="button">
          Open Folder
        </button>
        {onRemove ? (
          <button className="button-danger min-h-8 py-1.5" onClick={onRemove} type="button">
            Remove
          </button>
        ) : null}
      </div>
    </article>
  );
}

async function fetchJsonOr<T>(url: string, fallback: T): Promise<T> {
  try {
    const response = await fetch(url);
    if (!response.ok) return fallback;
    return await response.json();
  } catch {
    return fallback;
  }
}

function normalizePath(value: string) {
  let path = value.trim();
  const quotePairs = [["\"", "\""], ["'", "'"]];
  while (path.length >= 2 && quotePairs.some(([left, right]) => path.startsWith(left) && path.endsWith(right))) {
    path = path.slice(1, -1).trim();
  }
  return path;
}
