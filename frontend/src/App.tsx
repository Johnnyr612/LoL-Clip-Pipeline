import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { OutputPanel } from "./OutputPanel";
import { JobDashboard } from "./JobDashboard";
import { LabelReview } from "./LabelReview";
import { SettingsPanel } from "./SettingsPanel";
import type { JobRecord } from "./types";

function App() {
  const [selectedJob, setSelectedJob] = React.useState<JobRecord | null>(null);
  const [darkMode, setDarkMode] = React.useState(() => window.localStorage.getItem("lolClipTheme") === "dark");
  const [activeView, setActiveView] = React.useState<"jobs" | "labels" | "settings">("jobs");
  const [inputFolders, setInputFolders] = React.useState<string[]>(() => {
    try {
      const saved = JSON.parse(window.localStorage.getItem("lolClipInputFolders") || "[]");
      return Array.isArray(saved) ? saved.filter((item): item is string => typeof item === "string") : [];
    } catch {
      return [];
    }
  });

  React.useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    window.localStorage.setItem("lolClipTheme", darkMode ? "dark" : "light");
  }, [darkMode]);

  React.useEffect(() => {
    window.localStorage.setItem("lolClipInputFolders", JSON.stringify(inputFolders));
  }, [inputFolders]);

  return (
    <main className="app-page">
      <header className="app-header sticky top-0 z-20">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-5 py-4">
          <div className="min-w-0">
            <p className="section-kicker">Local creator tool</p>
            <h1 className="text-2xl font-semibold tracking-normal text-slate-950 dark:text-white">LoL Clip Pipeline</h1>
            <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">Fight detection, dynamic vertical crop, review, and publishing.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              className={activeView === "jobs" ? "button-primary" : "button"}
              onClick={() => setActiveView("jobs")}
              type="button"
            >
              Jobs
            </button>
            <button
              className={activeView === "labels" ? "button-primary" : "button"}
              onClick={() => setActiveView("labels")}
              type="button"
            >
              Label Review
            </button>
            <button
              className={activeView === "settings" ? "button-primary" : "button"}
              onClick={() => setActiveView("settings")}
              type="button"
            >
              Settings
            </button>
            <button
              className="button"
              onClick={() => setDarkMode((value) => !value)}
              type="button"
              aria-label="Toggle dark mode"
            >
              {darkMode ? "Light" : "Dark"}
            </button>
          </div>
        </div>
      </header>

      {activeView === "jobs" ? (
        <div className="mx-auto grid max-w-[92rem] grid-cols-1 gap-5 px-5 py-5 lg:grid-cols-[minmax(460px,520px)_minmax(0,1fr)]">
          <JobDashboard onSelectJob={setSelectedJob} selectedJob={selectedJob} />
          <section className="grid min-w-0 gap-4">
            <OutputPanel job={selectedJob} />
          </section>
        </div>
      ) : activeView === "labels" ? (
        <div className="mx-auto max-w-7xl px-5 py-5">
          <LabelReview />
        </div>
      ) : (
        <div className="mx-auto max-w-7xl px-5 py-5">
          <SettingsPanel inputFolders={inputFolders} setInputFolders={setInputFolders} />
        </div>
      )}
    </main>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

