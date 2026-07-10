import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { OutputPanel } from "./OutputPanel";
import { JobDashboard } from "./JobDashboard";
import { LabelReview } from "./LabelReview";
import type { JobRecord } from "./types";

function App() {
  const [selectedJob, setSelectedJob] = React.useState<JobRecord | null>(null);
  const [darkMode, setDarkMode] = React.useState(() => window.localStorage.getItem("lolClipTheme") === "dark");
  const [activeView, setActiveView] = React.useState<"jobs" | "labels">("jobs");

  React.useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    window.localStorage.setItem("lolClipTheme", darkMode ? "dark" : "light");
  }, [darkMode]);

  return (
    <main className="min-h-screen bg-panel text-ink transition-colors dark:bg-slate-950 dark:text-slate-100">
      <header className="border-b border-lane bg-white dark:border-slate-800 dark:bg-slate-900">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-5 py-4">
          <div>
            <h1 className="text-xl font-semibold tracking-normal">LoL Clip Pipeline</h1>
            <p className="text-sm text-slate-600 dark:text-slate-400">Fight detection and adaptive 3:4 crop.</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              className={`border px-3 py-2 text-sm font-semibold ${activeView === "jobs" ? "border-accent text-accent" : "border-lane text-slate-700 dark:border-slate-700 dark:text-slate-200"}`}
              onClick={() => setActiveView("jobs")}
              type="button"
            >
              Jobs
            </button>
            <button
              className={`border px-3 py-2 text-sm font-semibold ${activeView === "labels" ? "border-accent text-accent" : "border-lane text-slate-700 dark:border-slate-700 dark:text-slate-200"}`}
              onClick={() => setActiveView("labels")}
              type="button"
            >
              Label Review
            </button>
            <button
              className="border border-lane px-3 py-2 text-sm font-semibold text-slate-700 hover:border-accent dark:border-slate-700 dark:text-slate-200 dark:hover:border-accent"
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
        <div className="mx-auto grid max-w-7xl grid-cols-1 gap-4 px-5 py-5 lg:grid-cols-[360px_1fr]">
          <JobDashboard onSelectJob={setSelectedJob} selectedJob={selectedJob} />
          <section className="grid min-w-0 gap-4">
            <OutputPanel job={selectedJob} />
          </section>
        </div>
      ) : (
        <div className="mx-auto max-w-7xl px-5 py-5">
          <LabelReview />
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

