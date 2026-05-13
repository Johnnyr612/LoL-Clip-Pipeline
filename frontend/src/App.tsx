import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { CaptionEditor } from "./CaptionEditor";
import { JobDashboard } from "./JobDashboard";
import type { JobRecord } from "./types";

function App() {
  const [selectedJob, setSelectedJob] = React.useState<JobRecord | null>(null);

  return (
    <main className="min-h-screen">
      <header className="border-b border-lane bg-white">
        <div className="mx-auto max-w-7xl px-5 py-4">
          <div>
            <h1 className="text-xl font-semibold tracking-normal">LoL Clip Pipeline</h1>
            <p className="text-sm text-slate-600">Fight detection, adaptive 3:4 crop, and descriptions.</p>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-4 px-5 py-5 lg:grid-cols-[360px_1fr]">
        <JobDashboard onSelectJob={setSelectedJob} selectedJob={selectedJob} />
        <section className="grid min-w-0 gap-4">
          <CaptionEditor job={selectedJob} />
        </section>
      </div>
    </main>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
