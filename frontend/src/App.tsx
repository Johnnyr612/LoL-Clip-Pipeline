import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { CaptionEditor } from "./CaptionEditor";
import { JobDashboard } from "./JobDashboard";
import { MinimapDisplay } from "./MinimapDisplay";
import { SocialConnect } from "./SocialConnect";
import { TrainingPanel } from "./TrainingPanel";
import { VideoPreview } from "./VideoPreview";
import type { JobRecord } from "./types";

function App() {
  const [selectedJob, setSelectedJob] = React.useState<JobRecord | null>(null);

  return (
    <main className="min-h-screen">
      <header className="border-b border-lane bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <div>
            <h1 className="text-xl font-semibold tracking-normal">LoL Clip Pipeline</h1>
            <p className="text-sm text-slate-600">Fight detection, adaptive 3:4 crop, captions, and publishing.</p>
          </div>
          <SocialConnect />
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-4 px-5 py-5 lg:grid-cols-[360px_1fr]">
        <JobDashboard onSelectJob={setSelectedJob} selectedJob={selectedJob} />
        <section className="grid gap-4">
          <VideoPreview job={selectedJob} />
          <div className="grid gap-4 xl:grid-cols-[1fr_420px]">
            <MinimapDisplay job={selectedJob} />
            <CaptionEditor job={selectedJob} />
          </div>
          <TrainingPanel />
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
