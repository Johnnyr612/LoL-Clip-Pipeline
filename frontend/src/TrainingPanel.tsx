import React from "react";
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import type { TrainingMetric } from "./types";

export function TrainingPanel() {
  const [metrics, setMetrics] = React.useState<TrainingMetric[]>([]);
  const [clipsDir, setClipsDir] = React.useState("");
  const [labels, setLabels] = React.useState("");
  const [epochs, setEpochs] = React.useState("25");
  const [batchSize, setBatchSize] = React.useState("");

  async function startTraining() {
    await fetch("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        clips_dir: clipsDir,
        labels,
        epochs: Number(epochs) || 25,
        batch_size: batchSize ? Number(batchSize) : null
      })
    });
    const events = new EventSource("/train/stream");
    events.onmessage = (event) => {
      const metric = JSON.parse(event.data) as TrainingMetric;
      setMetrics((items) => [...items.slice(-40), metric]);
      if (metric.status === "complete" || metric.status === "failed") events.close();
    };
  }

  return (
    <section className="border border-lane bg-white p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold">Training</h2>
        <button
          className="border border-lane px-3 py-1 text-sm font-medium disabled:opacity-50"
          disabled={!clipsDir || !labels}
          onClick={startTraining}
        >
          Start
        </button>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <label className="grid gap-1 text-sm font-medium">
          Clips directory
          <input
            className="border border-lane px-3 py-2 text-sm outline-none focus:border-accent"
            value={clipsDir}
            onChange={(event) => setClipsDir(event.target.value)}
            placeholder="C:/Videos/training_clips"
          />
        </label>
        <label className="grid gap-1 text-sm font-medium">
          Labels JSON
          <input
            className="border border-lane px-3 py-2 text-sm outline-none focus:border-accent"
            value={labels}
            onChange={(event) => setLabels(event.target.value)}
            placeholder="C:/Videos/labels_all.json"
          />
        </label>
        <label className="grid gap-1 text-sm font-medium">
          Epochs
          <input
            className="border border-lane px-3 py-2 text-sm outline-none focus:border-accent"
            value={epochs}
            onChange={(event) => setEpochs(event.target.value)}
            inputMode="numeric"
          />
        </label>
        <label className="grid gap-1 text-sm font-medium">
          Batch size
          <input
            className="border border-lane px-3 py-2 text-sm outline-none focus:border-accent"
            value={batchSize}
            onChange={(event) => setBatchSize(event.target.value)}
            inputMode="numeric"
            placeholder="auto"
          />
        </label>
      </div>
      <div className="mt-4 h-52 border border-lane bg-panel p-3">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={metrics}>
            <XAxis dataKey="epoch" />
            <YAxis />
            <Line dataKey="train_loss" stroke="#2f7df6" dot={false} />
            <Line dataKey="val_loss" stroke="#d94b58" dot={false} />
            <Line dataKey="accuracy" stroke="#238a55" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
