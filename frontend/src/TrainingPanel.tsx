import React from "react";
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import type { TrainingMetric } from "./types";

export function TrainingPanel() {
  const [metrics, setMetrics] = React.useState<TrainingMetric[]>([]);

  async function startTraining() {
    await fetch("/train", { method: "POST" });
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
        <button className="border border-lane px-3 py-1 text-sm font-medium" onClick={startTraining}>
          Start
        </button>
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
