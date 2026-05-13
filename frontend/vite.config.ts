import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/jobs": "http://127.0.0.1:8000",
      "/process": "http://127.0.0.1:8000",
      "/train": "http://127.0.0.1:8000",
      "/auth": "http://127.0.0.1:8000",
      "/upload": "http://127.0.0.1:8000",
      "/outputs": "http://127.0.0.1:8000"
    }
  }
});
