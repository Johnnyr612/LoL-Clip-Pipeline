import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#15171d",
        panel: "#f7f8fb",
        lane: "#e8ebf2",
        accent: "#2f7df6",
        danger: "#d94b58",
        success: "#238a55"
      }
    }
  },
  plugins: []
} satisfies Config;
