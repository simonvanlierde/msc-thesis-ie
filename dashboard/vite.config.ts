import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Static SPA → Cloudflare Pages. Everything ships as static assets in ./dist.
export default defineConfig({
  base: "./",
  plugins: [react()],
  // Project-specific ports (several vite projects run on this machine): strictPort makes
  // a clash fail loudly instead of silently drifting to a port nobody is watching.
  server: { port: 5183, strictPort: true },
  preview: { port: 4183, strictPort: true },
  // MapLibre is a single ~1 MB lib; it's isolated in its own chunk and lazy-loaded
  // with the map views, so raise the warning threshold above it rather than chasing
  // an un-splittable dependency.
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1100 },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    coverage: {
      provider: "v8",
      include: ["src/lib/**/*.ts"],
      reporter: ["text", "html"],
    },
  },
});
