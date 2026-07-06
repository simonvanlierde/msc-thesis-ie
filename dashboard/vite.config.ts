import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Static SPA → Cloudflare Pages. Everything ships as static assets in ./dist.
export default defineConfig({
  base: "./",
  plugins: [react()],
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
