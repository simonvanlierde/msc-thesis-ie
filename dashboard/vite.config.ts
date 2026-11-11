import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Platforms cache a preview by image URL, so the card URL carries the PNG's hash.
const ogVersion = createHash("sha256")
  .update(readFileSync("public/og.png"))
  .digest("hex")
  .slice(0, 8);
const ogCardVersion = {
  name: "og-card-version",
  transformIndexHtml: (html: string) => html.replace('/og.png"', `/og.png?v=${ogVersion}"`),
};

// Static SPA → Cloudflare Pages. Everything ships as static assets in ./dist.
export default defineConfig({
  base: "./",
  plugins: [react(), ogCardVersion],
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
      reporter: ["text", "html", "lcov"],
    },
  },
});
