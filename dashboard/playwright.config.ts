import { defineConfig, devices } from "@playwright/test";

// a11y e2e runs against the production build served by `vite preview`, on this project's
// own port (4183, set in vite.config.ts — several vite projects share this machine, so the
// default 4173 risks reusing another project's server). reuseExistingServer keeps a warm
// preview between local runs; CI always starts fresh.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  reporter: "list",
  use: { baseURL: "http://localhost:4183" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "pnpm build && pnpm preview",
    url: "http://localhost:4183",
    reuseExistingServer: !process.env.CI,
    // build + preview, on a machine that often runs several projects at once
    timeout: 240_000,
  },
});
