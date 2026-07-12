// Build the social-preview card, the favicon and the apple-touch icon from the same
// thesis output the dashboard renders — the headline stat from scenarios.json, the heat
// band from temporal.json, the silhouette from the buurt geometry.
//
//   node scripts/build_og.mjs
//
// Rendering uses the Playwright chromium that already ships for the a11y tests, so this
// adds no dependency. Skips cleanly when the geodata builds haven't been run.

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DATA = join(ROOT, "public", "data");
const OUT = join(ROOT, "public");

// Mirrors HEAT_RAMP and heatBin() in src/lib/. Kept as a copy rather than imported: this
// script runs under plain node, and the seven values have not moved since ColorBrewer.
const HEAT_RAMP = ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"];
const heatColor = (v, vmax) =>
  HEAT_RAMP[v <= 0 || !(vmax > 0) ? 0 : Math.min(6, Math.floor((v / vmax) * 7))];

const readJSON = (name) => JSON.parse(readFileSync(join(DATA, name), "utf8"));
const tryJSON = (name) => {
  try {
    return readJSON(name);
  } catch {
    return null;
  }
};

const pct = (part, whole) => `${Math.round((part / whole) * 100)}%`;

/** Office's share of floor area and of total GHG, straight from the built scenario data. */
function headline(data) {
  const rows = data.scenarios.SQ.archetypes;
  const sum = (pred, key) => rows.filter(pred).reduce((t, a) => t + a[key], 0);
  const isOffice = (a) => a.use === "Office";
  const all = () => true;
  return {
    area: pct(sum(isOffice, "floor_area_m2"), sum(all, "floor_area_m2")),
    ghg: pct(sum(isOffice, "GHG_emissions_total_kgCO2eq"), sum(all, "GHG_emissions_total_kgCO2eq")),
  };
}

// Meteorological seasons by month index, mirroring SEASON_OF_MONTH in build_temporal.py.
const SEASON_OF_MONTH = [
  "Winter",
  "Winter",
  "Spring",
  "Spring",
  "Spring",
  "Summer",
  "Summer",
  "Summer",
  "Autumn",
  "Autumn",
  "Autumn",
  "Winter",
];
const DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/**
 * The 12x24 grid of citywide cooling power for a typical year: each month takes its season's
 * diurnal shape (GW by hour), rescaled so the day's energy matches that month's total (GWh).
 * temporal.json only carries four seasonal profiles, so months inside a season differ in
 * magnitude but not in shape — enough for a 168 px band.
 */
function heatGrid(data) {
  return SEASON_OF_MONTH.map((season, m) => {
    const shape = data.diurnal_by_season[season].total;
    const daily = shape.reduce((s, v) => s + v, 0);
    const scale = daily > 0 ? data.monthly.total[m] / DAYS_IN_MONTH[m] / daily : 0;
    return shape.map((v) => v * scale);
  });
}

/** The 12x24 heat band: city-average cooling intensity for every hour of a typical year. */
function heatBand(grid, width, height) {
  const cell = { w: width / 24, h: height / 12 };
  const vmax = Math.max(...grid.flat());
  const rects = grid.flatMap((row, m) =>
    row.map(
      (v, h) =>
        `<rect x="${(h * cell.w).toFixed(2)}" y="${(m * cell.h).toFixed(2)}" width="${cell.w.toFixed(2)}" height="${cell.h.toFixed(2)}" fill="${heatColor(v, vmax)}"/>`,
    ),
  );
  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">${rects.join("")}</svg>`;
}

/** City silhouette: every buurt polygon, equirectangular, fitted to a square viewBox. */
function silhouette(collection, size, pad) {
  const rings = [];
  const walk = (geom) => {
    const polys = geom.type === "MultiPolygon" ? geom.coordinates : [geom.coordinates];
    for (const poly of polys) for (const ring of poly) rings.push(ring);
  };
  for (const f of collection.features) walk(f.geometry);

  const lats = rings.flat().map((c) => c[1]);
  const lngs = rings.flat().map((c) => c[0]);
  const [minX, maxX] = [Math.min(...lngs), Math.max(...lngs)];
  const [minY, maxY] = [Math.min(...lats), Math.max(...lats)];
  // At 52°N a degree of longitude is ~0.6 of a degree of latitude; without this the city
  // comes out stretched sideways.
  const kx = Math.cos((((minY + maxY) / 2) * Math.PI) / 180);
  const w = (maxX - minX) * kx;
  const h = maxY - minY;
  const scale = (size - 2 * pad) / Math.max(w, h);
  const ox = pad + (size - 2 * pad - w * scale) / 2;
  const oy = pad + (size - 2 * pad - h * scale) / 2;
  const px = (c) => ox + (c[0] - minX) * kx * scale;
  const py = (c) => oy + (maxY - c[1]) * scale;

  // At 16 px a favicon has half a pixel per 1 viewBox unit, so vertices closer together
  // than MIN_STEP are invisible detail. Dropping them takes the path from ~20 kB to ~4 kB.
  const MIN_STEP = 0.08;
  const decimate = (pts) => {
    const out = [pts[0]];
    for (const p of pts.slice(1, -1)) {
      const last = out[out.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= MIN_STEP) out.push(p);
    }
    return out.length > 2 ? out : pts;
  };

  return rings
    .map((ring) => decimate(ring.map((c) => [px(c), py(c)])))
    .map((pts) => `M${pts.map(([x, y]) => `${x.toFixed(2)} ${y.toFixed(2)}`).join("L")}Z`)
    .join("");
}

async function shoot(html, width, height, out) {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  await page.setContent(html, { waitUntil: "load" });
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: out });
  await browser.close();
}

const scenarios = readJSON("scenarios.json");
const temporal = tryJSON("temporal.json");
const buurten = tryJSON("cooling_by_buurt.geojson");

if (!(temporal && buurten)) {
  // biome-ignore lint/suspicious/noConsole: CLI build script, this is its output
  console.log(
    "build_og: needs temporal.json and cooling_by_buurt.geojson — run `pnpm data` first.",
  );
  process.exit(0);
}

// ---- favicon: the city, in the heat ramp's deepest red -----------------------------
// The buurt polygons are drawn as one path. Stroking it in the fill colour seals the
// hairline seams between neighbours that the simplified geometry leaves behind.
const iconPath = silhouette(buurten, 32, 2);
const title = "<title>The Hague</title>";
const city = `<path d="${iconPath}" fill="#e31a1c" stroke="#e31a1c" stroke-width="0.3" stroke-linejoin="round"/>`;
writeFileSync(
  join(OUT, "icon.svg"),
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">${title}${city}</svg>\n`,
);

await shoot(
  `<body style="margin:0"><svg xmlns="http://www.w3.org/2000/svg" width="180" height="180" viewBox="0 0 32 32">${title}<rect width="32" height="32" fill="#f9f9f7"/>${city}</svg></body>`,
  180,
  180,
  join(OUT, "apple-touch-icon.png"),
);

// ---- og card ------------------------------------------------------------------------
const { area, ghg } = headline(scenarios);
const fonts = ["newsreader-latin-wght-normal", "public-sans-latin-wght-normal"].map((f) =>
  readFileSync(join(OUT, "fonts", `${f}.woff2`)).toString("base64"),
);

const card = `<!doctype html><meta charset="utf-8"><style>
  @font-face { font-family: Newsreader; src: url(data:font/woff2;base64,${fonts[0]}) format("woff2-variations"); font-weight: 100 900 }
  @font-face { font-family: "Public Sans"; src: url(data:font/woff2;base64,${fonts[1]}) format("woff2-variations"); font-weight: 100 900 }
  * { box-sizing: border-box }
  body { margin: 0; width: 1200px; height: 630px; background: #f9f9f7; color: #0b0b0b;
         font-family: "Public Sans", sans-serif; display: flex; flex-direction: column }
  .top { flex: 1 1 auto; padding: 64px 64px 0 }
  .kicker { font-size: 22px; color: #52514e; letter-spacing: 0.01em }
  h1 { font-family: Newsreader; font-weight: 500;
       font-size: 68px; line-height: 1.04; letter-spacing: -0.02em; margin: 28px 0 0;
       max-width: 17ch; text-wrap: balance }
  em { font-style: normal; color: #256abf }
  .credit { padding: 0 64px 22px; font-size: 20px; color: #52514e }
  .band { flex: 0 0 168px; display: block; width: 1200px; height: 168px }
  .band svg { display: block }
</style>
<body>
  <div class="top">
    <div class="kicker">Cooling for Comfort, Warming the World · The Hague</div>
    <h1>Offices fill just <em>${area}</em> of the floor area but drive <em>${ghg}</em> of cooling emissions.</h1>
  </div>
  <div class="credit">Every hour of a typical year, below · MSc Industrial Ecology thesis, Leiden University &amp; TU Delft</div>
  <div class="band">${heatBand(heatGrid(temporal), 1200, 168)}</div>
</body>`;

await shoot(card, 1200, 630, join(OUT, "og.png"));
// biome-ignore lint/suspicious/noConsole: CLI build script, this is its output
console.log(
  `build_og: wrote og.png, icon.svg, apple-touch-icon.png (offices ${area} area / ${ghg} GHG)`,
);
