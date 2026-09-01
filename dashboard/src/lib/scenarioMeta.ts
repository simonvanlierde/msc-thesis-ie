// Narrative metadata per scenario, for the "choose your 2050" story. The prose and the
// four assumption chips are grounded in the thesis (KNMI '14/'21 climate scenarios,
// §3.3.2 Table 26; grid carbon intensities Table 29; refrigerant GWPs Table 32; comfort
// thresholds §3.3.2.2) and the committed model inputs (parameters.toml). Demand/emission
// totals are NOT restated here as data — the tiles and charts read those live from
// scenarios.json; the figures that appear in prose are the same published headlines.

import type { ScenarioKey } from "./types";

export type ScenarioKind = "now" | "soon" | "future";

export interface ScenarioMeta {
  kind: ScenarioKind;
  /** Short card title, e.g. "Low". */
  short: string;
  /** One-line character of the path. */
  tagline: string;
  /** Plain-language description a non-expert grasps without a chart. */
  blurb: string;
  /** The four defining model assumptions, as chips. Static scenario inputs, not outputs. */
  assumptions: { warming: string; comfort: string; grid: string; refrigerant: string };
  /** The full assumption list behind the chips (data/input/parameters/parameters.toml),
   *  for the card's disclosure. Only the 2050 paths carry one. */
  details?: string[];
}

export const SCENARIO_META: Record<ScenarioKey, ScenarioMeta> = {
  SQ: {
    kind: "now",
    short: "Today",
    tagline: "The city as it is",
    blurb:
      "The Hague needs 527 GWh of cooling a year, and most of it is invisible: 54% of the demand goes unmet. A small, energy-hungry office stock, just 11% of the floor area, drives 54% of demand and 79% of the emissions.",
    assumptions: {
      warming: "Baseline · 2021–2025",
      comfort: "Cools at 25 °C",
      // 2025 Dutch grid intensity (electricitymaps.com), what the current results were
      // computed with — parameters.toml [scenario."SQ"].
      grid: "262 g CO₂/kWh",
      refrigerant: "R-134A · GWP 1603",
    },
  },
  "2030": {
    kind: "soon",
    short: "2030",
    tagline: "Close, and mostly locked in",
    blurb:
      "Summers run about 0.3 °C warmer, but a cleaner grid more than offsets it: emissions fall to 13.3 kt even as demand holds flat. The real choices come at 2050.",
    assumptions: {
      warming: "+0.3 °C summer",
      comfort: "Cools at 25 °C",
      grid: "159 g CO₂/kWh",
      refrigerant: "R-32 · GWP 809",
    },
  },
  "2050_L": {
    kind: "future",
    short: "Low",
    tagline: "The adaptive path",
    blurb:
      "People accept a slightly warmer indoors, the grid runs near-clean, and refrigerants no longer warm the climate. Even in a warming climate, cooling demand falls and emissions drop to a seventh of today's.",
    assumptions: {
      warming: "+0.6 °C summer",
      comfort: "Adapts up to 26 °C",
      grid: "29 g CO₂/kWh",
      refrigerant: "Natural · GWP 0",
    },
    details: [
      "Summers +0.6 °C over the 2021–2025 baseline (KNMI low-emission climate path)",
      "Adaptive comfort: buildings only cool once indoor temperature passes 26 °C",
      "Near-clean grid at 29 g CO₂ per kWh",
      "Natural refrigerants with zero global-warming potential",
      "New residential floor area grows 21%; the old office stock shrinks 43%",
    ],
  },
  "2050_M": {
    kind: "future",
    short: "Medium",
    tagline: "Status-quo policy",
    blurb:
      "Today's comfort expectations hold and the grid gets cleaner at the pace the region has announced. Demand grows with the city to 655 GWh, but the cleaner grid still keeps emissions down at 4.9 kt.",
    assumptions: {
      warming: "+0.9 °C summer",
      comfort: "Cools at 25 °C",
      grid: "42 g CO₂/kWh",
      refrigerant: "Low-GWP · GWP ~1",
    },
    details: [
      "Summers +0.9 °C over the 2021–2025 baseline (KNMI middle climate path)",
      "Comfort unchanged: buildings cool once indoor temperature passes 25 °C",
      "Grid at 42 g CO₂ per kWh, the region's announced decarbonisation pace",
      "Low-GWP refrigerants (global-warming potential ≈ 1)",
      "New residential floor area nearly doubles (+97%); the old office stock shrinks 23%",
    ],
  },
  "2050_H": {
    kind: "future",
    short: "High",
    tagline: "Business as usual",
    blurb:
      "Summers run +1.3 °C hotter, and people expect more cooling, not less. Cleanup stalls: the grid stays as dirty as 2030 and refrigerants sit at the worst warming potential the EU's F-gas rules still allow. Cooling demand grows 79%; emissions hit 32.3 kt, hard to square with the Netherlands' net-zero goal.",
    assumptions: {
      warming: "+1.3 °C summer",
      comfort: "Falls to 23 °C",
      grid: "159 g CO₂/kWh · stalls",
      refrigerant: "F-gas ceiling · GWP 150",
    },
    details: [
      "Summers +1.3 °C over the 2021–2025 baseline (KNMI high-emission climate path)",
      "Comfort expectations rise: buildings start cooling at 23 °C indoors",
      "Decarbonisation stalls: the grid stays at 2030's 159 g CO₂ per kWh",
      "Refrigerants at GWP 150, the worst Reg. (EU) 2024/573 still permits, not best-available",
      "New residential floor area nearly triples (+182%); the old office stock shrinks 14%",
    ],
  },
};

/** The three 2050 paths, in low→high impact order — the fork's hero choices. */
export const PATHS_2050: ScenarioKey[] = ["2050_L", "2050_M", "2050_H"];

/** "Today" / "2030" / "2050 Low" — names a scenario for chart rows, pills and captions. */
export function scenarioLabel(k: ScenarioKey): string {
  const m = SCENARIO_META[k];
  return m.kind === "future" ? `2050 ${m.short}` : m.short;
}
