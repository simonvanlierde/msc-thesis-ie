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
      "The Hague needs 1,146 GWh of cooling a year — and most of it is invisible: 77% of the demand goes unmet. A small, energy-hungry office stock, just 13% of the floor area, drives 34% of demand and 65% of the emissions.",
    assumptions: {
      warming: "Baseline · 2018–2022",
      comfort: "Cools at 25 °C",
      // Life-cycle intensity of the 2019 Dutch grid — what the published results were
      // computed with (thesis Table 29), not the lower 2025 production-mix figure.
      grid: "427 g CO₂/kWh",
      refrigerant: "R-134A · GWP 1603",
    },
  },
  "2030": {
    kind: "soon",
    short: "2030",
    tagline: "Close, and mostly locked in",
    blurb:
      "Summers run about 0.3 °C warmer, but a cleaner grid more than offsets it: emissions fall to 25.1 kt even as demand holds flat. The near future bends downward on its own — 2050 is where the paths fork.",
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
      "Stringent adaptation meets aggressive decarbonisation: people accept a slightly warmer indoors, the grid runs near-clean, and refrigerants go zero-GWP. Even in a warming climate, cooling demand falls and emissions drop to a tenth of today's.",
    assumptions: {
      warming: "+0.6 °C summer",
      comfort: "Adapts up to 26 °C",
      grid: "29 g CO₂/kWh",
      refrigerant: "Natural · GWP 0",
    },
    details: [
      "Summers +0.6 °C over the 2018–2022 baseline (KNMI low-emission climate path)",
      "Urban heat island roughly today's: +0.1 °C by day and by night",
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
      "Today's comfort expectations hold and the grid decarbonises at a regional pace. Demand grows with the city to 1,355 GWh, but the cleaner grid still keeps emissions down at 9.3 kt.",
    assumptions: {
      warming: "+0.9 °C summer",
      comfort: "Cools at 25 °C",
      grid: "42 g CO₂/kWh",
      refrigerant: "Low-GWP · GWP ~1",
    },
    details: [
      "Summers +0.9 °C over the 2018–2022 baseline (KNMI middle climate path)",
      "Urban heat island intensifies slightly: +0.3 °C by day and by night",
      "Comfort unchanged: buildings cool once indoor temperature passes 25 °C",
      "Grid at 42 g CO₂ per kWh — the region's announced decarbonisation pace",
      "Low-GWP refrigerants (global-warming potential ≈ 1)",
      "New residential floor area nearly doubles (+97%); the old office stock shrinks 23%",
    ],
  },
  "2050_H": {
    kind: "future",
    short: "High",
    tagline: "Business as usual",
    blurb:
      "Summers run +1.3 °C hotter with the worst urban-heat-island effect, and people expect more cooling, not less. Decarbonisation stalls — the grid stays as dirty as 2030 and high-GWP refrigerants return. Cooling demand nearly doubles; emissions hit 83.6 kt, challenging the Netherlands' net-zero goal.",
    assumptions: {
      warming: "+1.3 °C summer",
      comfort: "Falls to 23 °C",
      grid: "159 g CO₂/kWh · stalls",
      refrigerant: "R-32 returns · GWP 809",
    },
    details: [
      "Summers +1.3 °C over the 2018–2022 baseline (KNMI high-emission climate path)",
      "Urban heat island worsens: +1.5 °C by day and by night",
      "Comfort expectations rise: buildings start cooling at 23 °C indoors",
      "Decarbonisation stalls — the grid stays at 2030's 159 g CO₂ per kWh",
      "High-warming refrigerants return (R-32, global-warming potential 809)",
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
