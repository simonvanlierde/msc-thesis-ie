// Colours from the validated data-viz reference palette (colourblind-safe, both modes
// selected — not an auto-flip). Mirrors the tokens in src/index.css; MapLibre and nivo
// need concrete hexes in JavaScript, CSS gets the same values through light-dark().
//
// The rule the page holds to:
//   focus   — magenta. Reserved for keyboard focus, used by no chart, ever.
//   accent  — blue. Ink, links, anything interactive.
//   data    — the sequential blue ramp and the categorical hues. Saturated warm
//             colour only ever means (in the office/residential pair) "this series
//             is the office one".
//
// Residential/office stay blue+orange: it is the canonical colourblind-safe two-category
// pair.

export type Mode = "light" | "dark";

// Sequential blue ramp (7 steps) for the choropleth, per mode. Lightest = near zero.
const SEQUENTIAL: Record<Mode, string[]> = {
  light: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
  // On the dark surface the same hues read, stepped a touch brighter at the light end.
  dark: ["#1f4a7a", "#215d9e", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6"],
};

export interface Palette {
  mode: Mode;
  surface: string;
  page: string;
  textPrimary: string;
  textSecondary: string;
  muted: string;
  grid: string;
  baseline: string;
  border: string;
  focus: string;
  sequential: string[];
  use: { Residential: string; Office: string };
  stage: {
    production_phase: string;
    electricity: string;
    refrigerant_leaks: string;
    EoL_phase: string;
  };
}

const LIGHT: Palette = {
  mode: "light",
  surface: "#fcfcfb",
  page: "#f9f9f7",
  textPrimary: "#0b0b0b",
  textSecondary: "#52514e",
  muted: "#6f6d68",
  grid: "#e1e0d9",
  baseline: "#c3c2b7",
  border: "rgba(11,11,11,0.10)",
  focus: "#c4007a",
  sequential: SEQUENTIAL.light,
  use: { Residential: "#2a78d6", Office: "#eb6834" },
  stage: {
    production_phase: "#009e73", // Okabe-Ito bluish-green
    electricity: "#2a78d6", // blue
    refrigerant_leaks: "#eb6834", // orange
    EoL_phase: "#4a3aa7", // violet
  },
};

const DARK: Palette = {
  mode: "dark",
  surface: "#1a1a19",
  page: "#0d0d0d",
  textPrimary: "#ffffff",
  textSecondary: "#c3c2b7",
  muted: "#9b998f",
  grid: "#2c2c2a",
  baseline: "#383835",
  border: "rgba(255,255,255,0.10)",
  focus: "#ff6bc4",
  sequential: SEQUENTIAL.dark,
  use: { Residential: "#3987e5", Office: "#d95926" },
  stage: {
    // #00b585 sat at L 0.685, outside the dark categorical band (0.48–0.67).
    production_phase: "#00a476",
    electricity: "#3987e5",
    refrigerant_leaks: "#d95926",
    EoL_phase: "#9085e9",
  },
};

export function getPalette(mode: Mode): Palette {
  return mode === "dark" ? DARK : LIGHT;
}
