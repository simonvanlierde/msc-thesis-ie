// Colours from the validated data-viz reference palette (colourblind-safe, both
// modes selected — not an auto-flip). Categorical hues are used in fixed order;
// the sequential blue ramp encodes choropleth magnitude, light -> dark.

export type Mode = "light" | "dark";

// Sequential blue ramp (7 steps) for the choropleth, per mode. Lightest = near zero.
const SEQUENTIAL: Record<Mode, string[]> = {
  light: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
  // On the dark surface the same hues read, stepped a touch brighter at the light end.
  dark: ["#1f4a7a", "#215d9e", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6"],
};

// Categorical slots (identity), fixed order. residential vs office use slots 1 & 8
// (blue / orange — a canonical colourblind-safe pair); LCA stages use slots 4,1,8,5.
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
  muted: "#898781",
  grid: "#e1e0d9",
  baseline: "#c3c2b7",
  border: "rgba(11,11,11,0.10)",
  sequential: SEQUENTIAL.light,
  use: { Residential: "#2a78d6", Office: "#eb6834" },
  stage: {
    production_phase: "#008300", // green
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
  muted: "#898781",
  grid: "#2c2c2a",
  baseline: "#383835",
  border: "rgba(255,255,255,0.10)",
  sequential: SEQUENTIAL.dark,
  use: { Residential: "#3987e5", Office: "#d95926" },
  stage: {
    production_phase: "#008300",
    electricity: "#3987e5",
    refrigerant_leaks: "#d95926",
    EoL_phase: "#9085e9",
  },
};

export function getPalette(mode: Mode): Palette {
  return mode === "dark" ? DARK : LIGHT;
}

// Warm "heat" ramp for the animated year map — ColorBrewer YlOrRd (7-class),
// listed colourblind-safe. Pale yellow = little cooling, deep red = intense.
// Theme-independent: heat reads the same on either surface.
export const HEAT_RAMP = [
  "#ffffb2",
  "#fed976",
  "#feb24c",
  "#fd8d3c",
  "#fc4e2a",
  "#e31a1c",
  "#b10026",
];
