// Choropleth binning: quantile classes so the colour scale spreads across the
// real distribution rather than being dominated by a few large buurten. Meaning
// is never colour-alone — every bin has an explicit numeric range in the legend
// and every buurt shows its value on hover.

import type { BuurtProps, ScenarioKey } from "./types";

export type MapMetric = "intensity" | "total";

/** Value shown on the map for one buurt: cooling intensity (kWh/m²) or total (GWh). */
export function metricValue(
  p: BuurtProps,
  scenario: ScenarioKey,
  metric: MapMetric,
): number | null {
  const e = p[`${scenario}__E_cooling_kWh`] as number | undefined;
  // The old guard was `e === null`, which let a missing key through as undefined and on
  // into NaN. A buurt outside a scenario has no key at all.
  if (typeof e !== "number" || !Number.isFinite(e)) return null;
  if (metric === "total") return e / 1e6; // GWh
  const area = p[`${scenario}__floor_area_m2`] as number | undefined;
  if (!area) return null;
  return e / area; // kWh per m² of floor area
}

/** Quantile thresholds splitting sorted values into `bins` classes (bins-1 cut points). */
export function quantileBreaks(values: number[], bins: number): number[] {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (v.length === 0) return [];
  const breaks: number[] = [];
  for (let i = 1; i < bins; i++) {
    const q = (i / bins) * (v.length - 1);
    const lo = Math.floor(q);
    const hi = Math.ceil(q);
    breaks.push(v[lo] + (v[hi] - v[lo]) * (q - lo));
  }
  return breaks;
}

/** Index of the bin a value falls into, given ascending thresholds. */
export function binIndex(value: number, breaks: number[]): number {
  let i = 0;
  while (i < breaks.length && value > breaks[i]) i++;
  return i;
}

/** Legend rows: [colour, "lo – hi"] for each bin, given breaks and the ramp. */
export function legendRows(
  breaks: number[],
  ramp: string[],
  fmt: (n: number) => string,
): Array<{ color: string; label: string }> {
  const rows: Array<{ color: string; label: string }> = [];
  for (let i = 0; i < ramp.length; i++) {
    const lo = i === 0 ? null : breaks[i - 1];
    const hi = i < breaks.length ? breaks[i] : null;
    const label =
      lo === null
        ? `< ${fmt(hi as number)}`
        : hi === null
          ? `≥ ${fmt(lo)}`
          : `${fmt(lo)} – ${fmt(hi)}`;
    rows.push({ color: ramp[i], label });
  }
  return rows;
}
