// Pure transforms over the scenario data — everything the charts and summary need.
// These are the unit-tested core (see transform.test.ts).

import type { Archetype, GhgStages, Scenario } from "./types";

export type Facet = "use" | "age" | "form";

export interface RollupRow {
  key: string;
  floor_area_m2: number;
  E_cooling_kWh: number;
  GHG_emissions_total_kgCO2eq: number;
  ADP_kgSbeq: number;
  CSI_kgSieq: number;
}

/** Sum the archetype metrics grouped by a facet (Residential/Office, New/Old, …). */
export function rollup(archetypes: Archetype[], facet: Facet): RollupRow[] {
  const acc = new Map<string, RollupRow>();
  for (const a of archetypes) {
    const key = a[facet];
    const row = acc.get(key) ?? {
      key,
      floor_area_m2: 0,
      E_cooling_kWh: 0,
      GHG_emissions_total_kgCO2eq: 0,
      ADP_kgSbeq: 0,
      CSI_kgSieq: 0,
    };
    row.floor_area_m2 += a.floor_area_m2;
    row.E_cooling_kWh += a.E_cooling_kWh;
    row.GHG_emissions_total_kgCO2eq += a.GHG_emissions_total_kgCO2eq;
    row.ADP_kgSbeq += a.ADP_kgSbeq;
    row.CSI_kgSieq += a.CSI_kgSieq;
    acc.set(key, row);
  }
  return [...acc.values()];
}

export const STAGE_ORDER: (keyof GhgStages)[] = [
  "production_phase",
  "electricity",
  "refrigerant_leaks",
  "EoL_phase",
];

export interface StageRow {
  stage: keyof GhgStages;
  label: string;
  value: number;
}

/** Life-cycle GHG broken down by stage, in life-cycle order, kg CO2-eq. */
export function lcaStages(scenario: Scenario, labels: Record<keyof GhgStages, string>): StageRow[] {
  return STAGE_ORDER.map((stage) => ({
    stage,
    label: labels[stage],
    value: scenario.lca_by_stage[stage],
  }));
}

/** Share of a subgroup (by facet value) within the scenario, for a metric. */
export function share(
  archetypes: Archetype[],
  facet: Facet,
  value: string,
  metric: keyof RollupRow,
): number {
  const rows = rollup(archetypes, facet);
  const total = rows.reduce((s, r) => s + (r[metric] as number), 0);
  const sub = rows.find((r) => r.key === value);
  if (!sub || total === 0) return 0;
  return (sub[metric] as number) / total;
}

/**
 * Electricity-per-cooling factor per building use (kWh electricity / kWh cooling demand),
 * from a scenario's archetypes. In the model this is PUE × market penetration, both hourly
 * constants — so scaling an hourly cooling series by the use's annual factor reproduces the
 * hourly electricity series exactly up to the archetype mix within the use.
 */
export function elecFactors(archetypes: Archetype[]): { residential: number; office: number } {
  const factor = (use: Archetype["use"]) => {
    const rows = archetypes.filter((a) => a.use === use);
    const cooling = rows.reduce((s, a) => s + a.E_cooling_kWh, 0);
    const elec = rows.reduce((s, a) => s + a.electricity_kWh, 0);
    return cooling === 0 ? 0 : elec / cooling;
  };
  return { residential: factor("Residential"), office: factor("Office") };
}

/** Headline framing numbers (offices: small footprint, outsized demand & emissions). */
export function officeHeadline(archetypes: Archetype[]) {
  return {
    areaShare: share(archetypes, "use", "Office", "floor_area_m2"),
    demandShare: share(archetypes, "use", "Office", "E_cooling_kWh"),
    ghgShare: share(archetypes, "use", "Office", "GHG_emissions_total_kgCO2eq"),
  };
}
