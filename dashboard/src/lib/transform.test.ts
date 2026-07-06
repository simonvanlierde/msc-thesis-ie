import { describe, expect, it } from "vitest";
import { lcaStages, officeHeadline, rollup, share } from "./transform";
import type { Archetype, GhgStages, Scenario } from "./types";

function arch(over: Partial<Archetype>): Archetype {
  return {
    building_type: "x",
    energy_class: "A",
    use: "Residential",
    age: "New",
    form: "Lowrise",
    floor_area_m2: 0,
    E_cooling_kWh: 0,
    P_peak_kW: 0,
    electricity_kWh: 0,
    ghg: { production_phase: 0, electricity: 0, refrigerant_leaks: 0, EoL_phase: 0 },
    GHG_emissions_total_kgCO2eq: 0,
    ADP_kgSbeq: 0,
    CSI_kgSieq: 0,
    ...over,
  };
}

const sample: Archetype[] = [
  arch({
    use: "Residential",
    floor_area_m2: 870,
    E_cooling_kWh: 660,
    GHG_emissions_total_kgCO2eq: 350,
  }),
  arch({ use: "Office", floor_area_m2: 130, E_cooling_kWh: 340, GHG_emissions_total_kgCO2eq: 650 }),
];

describe("rollup", () => {
  it("sums metrics per facet value", () => {
    const rows = rollup(sample, "use");
    expect(rows).toHaveLength(2);
    const office = rows.find((r) => r.key === "Office");
    expect(office?.floor_area_m2).toBe(130);
    expect(office?.E_cooling_kWh).toBe(340);
  });

  it("does not mutate the input", () => {
    const before = sample[0].floor_area_m2;
    rollup(sample, "use");
    expect(sample[0].floor_area_m2).toBe(before);
  });
});

describe("share", () => {
  it("computes a subgroup's fraction of the whole", () => {
    expect(share(sample, "use", "Office", "floor_area_m2")).toBeCloseTo(0.13, 5);
    expect(share(sample, "use", "Office", "E_cooling_kWh")).toBeCloseTo(0.34, 5);
  });

  it("returns 0 for an unknown value", () => {
    expect(share(sample, "use", "Nope", "floor_area_m2")).toBe(0);
  });
});

describe("officeHeadline", () => {
  it("reproduces the thesis framing (offices: small area, large emissions)", () => {
    const h = officeHeadline(sample);
    expect(h.areaShare).toBeCloseTo(0.13, 5);
    expect(h.demandShare).toBeCloseTo(0.34, 5);
    expect(h.ghgShare).toBeCloseTo(0.65, 5);
  });
});

describe("lcaStages", () => {
  it("returns stages in life-cycle order with labels", () => {
    const stages: GhgStages = {
      production_phase: 1,
      electricity: 10,
      refrigerant_leaks: 2,
      EoL_phase: 0.5,
    };
    const scenario = { lca_by_stage: stages } as Scenario;
    const labels: Record<keyof GhgStages, string> = {
      production_phase: "Production",
      electricity: "Electricity",
      refrigerant_leaks: "Refrigerant",
      EoL_phase: "End-of-life",
    };
    const out = lcaStages(scenario, labels);
    expect(out.map((r) => r.stage)).toEqual([
      "production_phase",
      "electricity",
      "refrigerant_leaks",
      "EoL_phase",
    ]);
    expect(out[1]).toMatchObject({ label: "Electricity", value: 10 });
  });
});
