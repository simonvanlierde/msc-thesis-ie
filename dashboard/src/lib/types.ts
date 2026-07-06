// Shapes of the JSON produced by dashboard/scripts/*.py. Kept in sync by hand —
// the build scripts are the source of truth.

export type ScenarioKey = "SQ" | "2030" | "2050_L" | "2050_M" | "2050_H";

export interface GhgStages {
  production_phase: number;
  electricity: number;
  refrigerant_leaks: number;
  EoL_phase: number;
}

export interface Archetype {
  building_type: string;
  energy_class: string;
  use: "Residential" | "Office";
  age: "New" | "Old";
  form: "Highrise" | "Lowrise";
  floor_area_m2: number;
  E_cooling_kWh: number;
  P_peak_kW: number;
  electricity_kWh: number;
  ghg: GhgStages;
  GHG_emissions_total_kgCO2eq: number;
  ADP_kgSbeq: number;
  CSI_kgSieq: number;
}

export interface ScenarioTotals {
  floor_area_m2: number;
  E_cooling_kWh: number;
  electricity_kWh: number;
  GHG_emissions_total_kgCO2eq: number;
  ADP_kgSbeq: number;
  CSI_kgSieq: number;
}

export interface Scenario {
  label: string;
  totals: ScenarioTotals;
  lca_by_stage: GhgStages;
  archetypes: Archetype[];
}

export interface ScenariosData {
  meta: {
    source: string;
    doi_data: string;
    ghg_stages: Record<keyof GhgStages, string>;
    categories: Record<string, { label: string; unit: string }>;
    scenario_order: ScenarioKey[];
  };
  scenarios: Record<ScenarioKey, Scenario>;
}

export interface TemporalData {
  meta: {
    source: string;
    scenario: string;
    weather_years: string;
    sample_buildings: number;
    validation: string;
    units: { diurnal: string; monthly: string };
  };
  hour_of_day: number[];
  months: string[];
  seasons: string[];
  uses: string[];
  diurnal_by_season: Record<string, Record<string, number[]>>;
  monthly: Record<string, number[]>;
}

export interface HourlyFrames {
  meta: {
    source: string;
    scenario: string;
    weather_years: string;
    metric: string;
    vmax: number;
    frame_order: string;
  };
  months: string[];
  hours: number[];
  buurtcodes: string[];
  frames: number[][]; // frames[frameIndex][buurtIndex]
}

export type BuurtProps = {
  buurtcode: string;
  buurtnaam: string;
} & Record<string, number | string>;

export interface BuurtCollection {
  type: "FeatureCollection";
  metadata: { source: string; doi_data: string; scenarios: string[]; note: string };
  features: Array<{
    type: "Feature";
    properties: BuurtProps;
    geometry: GeoJSON.Geometry;
  }>;
}
