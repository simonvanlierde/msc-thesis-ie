// Load the static datasets produced by the Python build scripts.

import type { BuurtCollection, ScenariosData, TemporalData } from "./types";

const BASE = import.meta.env.BASE_URL;

async function getJSON<T>(name: string): Promise<T> {
  const res = await fetch(`${BASE}data/${name}`);
  if (!res.ok) throw new Error(`Failed to load ${name}: ${res.status}`);
  return (await res.json()) as T;
}

export interface Datasets {
  scenarios: ScenariosData;
  temporal: TemporalData;
  buurten: BuurtCollection | null; // optional: absent until geodata is built
}

/** Everything the page needs to render its first screen. */
export async function loadDatasets(): Promise<Datasets> {
  const [scenarios, temporal, buurten] = await Promise.all([
    getJSON<ScenariosData>("scenarios.json"),
    getJSON<TemporalData>("temporal.json"),
    getJSON<BuurtCollection>("cooling_by_buurt.geojson").catch(() => null),
  ]);
  return { scenarios, temporal, buurten };
}
