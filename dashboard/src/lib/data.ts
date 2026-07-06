// Load the three static datasets produced by the Python build scripts.

import type { BuurtCollection, HourlyFrames, ScenariosData, TemporalData } from "./types";

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
  frames: HourlyFrames | null; // optional: per-buurt hourly time-lapse
}

export async function loadDatasets(): Promise<Datasets> {
  const [scenarios, temporal, buurten, frames] = await Promise.all([
    getJSON<ScenariosData>("scenarios.json"),
    getJSON<TemporalData>("temporal.json"),
    getJSON<BuurtCollection>("cooling_by_buurt.geojson").catch(() => null),
    getJSON<HourlyFrames>("cooling_frames.json").catch(() => null),
  ]);
  return { scenarios, temporal, buurten, frames };
}
