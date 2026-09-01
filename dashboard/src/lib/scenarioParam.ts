// Mirror the chosen scenario into the URL, so a shared link opens on that path.
// Written as "?scenario=2050-h" (hyphen, lower-case — friendlier in a link than the
// internal "2050_H"); parsing accepts either separator and any case.

import type { ScenarioKey } from "./types";

export const SCENARIO_PARAM = "scenario";

const KEYS: ScenarioKey[] = ["SQ", "2030", "2050_L", "2050_M", "2050_H"];

/** Parse ?scenario= from a search string; null on absent or unrecognised values. */
export function parseScenarioParam(search: string): ScenarioKey | null {
  const raw = new URLSearchParams(search).get(SCENARIO_PARAM);
  if (!raw) return null;
  const norm = raw.toUpperCase().replace(/-/g, "_");
  return KEYS.find((k) => k === norm) ?? null;
}

/** The search string with ?scenario= set to `key`, preserving other params. */
export function withScenarioParam(search: string, key: ScenarioKey): string {
  const params = new URLSearchParams(search);
  params.set(SCENARIO_PARAM, key.toLowerCase().replace("_", "-"));
  return `?${params}`;
}
