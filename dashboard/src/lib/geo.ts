import type { BuurtCollection } from "./types";

/** Lng/lat bounding box [west, south, east, north] over a FeatureCollection. */
export function bbox(fc: BuurtCollection): [number, number, number, number] {
  let minX = 180;
  let minY = 90;
  let maxX = -180;
  let maxY = -90;
  const scan = (c: unknown): void => {
    if (typeof c === "number") return;
    const arr = c as unknown[];
    if (typeof arr[0] === "number") {
      const [x, y] = arr as number[];
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    } else for (const p of arr) scan(p);
  };
  for (const f of fc.features) scan((f.geometry as { coordinates: unknown }).coordinates);
  return [minX, minY, maxX, maxY];
}
