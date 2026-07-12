import { describe, expect, it } from "vitest";
import { bbox } from "./geo";
import type { BuurtCollection } from "./types";

const fc = (...geometries: unknown[]): BuurtCollection =>
  ({ type: "FeatureCollection", features: geometries.map((geometry) => ({ geometry })) }) as never;

describe("bbox", () => {
  it("spans a polygon's ring", () => {
    expect(
      bbox(
        fc({
          type: "Polygon",
          coordinates: [
            [
              [4.2, 52.0],
              [4.4, 52.1],
              [4.3, 52.2],
            ],
          ],
        }),
      ),
    ).toEqual([4.2, 52.0, 4.4, 52.2]);
  });

  it("spans every feature, at any nesting depth", () => {
    expect(
      bbox(
        fc(
          { type: "Polygon", coordinates: [[[4.3, 52.0]]] },
          {
            type: "MultiPolygon",
            coordinates: [[[[4.1, 51.9]]], [[[4.5, 52.3]]]],
          },
        ),
      ),
    ).toEqual([4.1, 51.9, 4.5, 52.3]);
  });
});
