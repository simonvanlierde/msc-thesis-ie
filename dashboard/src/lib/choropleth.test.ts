import { describe, expect, it } from "vitest";
import { binIndex, legendRows, metricValue, quantileBreaks } from "./choropleth";
import type { BuurtProps } from "./types";

describe("metricValue", () => {
  const p: BuurtProps = {
    buurtcode: "BU1",
    buurtnaam: "Test",
    SQ__E_cooling_kWh: 2_000_000,
    SQ__floor_area_m2: 100_000,
  };
  it("returns intensity as kWh per m²", () => {
    expect(metricValue(p, "SQ", "intensity")).toBeCloseTo(20, 5);
  });
  it("returns total as GWh", () => {
    expect(metricValue(p, "SQ", "total")).toBeCloseTo(2, 5);
  });
  it("returns null when the scenario has no data", () => {
    expect(metricValue(p, "2050_H", "total")).toBeNull();
  });
});

describe("quantileBreaks + binIndex", () => {
  it("produces bins-1 ascending cut points", () => {
    const breaks = quantileBreaks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5);
    expect(breaks).toHaveLength(4);
    for (let i = 1; i < breaks.length; i++) expect(breaks[i]).toBeGreaterThanOrEqual(breaks[i - 1]);
  });
  it("assigns values to the right bin", () => {
    const breaks = [10, 20, 30];
    expect(binIndex(5, breaks)).toBe(0);
    expect(binIndex(15, breaks)).toBe(1);
    expect(binIndex(35, breaks)).toBe(3);
  });
  it("handles empty input", () => {
    expect(quantileBreaks([], 5)).toEqual([]);
  });
});

describe("legendRows", () => {
  it("labels the first and last bins as open-ended ranges", () => {
    const rows = legendRows([10, 20], ["#a", "#b", "#c"], (n) => `${n}`);
    expect(rows).toHaveLength(3);
    expect(rows[0].label).toBe("< 10");
    expect(rows[2].label).toBe("≥ 20");
    expect(rows[1].label).toBe("10 – 20");
  });
});
