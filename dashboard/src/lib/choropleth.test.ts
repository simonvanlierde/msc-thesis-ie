import fc from "fast-check";
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

describe("quantileBreaks + binIndex properties", () => {
  const finite = fc.double({ noNaN: true, noDefaultInfinity: true });
  const values = fc.array(finite, { minLength: 1 });
  const bins = fc.integer({ min: 1, max: 12 });

  it("returns bins-1 ascending breaks within [min, max]", () => {
    fc.assert(
      fc.property(values, bins, (vs, n) => {
        const breaks = quantileBreaks(vs, n);
        expect(breaks).toHaveLength(n - 1);
        const lo = Math.min(...vs);
        const hi = Math.max(...vs);
        for (const [i, b] of breaks.entries()) {
          expect(b).toBeGreaterThanOrEqual(lo);
          expect(b).toBeLessThanOrEqual(hi);
          if (i > 0) expect(b).toBeGreaterThanOrEqual(breaks[i - 1]);
        }
      }),
    );
  });

  it("keeps breaks finite when the values span more than Number.MAX_VALUE", () => {
    // fast-check's counterexample: the span overflowed to Infinity and the break with it.
    expect(quantileBreaks([-3.692304572639232e293, 1.7976931348623121e308], 2)).toEqual([
      8.988465674311542e307,
    ]);
  });

  it("puts every value in bin 0..bins-1", () => {
    fc.assert(
      fc.property(values, bins, fc.double(), (vs, n, v) => {
        const i = binIndex(v, quantileBreaks(vs, n));
        expect(i).toBeGreaterThanOrEqual(0);
        expect(i).toBeLessThanOrEqual(n - 1);
      }),
    );
  });
});
