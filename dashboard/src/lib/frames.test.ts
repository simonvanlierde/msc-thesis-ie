import { describe, expect, it } from "vitest";
import { frameLabel, frameParts, heatBin, heatColor, heatLegend } from "./frames";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

describe("frameParts / frameLabel", () => {
  it("splits month-major hour-minor indices", () => {
    expect(frameParts(0)).toEqual({ month: 0, hour: 0 });
    expect(frameParts(15)).toEqual({ month: 0, hour: 15 });
    expect(frameParts(6 * 24 + 15)).toEqual({ month: 6, hour: 15 });
  });
  it("wraps out-of-range indices", () => {
    expect(frameParts(288)).toEqual({ month: 0, hour: 0 });
    expect(frameParts(-1)).toEqual({ month: 11, hour: 23 });
  });
  it("labels a frame", () => {
    expect(frameLabel(6 * 24 + 15, MONTHS)).toBe("Jul · 15:00");
    expect(frameLabel(0, MONTHS)).toBe("Jan · 00:00");
  });
});

describe("heatBin / heatColor", () => {
  const ramp = ["a", "b", "c", "d", "e", "f", "g"]; // 7 bins
  it("maps zero and negatives to the first bin", () => {
    expect(heatBin(0, 30, 7)).toBe(0);
    expect(heatBin(-5, 30, 7)).toBe(0);
  });
  it("clamps values at/above vmax to the top bin", () => {
    expect(heatBin(30, 30, 7)).toBe(6);
    expect(heatBin(100, 30, 7)).toBe(6);
  });
  it("bins mid-range values", () => {
    expect(heatBin(15, 30, 7)).toBe(3); // half of vmax → bin 3
  });
  it("returns first bin when vmax is invalid", () => {
    expect(heatBin(5, 0, 7)).toBe(0);
  });
  it("heatColor indexes the ramp", () => {
    expect(heatColor(30, 30, ramp)).toBe("g");
    expect(heatColor(0, 30, ramp)).toBe("a");
  });
});

describe("heatLegend", () => {
  it("produces one row per ramp step with the top open-ended", () => {
    const rows = heatLegend(70, ["x", "y"], (n) => `${n}`);
    expect(rows).toHaveLength(2);
    expect(rows[0].label).toBe("0–35");
    expect(rows[1].label).toBe("≥ 35");
  });
});
