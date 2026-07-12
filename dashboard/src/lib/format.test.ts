import { describe, expect, it } from "vitest";
import { compact, gwh, ktCO2, num, pct } from "./format";

describe("format helpers", () => {
  it("gwh converts kWh to GWh", () => {
    expect(gwh(1_146_000_000)).toBe("1,146 GWh");
  });
  it("ktCO2 converts kg to kt", () => {
    expect(ktCO2(8_355_000)).toBe("8.4 kt CO₂-eq");
  });
  it("pct guards divide-by-zero", () => {
    expect(pct(1, 0)).toBe("—");
    expect(pct(13, 100)).toBe("13%");
  });
  it("num adds thousands separators", () => {
    expect(num(59381)).toBe("59,381");
  });
  it("compact shortens, with an optional unit", () => {
    expect(compact(1_200_000)).toBe("1.2M");
    expect(compact(34_000, "m²")).toBe("34K m²");
  });
});
