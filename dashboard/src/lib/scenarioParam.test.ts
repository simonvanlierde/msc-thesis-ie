import { describe, expect, it } from "vitest";
import { parseScenarioParam, withScenarioParam } from "./scenarioParam";

describe("scenario URL param", () => {
  it("parses every scenario id, in hyphen or underscore form, any case", () => {
    expect(parseScenarioParam("?scenario=2050-H")).toBe("2050_H");
    expect(parseScenarioParam("?scenario=2050_l")).toBe("2050_L");
    expect(parseScenarioParam("?scenario=sq")).toBe("SQ");
    expect(parseScenarioParam("?scenario=2030")).toBe("2030");
  });

  it("returns null on garbage or a missing param", () => {
    expect(parseScenarioParam("")).toBeNull();
    expect(parseScenarioParam("?scenario=")).toBeNull();
    expect(parseScenarioParam("?scenario=2070-X")).toBeNull();
    expect(parseScenarioParam("?other=1")).toBeNull();
  });

  it("writes the link-friendly form and preserves other params", () => {
    expect(withScenarioParam("", "2050_H")).toBe("?scenario=2050-h");
    expect(withScenarioParam("?a=1&scenario=sq", "2050_M")).toBe("?a=1&scenario=2050-m");
  });

  it("round-trips every written form back to the same key", () => {
    for (const key of ["SQ", "2030", "2050_L", "2050_M", "2050_H"] as const) {
      expect(parseScenarioParam(withScenarioParam("", key))).toBe(key);
    }
  });
});
