import { describe, expect, it } from "vitest";
import { PATHS_2050, SCENARIO_META, scenarioLabel } from "./scenarioMeta";

describe("scenario metadata", () => {
  it("labels present-day scenarios plainly and 2050 paths with their year", () => {
    expect(scenarioLabel("SQ")).toBe("Today");
    expect(scenarioLabel("2030")).toBe("2030");
    expect(scenarioLabel("2050_H")).toBe("2050 High");
  });

  it("gives every 2050 path the four assumption chips and a disclosure list", () => {
    for (const key of PATHS_2050) {
      const m = SCENARIO_META[key];
      expect(m.kind).toBe("future");
      expect(Object.values(m.assumptions).filter(Boolean)).toHaveLength(4);
      expect(m.details?.length).toBeGreaterThan(0);
    }
  });
});
