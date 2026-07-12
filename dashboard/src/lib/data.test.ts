import { afterEach, describe, expect, it, vi } from "vitest";
import { loadDatasets } from "./data";

/** Fetch stub: a name→body map; anything else 404s. */
function stubFetch(bodies: Record<string, unknown>) {
  vi.stubGlobal("fetch", (url: string) => {
    const name = url.split("/").pop() as string;
    return Promise.resolve(
      name in bodies
        ? ({ ok: true, json: () => Promise.resolve(bodies[name]) } as Response)
        : ({ ok: false, status: 404 } as Response),
    );
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("loadDatasets", () => {
  it("loads the three datasets", async () => {
    stubFetch({
      "scenarios.json": { s: 1 },
      "temporal.json": { t: 1 },
      "cooling_by_buurt.geojson": { b: 1 },
    });
    await expect(loadDatasets()).resolves.toEqual({
      scenarios: { s: 1 },
      temporal: { t: 1 },
      buurten: { b: 1 },
    });
  });

  it("tolerates missing geodata but not missing scenarios", async () => {
    stubFetch({ "scenarios.json": { s: 1 }, "temporal.json": { t: 1 } });
    await expect(loadDatasets()).resolves.toMatchObject({ buurten: null });

    stubFetch({ "temporal.json": { t: 1 } });
    await expect(loadDatasets()).rejects.toThrow("scenarios.json: 404");
  });
});
