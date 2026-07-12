import type maplibregl from "maplibre-gl";
import { afterEach, describe, expect, it, vi } from "vitest";
import { attachTooltip, firstSymbolId, loadStyle, plainStyle } from "./basemap";
import { getPalette } from "./palette";

// A real Popup needs a real Map (WebGL); record what it is told instead.
vi.mock("maplibre-gl", () => ({
  default: {
    Popup: class {
      options: unknown;
      content: HTMLElement | null = null;
      removed = false;
      constructor(options: unknown) {
        this.options = options;
      }
      setLngLat() {
        return this;
      }
      setDOMContent(el: HTMLElement) {
        this.content = el;
        return this;
      }
      addTo() {
        return this;
      }
      remove() {
        this.removed = true;
      }
    },
  },
}));

const light = getPalette("light");

afterEach(() => vi.unstubAllGlobals());

describe("loadStyle", () => {
  it("uses CARTO's style when the CDN answers", async () => {
    const carto = { version: 8, sources: {}, layers: [] };
    vi.stubGlobal("fetch", () =>
      Promise.resolve({ ok: true, json: () => Promise.resolve(carto) } as Response),
    );
    await expect(loadStyle(light)).resolves.toEqual({ style: carto, basemap: true });
  });

  it("falls back to the plain background when the CDN is unreachable or errors", async () => {
    const plain = { style: plainStyle(light), basemap: false };

    vi.stubGlobal("fetch", () => Promise.reject(new Error("offline")));
    await expect(loadStyle(light)).resolves.toEqual(plain);

    vi.stubGlobal("fetch", () => Promise.resolve({ ok: false, status: 503 } as Response));
    await expect(loadStyle(light)).resolves.toEqual(plain);
  });
});

describe("firstSymbolId", () => {
  const fake = (layers?: { id: string; type: string }[]) =>
    ({ getStyle: () => ({ layers }) }) as unknown as maplibregl.Map;

  it("finds the first symbol layer, so data slides under the labels", () => {
    expect(
      firstSymbolId(
        fake([
          { id: "bg", type: "background" },
          { id: "labels", type: "symbol" },
          { id: "more-labels", type: "symbol" },
        ]),
      ),
    ).toBe("labels");
  });

  it("is undefined on a style with no labels — the plain fallback", () => {
    expect(firstSymbolId(fake([{ id: "bg", type: "background" }]))).toBeUndefined();
    expect(firstSymbolId(fake(undefined))).toBeUndefined();
  });
});

describe("attachTooltip", () => {
  /** Fake map: remembers its handlers so a test can fire them. */
  function fakeMap() {
    const on = new Map<string, (e: unknown) => void>();
    const canvas = document.createElement("canvas");
    const map = {
      on: (type: string, _layer: string, fn: (e: unknown) => void) => on.set(type, fn),
      getCanvas: () => canvas,
    } as unknown as maplibregl.Map;
    return { map, on, canvas };
  }

  const event = { lngLat: [4.3, 52.1], features: [{ properties: { name: "Scheveningen" } }] };
  const text = (f: maplibregl.MapGeoJSONFeature) =>
    [String(f.properties.name), "42 GWh"] as [string, string];

  it("follows the cursor on a hover pointer, and leaves with it", () => {
    vi.stubGlobal("matchMedia", () => ({ matches: true }));
    const { map, on, canvas } = fakeMap();
    // biome-ignore lint/suspicious/noExplicitAny: the mocked Popup exposes its recorded state.
    const popup = attachTooltip(map, "fills", text) as any;
    expect(popup.options).toEqual({ closeButton: false, closeOnClick: false });

    on.get("mousemove")?.(event);
    expect(canvas.style.cursor).toBe("pointer");
    expect(popup.content?.textContent).toBe("Scheveningen42 GWh");

    on.get("mouseleave")?.({});
    expect(canvas.style.cursor).toBe("");
    expect(popup.removed).toBe(true);
  });

  it("opens a dismissible popup on tap where there is no cursor", () => {
    vi.stubGlobal("matchMedia", () => ({ matches: false }));
    const { map, on } = fakeMap();
    // biome-ignore lint/suspicious/noExplicitAny: the mocked Popup exposes its recorded state.
    const popup = attachTooltip(map, "fills", text) as any;
    expect(popup.options).toEqual({ closeButton: true, closeOnClick: true });
    expect(on.has("mousemove")).toBe(false);

    on.get("click")?.(event);
    expect(popup.content?.textContent).toBe("Scheveningen42 GWh");

    on.get("click")?.({ features: [] }); // nothing under the tap: leave the popup alone
    expect(popup.content?.textContent).toBe("Scheveningen42 GWh");
  });
});
