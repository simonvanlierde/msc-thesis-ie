// Basemap and map-tooltip plumbing shared by MapView and HourlyMap.
//
// The choropleth sits on CARTO's positron / dark-matter raster-free vector basemap so a
// reader can find their own street. If the CDN can't be reached — offline, blocked, a bad
// day at CARTO — we fall back to the plain coloured background the dashboard shipped with.
// The map still works; it just loses its streets.

import maplibregl, { type StyleSpecification } from "maplibre-gl";
import type { Mode, Palette } from "./palette";

const CARTO: Record<Mode, string> = {
  light: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
};

/** The offline fallback: one flat background layer in the page colour. */
export function plainStyle(palette: Palette): StyleSpecification {
  return {
    version: 8,
    sources: {},
    layers: [{ id: "bg", type: "background", paint: { "background-color": palette.page } }],
  };
}

export interface BasemapStyle {
  style: StyleSpecification;
  /** True when the CARTO style loaded — drives attribution and fill opacity. */
  basemap: boolean;
}

export async function loadStyle(palette: Palette): Promise<BasemapStyle> {
  try {
    const res = await fetch(CARTO[palette.mode]);
    if (!res.ok) throw new Error(`${res.status}`);
    return { style: (await res.json()) as StyleSpecification, basemap: true };
  } catch {
    return { style: plainStyle(palette), basemap: false };
  }
}

/** Id of the style's first symbol layer, so data slides under the place labels. */
export function firstSymbolId(map: maplibregl.Map): string | undefined {
  return map.getStyle().layers?.find((l) => l.type === "symbol")?.id;
}

function tooltipNode(title: string, value: string): HTMLElement {
  const el = document.createElement("div");
  el.className = "tooltip";
  const strong = document.createElement("strong");
  strong.textContent = title;
  el.append(strong, value);
  return el;
}

/**
 * Popup on a fill layer, for both pointer kinds. On a mouse it follows the cursor and
 * leaves when the cursor does. On touch there is no cursor and no mouseleave, so a tap
 * opens a popup that has a close button and dismisses when you tap elsewhere.
 *
 * `text` receives the whole feature, because the year map keeps its value in feature
 * state rather than in the feature's properties.
 */
export function attachTooltip(
  map: maplibregl.Map,
  layerId: string,
  text: (feature: maplibregl.MapGeoJSONFeature) => [title: string, value: string],
): maplibregl.Popup {
  const hover = typeof matchMedia !== "undefined" && matchMedia("(hover: hover)").matches;
  const popup = new maplibregl.Popup({ closeButton: !hover, closeOnClick: !hover });

  const show = (e: maplibregl.MapLayerMouseEvent) => {
    const f = e.features?.[0];
    if (!f) return;
    const [title, value] = text(f);
    popup.setLngLat(e.lngLat).setDOMContent(tooltipNode(title, value)).addTo(map);
  };

  if (hover) {
    map.on("mousemove", layerId, (e) => {
      map.getCanvas().style.cursor = "pointer";
      show(e);
    });
    map.on("mouseleave", layerId, () => {
      map.getCanvas().style.cursor = "";
      popup.remove();
    });
  } else {
    map.on("click", layerId, show);
  }
  return popup;
}
