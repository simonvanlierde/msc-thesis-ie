import { type GeoJSONSource, MapLibreMap, NavigationControl, setWorkerUrl } from "maplibre-gl";
// `?worker&url` and not `?url`: the worker imports a sibling shared chunk, and only the worker
// form makes Vite bundle that graph. With `?url` the file is copied verbatim, its import 404s,
// and the worker dies silently — no tiles are ever parsed, so the map never fires `load`.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { useEffect, useMemo, useRef, useState } from "react";
import "maplibre-gl/dist/maplibre-gl.css";
import { attachTooltip, firstSymbolId, loadStyle, plainStyle } from "../lib/basemap";
import {
  binIndex,
  legendRows,
  type MapMetric,
  metricValue,
  quantileBreaks,
} from "../lib/choropleth";
import { num } from "../lib/format";
import { bbox } from "../lib/geo";
import type { Palette } from "../lib/palette";
import { scenarioLabel } from "../lib/scenarioMeta";
import type { BuurtCollection, ScenarioKey } from "../lib/types";
import { Legend } from "./Legend";
import { Segmented } from "./Segmented";
import { ValueStrip } from "./ValueStrip";

// v6 loads its worker from a separate file at runtime and bundlers cannot rewrite that URL, so
// every bundler-based app must point maplibre at it before the first map is built.
setWorkerUrl(new URL(workerUrl, document.baseURI).href);

interface Props {
  buurten: BuurtCollection | null;
  scenario: ScenarioKey;
  palette: Palette;
  onRetry: () => void;
}

export const METRIC_LABEL: Record<MapMetric, string> = {
  intensity: "Cooling intensity (kWh per m² floor area)",
  total: "Total cooling demand (GWh)",
};
const METRIC_UNIT: Record<MapMetric, string> = { intensity: "kWh/m²", total: "GWh" };
const METRIC_OPTIONS: { value: MapMetric; label: string }[] = [
  { value: "intensity", label: "Intensity (per m²)" },
  { value: "total", label: "Total (GWh)" },
];
const fmt1 = (n: number) => num(n, 1);

export function MapView({ buurten, scenario, palette, onRetry }: Props) {
  const [metric, setMetric] = useState<MapMetric>("intensity");
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const [styleVersion, setStyleVersion] = useState(0);

  // Latest palette + basemap flag for the style.load handler, which outlives the render
  // that created it.
  const pal = useRef(palette);
  pal.current = palette;
  const hasBasemap = useRef(false);

  // Compute per-feature value + colour + the legend for the current scenario/metric/theme.
  // Class breaks pool every scenario's values, so the colour scale holds still when the
  // scenario switches and the maps compare directly.
  const view = useMemo(() => {
    if (!buurten) return null;
    const allScenarios = buurten.metadata.scenarios as ScenarioKey[];
    const pooled = buurten.features.flatMap((f) =>
      allScenarios
        .map((s) => metricValue(f.properties, s, metric))
        .filter((v): v is number => v !== null),
    );
    const values = buurten.features
      .map((f) => metricValue(f.properties, scenario, metric))
      .filter((v): v is number => v !== null);
    const breaks = quantileBreaks(pooled, palette.sequential.length);
    const features = buurten.features.map((f) => {
      const v = metricValue(f.properties, scenario, metric);
      const color = v === null ? palette.grid : palette.sequential[binIndex(v, breaks)];
      return { ...f, properties: { ...f.properties, __v: v ?? -1, __c: color } };
    });
    const fc = { ...buurten, features } as unknown as GeoJSON.FeatureCollection;
    return { fc, values, breaks, legend: legendRows(breaks, palette.sequential, fmt1) };
  }, [buurten, scenario, metric, palette]);

  // Create the map immediately so theme changes can cancel obsolete style requests.
  useEffect(() => {
    if (!(container.current && buurten)) return;
    const m = new MapLibreMap({
      container: container.current,
      style: plainStyle(pal.current),
      // CARTO attribution becomes available when its style replaces the plain fallback.
      attributionControl: { compact: true },
      dragRotate: false,
      // lets the WebGL canvas appear in screenshots (maplibre v5 nests this option)
      canvasContextAttributes: { preserveDrawingBuffer: true },
    });
    m.addControl(new NavigationControl({ showCompass: false }), "top-right");
    m.fitBounds(bbox(buurten), { padding: 20, animate: false });

    // Fires on first load and again after every setStyle, which drops our layers.
    m.on("style.load", () => {
      m.addSource("buurten", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      m.addLayer(
        {
          id: "fill",
          type: "fill",
          source: "buurten",
          // semi-transparent over a basemap so the streets underneath still read
          paint: {
            "fill-color": ["get", "__c"],
            "fill-opacity": hasBasemap.current ? 0.72 : 0.9,
          },
        },
        firstSymbolId(m), // under the place labels
      );
      m.addLayer(
        {
          id: "outline",
          type: "line",
          source: "buurten",
          paint: { "line-color": pal.current.baseline, "line-width": 0.5 },
        },
        firstSymbolId(m),
      );
      setStyleVersion((version) => version + 1);
    });

    attachTooltip(m, "fill", (f) => [
      String(f.properties.buurtnaam),
      Number(f.properties.__v) < 0 ? "no data" : fmt1(Number(f.properties.__v)),
    ]);

    map.current = m;

    return () => {
      map.current?.remove();
      map.current = null;
    };
  }, [buurten]);

  // One owner for initial and subsequent style requests; the current theme always wins.
  useEffect(() => {
    const m = map.current;
    if (!(m && buurten)) return;
    let cancelled = false;
    container.current?.removeAttribute("data-map-idle");
    loadStyle(palette).then(({ style, basemap }) => {
      if (cancelled || map.current !== m) return;
      hasBasemap.current = basemap;
      m.setStyle(style); // style.load re-adds the source and layers
      // The placeholder's idle event must not mark a pending basemap as finished.
      m.once("idle", () => {
        if (!cancelled && map.current === m) container.current?.setAttribute("data-map-idle", "");
      });
    });
    return () => {
      cancelled = true;
    };
  }, [palette, buurten]);

  // Every style replacement recreates the source, even if the view data stays the same.
  useEffect(() => {
    const m = map.current;
    if (!(m && view && styleVersion)) return;
    (m.getSource("buurten") as GeoJSONSource | undefined)?.setData(view.fc);
  }, [view, styleVersion]);

  return (
    <section id="map" aria-labelledby="map-h">
      <h2 id="map-h">
        Where cooling is needed <span className="h-scenario">· {scenarioLabel(scenario)}</span>
      </h2>
      <p className="lede">
        The cooling demand of ~59,000 individual buildings, added up for each of the city's 112
        neighbourhoods. Darker means more cooling.
      </p>

      <div className="viewctl">
        <Segmented
          name="mapmetric"
          legend="Shown on map"
          options={METRIC_OPTIONS}
          value={metric}
          onChange={setMetric}
        />
      </div>

      {buurten && view ? (
        <figure className="figure">
          {/* biome-ignore lint/a11y/useSemanticElements: a slippy map is not a fieldset; group labels the interactive region */}
          <div
            className="map"
            ref={container}
            role="group"
            aria-label={`Interactive choropleth map of ${METRIC_LABEL[metric]} across The Hague neighbourhoods. A data table of all neighbourhoods follows.`}
          />
          <Legend items={view.legend} title={METRIC_LABEL[metric]} />
          <ValueStrip
            values={view.values}
            breaks={view.breaks}
            fmt={fmt1}
            unit={METRIC_UNIT[metric]}
          />
          <figcaption>
            {METRIC_LABEL[metric]}. Scenario shown: {scenarioLabel(scenario)}. The colour scale is
            fixed across all scenarios, so the maps compare directly. Each tick on the strip is one
            neighbourhood; the taller lines mark the class breaks.
          </figcaption>
          <BuurtTable fc={buurten} scenario={scenario} metric={metric} />
        </figure>
      ) : (
        <div className="note">
          <p>Neighbourhood data is unavailable. The other results are still available.</p>
          <button type="button" className="iconbtn" onClick={onRetry}>
            Try again
          </button>
        </div>
      )}
    </section>
  );
}

function BuurtTable({
  fc,
  scenario,
  metric,
}: {
  fc: BuurtCollection;
  scenario: ScenarioKey;
  metric: MapMetric;
}) {
  const rows = fc.features
    .map((f) => ({
      code: f.properties.buurtcode,
      name: f.properties.buurtnaam,
      v: metricValue(f.properties, scenario, metric),
    }))
    .sort((a, b) => (b.v ?? -Infinity) - (a.v ?? -Infinity));
  return (
    <details className="datatable">
      <summary>Data table: all neighbourhoods</summary>
      {/* biome-ignore lint/a11y/noNoninteractiveTabindex: keyboard users must be able to scroll wide tables */}
      <section className="table-scroll" aria-label="Neighbourhood data" tabIndex={0}>
        <table>
          <caption>
            {METRIC_LABEL[metric]} · {scenarioLabel(scenario)}
          </caption>
          <thead>
            <tr>
              <th scope="col">Neighbourhood</th>
              <th scope="col">{METRIC_UNIT[metric]}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.code}>
                <td>{r.name}</td>
                <td>{r.v === null ? "No data" : num(r.v, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </details>
  );
}
