import maplibregl from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
import "maplibre-gl/dist/maplibre-gl.css";
import { attachTooltip, firstSymbolId, loadStyle } from "../lib/basemap";
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

interface Props {
  buurten: BuurtCollection | null;
  scenario: ScenarioKey;
  palette: Palette;
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

export function MapView({ buurten, scenario, palette }: Props) {
  const [metric, setMetric] = useState<MapMetric>("intensity");
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);

  // Latest palette + basemap flag for the style.load handler, which outlives the render
  // that created it.
  const pal = useRef(palette);
  pal.current = palette;
  const hasBasemap = useRef(false);

  // Compute per-feature value + colour + the legend for the current scenario/metric/theme.
  const view = useMemo(() => {
    if (!buurten) return null;
    const values = buurten.features
      .map((f) => metricValue(f.properties, scenario, metric))
      .filter((v): v is number => v !== null);
    const breaks = quantileBreaks(values, palette.sequential.length);
    const features = buurten.features.map((f) => {
      const v = metricValue(f.properties, scenario, metric);
      const color = v === null ? palette.grid : palette.sequential[binIndex(v, breaks)];
      return { ...f, properties: { ...f.properties, __v: v ?? -1, __c: color } };
    });
    const fc = { ...buurten, features } as unknown as GeoJSON.FeatureCollection;
    return { fc, values, breaks, legend: legendRows(breaks, palette.sequential, fmt1) };
  }, [buurten, scenario, metric, palette]);

  // Init the map once. Style loading is async (CARTO, with an offline fallback), so the
  // map is created inside the promise rather than at the top of the effect.
  // biome-ignore lint/correctness/useExhaustiveDependencies: run once on mount only
  useEffect(() => {
    if (!(container.current && buurten)) return;
    let cancelled = false;

    loadStyle(pal.current).then(({ style, basemap }) => {
      if (cancelled || !container.current) return;
      hasBasemap.current = basemap;
      const m = new maplibregl.Map({
        container: container.current,
        style,
        // CARTO's licence requires attribution; the plain fallback has nothing to credit.
        attributionControl: basemap && { compact: true },
        dragRotate: false,
        // lets the WebGL canvas appear in screenshots (maplibre v5 nests this option)
        canvasContextAttributes: { preserveDrawingBuffer: true },
      });
      m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
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
        setReady(true);
      });

      attachTooltip(m, "fill", (f) => [
        String(f.properties.buurtnaam),
        Number(f.properties.__v) < 0 ? "no data" : fmt1(Number(f.properties.__v)),
      ]);

      map.current = m;
    });

    return () => {
      cancelled = true;
      map.current?.remove();
      map.current = null;
    };
  }, []);

  // Theme change → new basemap style. Skipped on mount, when the map does not exist yet.
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    let cancelled = false;
    setReady(false);
    loadStyle(palette).then(({ style, basemap }) => {
      if (cancelled || map.current !== m) return;
      hasBasemap.current = basemap;
      m.setStyle(style); // style.load re-adds the source and layers
    });
    return () => {
      cancelled = true;
    };
  }, [palette]);

  // Push new data / colours whenever the view or the style changes.
  useEffect(() => {
    const m = map.current;
    if (!(m && ready && view)) return;
    (m.getSource("buurten") as maplibregl.GeoJSONSource | undefined)?.setData(view.fc);
  }, [view, ready]);

  return (
    <section id="map" aria-labelledby="map-h">
      <h2 id="map-h">Where cooling is needed</h2>
      <p className="lede">
        Cooling demand aggregated from ~59,000 individual buildings to the city's 112 neighbourhoods
        (buurten). Darker means more cooling.
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
            aria-label={`Interactive choropleth map of ${METRIC_LABEL[metric]} across The Hague neighbourhoods. A data table of the highest neighbourhoods follows.`}
          />
          <Legend items={view.legend} title={METRIC_LABEL[metric]} />
          <ValueStrip
            values={view.values}
            breaks={view.breaks}
            fmt={fmt1}
            unit={METRIC_UNIT[metric]}
          />
          <figcaption>
            {METRIC_LABEL[metric]}. Scenario shown: {scenarioLabel(scenario)}. Each tick on the
            strip is one neighbourhood; the rules are the class breaks.
          </figcaption>
          <BuurtTable fc={buurten} scenario={scenario} metric={metric} />
        </figure>
      ) : (
        <p className="note">
          Map data not built yet. Run <code>python dashboard/scripts/build_choropleth.py</code> with
          the geodata present to generate the neighbourhood layer.
        </p>
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
    .map((f) => ({ name: f.properties.buurtnaam, v: metricValue(f.properties, scenario, metric) }))
    .filter((r) => r.v !== null)
    .sort((a, b) => (b.v as number) - (a.v as number))
    .slice(0, 15);
  return (
    <details className="datatable">
      <summary>Data table — 15 highest neighbourhoods</summary>
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
            <tr key={r.name}>
              <td>{r.name}</td>
              <td>{num(r.v as number, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}
