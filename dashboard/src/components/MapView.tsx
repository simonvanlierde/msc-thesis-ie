import maplibregl from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
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
import type { BuurtCollection, ScenarioKey } from "../lib/types";
import { Legend } from "./Legend";

interface Props {
  buurten: BuurtCollection | null;
  scenario: ScenarioKey;
  palette: Palette;
}

const METRIC_LABEL: Record<MapMetric, string> = {
  intensity: "Cooling intensity (kWh per m² floor area)",
  total: "Total cooling demand (GWh)",
};
const fmt: Record<MapMetric, (n: number) => string> = {
  intensity: (n) => num(n, 1),
  total: (n) => num(n, 1),
};

export function MapView({ buurten, scenario, palette }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);
  const [metric, setMetric] = useState<MapMetric>("intensity");

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
    const legend = legendRows(breaks, palette.sequential, fmt[metric]);
    return { fc, legend };
  }, [buurten, scenario, metric, palette]);

  // Init the map once.
  // biome-ignore lint/correctness/useExhaustiveDependencies: run once on mount only
  useEffect(() => {
    if (!(container.current && buurten)) return;
    const m = new maplibregl.Map({
      container: container.current,
      style: {
        version: 8,
        sources: {},
        layers: [{ id: "bg", type: "background", paint: { "background-color": palette.page } }],
      },
      attributionControl: false,
      dragRotate: false,
      // lets the WebGL canvas appear in screenshots (maplibre v5 nests this option)
      canvasContextAttributes: { preserveDrawingBuffer: true },
    });
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    m.on("load", () => {
      m.addSource("buurten", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      m.addLayer({
        id: "fill",
        type: "fill",
        source: "buurten",
        paint: { "fill-color": ["get", "__c"], "fill-opacity": 0.9 },
      });
      m.addLayer({
        id: "outline",
        type: "line",
        source: "buurten",
        paint: { "line-color": palette.baseline, "line-width": 0.5 },
      });
      m.fitBounds(bbox(buurten), { padding: 20, animate: false });
      setReady(true);
    });

    const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false });
    m.on("mousemove", "fill", (e) => {
      const f = e.features?.[0];
      if (!f) return;
      m.getCanvas().style.cursor = "pointer";
      const p = f.properties as { buurtnaam: string; __v: number };
      popup
        .setLngLat(e.lngLat)
        .setHTML(
          `<div class="tooltip"><strong>${p.buurtnaam}</strong>${p.__v < 0 ? "no data" : `${num(p.__v, 1)}`}</div>`,
        )
        .addTo(m);
    });
    m.on("mouseleave", "fill", () => {
      m.getCanvas().style.cursor = "";
      popup.remove();
    });

    map.current = m;
    return () => {
      m.remove();
      map.current = null;
    };
  }, []);

  // Push new data / colours whenever the view changes.
  useEffect(() => {
    const m = map.current;
    if (!(m && ready && view)) return;
    (m.getSource("buurten") as maplibregl.GeoJSONSource | undefined)?.setData(view.fc);
    m.setPaintProperty("bg", "background-color", palette.page);
    m.setPaintProperty("outline", "line-color", palette.baseline);
  }, [view, ready, palette]);

  return (
    <section id="map" aria-labelledby="map-h">
      <h2 id="map-h">Where cooling is needed</h2>
      <p className="lede">
        Cooling demand aggregated from ~59,000 individual buildings to the city's 112 neighbourhoods
        (buurten). Darker means more cooling.
      </p>

      <fieldset className="segmented" style={{ marginBottom: "0.9rem" }}>
        <legend>Shown on map</legend>
        <div className="segmented__row">
          {(["intensity", "total"] as MapMetric[]).map((mk) => (
            <label key={mk}>
              <input
                type="radio"
                name="mapmetric"
                checked={metric === mk}
                onChange={() => setMetric(mk)}
              />
              {mk === "intensity" ? "Intensity (per m²)" : "Total (GWh)"}
            </label>
          ))}
        </div>
      </fieldset>

      {buurten ? (
        <figure className="figure">
          {/* biome-ignore lint/a11y/useSemanticElements: a slippy map is not a fieldset; group labels the interactive region */}
          <div
            className="map"
            ref={container}
            role="group"
            aria-label={`Interactive choropleth map of ${METRIC_LABEL[metric]} across The Hague neighbourhoods. A data table of the highest neighbourhoods follows.`}
          />
          {view && <Legend items={view.legend} title={METRIC_LABEL[metric]} />}
          <figcaption>
            {METRIC_LABEL[metric]}. Scenario shown: {scenario}.
          </figcaption>
          {view && <BuurtTable fc={buurten} scenario={scenario} metric={metric} />}
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
          {METRIC_LABEL[metric]} · scenario {scenario}
        </caption>
        <thead>
          <tr>
            <th scope="col">Neighbourhood</th>
            <th scope="col">{metric === "intensity" ? "kWh/m²" : "GWh"}</th>
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
