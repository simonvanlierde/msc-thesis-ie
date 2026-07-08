import maplibregl from "maplibre-gl";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "maplibre-gl/dist/maplibre-gl.css";
import { attachTooltip, firstSymbolId, loadStyle } from "../lib/basemap";
import { loadFrames } from "../lib/data";
import { num } from "../lib/format";
import { FRAME_COUNT, frameLabel, heatLegend } from "../lib/frames";
import { bbox } from "../lib/geo";
import { HEAT_RAMP, type Palette } from "../lib/palette";
import type { BuurtCollection, HourlyFrames } from "../lib/types";
import { Legend } from "./Legend";
import { YearCarpet } from "./YearCarpet";

interface Props {
  buurten: BuurtCollection | null;
  palette: Palette;
}

/**
 * The time-lapse grid is the page's biggest payload and nothing above it needs the data,
 * so it is fetched here, after first paint. Splitting the fetch out also lets the map
 * below assume its data exists for the whole of its life, which its mount-only MapLibre
 * setup depends on.
 */
export function HourlyMap({ buurten, palette }: Props) {
  const [frames, setFrames] = useState<HourlyFrames | null | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    loadFrames().then((f) => {
      if (!cancelled) setFrames(f);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  if (frames === undefined) {
    return (
      <section id="year" aria-labelledby="year-h">
        <h2 id="year-h">A year of cooling, hour by hour</h2>
        <p className="loading" role="status">
          Loading the year…
        </p>
      </section>
    );
  }

  if (!(buurten && frames)) {
    return (
      <section id="year" aria-labelledby="year-h">
        <h2 id="year-h">A year of cooling</h2>
        <p className="note">
          Time-lapse data not built yet. Run{" "}
          <code>python dashboard/scripts/build_hourly_frames.py</code> with the geodata present.
        </p>
      </section>
    );
  }

  return <YearMap buurten={buurten} frames={frames} palette={palette} />;
}

interface YearMapProps {
  buurten: BuurtCollection;
  frames: HourlyFrames;
  palette: Palette;
}

const STEP_MS = 150; // ~6.7 frames/second when playing
const fmt1 = (n: number) => num(n, 1);

/** Data-driven fill: the same equal-width bins as heatBin(), evaluated on the GPU from
 *  feature state, so a frame change pushes 112 numbers instead of 112 polygons. */
function heatExpression(vmax: number): maplibregl.ExpressionSpecification {
  const step: unknown[] = ["step", ["coalesce", ["feature-state", "v"], 0], HEAT_RAMP[0]];
  for (let i = 1; i < HEAT_RAMP.length; i++) step.push((vmax * i) / HEAT_RAMP.length, HEAT_RAMP[i]);
  return step as unknown as maplibregl.ExpressionSpecification;
}

function YearMap({ buurten, frames, palette }: YearMapProps) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);
  const [frame, setFrame] = useState(6 * 24 + 15); // open on a July afternoon
  const [playing, setPlaying] = useState(false);

  const pal = useRef(palette);
  pal.current = palette;
  const hasBasemap = useRef(false);

  const vmax = frames.meta.vmax;

  // Geometry for the buurten that have frame data, keyed by buurtcode so feature state
  // can address them. Uploaded to the GPU once; only the state changes per frame.
  const work = useMemo(() => {
    const codes = new Set(frames.buurtcodes);
    return {
      type: "FeatureCollection",
      features: buurten.features.filter((f) => codes.has(f.properties.buurtcode)),
    } as unknown as GeoJSON.FeatureCollection;
  }, [buurten, frames]);

  // City average per frame — the carpet plot's 288 cells.
  const cityMean = useMemo(
    () => frames.frames.map((row) => row.reduce((s, v) => s + v, 0) / row.length),
    [frames],
  );

  const paint = useCallback(
    (idx: number) => {
      const m = map.current;
      if (!(m && ready)) return;
      const row = frames.frames[idx];
      frames.buurtcodes.forEach((code, i) => {
        m.setFeatureState({ source: "frames", id: code }, { v: row[i] });
      });
    },
    [ready, frames],
  );

  // Init the map once; style loads async (CARTO, with an offline fallback).
  // biome-ignore lint/correctness/useExhaustiveDependencies: mount-only setup
  useEffect(() => {
    if (!container.current) return;
    let cancelled = false;

    loadStyle(pal.current).then(({ style, basemap }) => {
      if (cancelled || !container.current) return;
      hasBasemap.current = basemap;
      const m = new maplibregl.Map({
        container: container.current,
        style,
        attributionControl: basemap && { compact: true },
        dragRotate: false,
        canvasContextAttributes: { preserveDrawingBuffer: true },
      });
      m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
      m.fitBounds(bbox(buurten), { padding: 20, animate: false });

      m.on("style.load", () => {
        m.addSource("frames", { type: "geojson", data: work, promoteId: "buurtcode" });
        m.addLayer(
          {
            id: "fill",
            type: "fill",
            source: "frames",
            paint: {
              "fill-color": heatExpression(frames.meta.vmax),
              "fill-opacity": hasBasemap.current ? 0.78 : 0.92,
            },
          },
          firstSymbolId(m),
        );
        m.addLayer(
          {
            id: "outline",
            type: "line",
            source: "frames",
            paint: { "line-color": pal.current.baseline, "line-width": 0.4 },
          },
          firstSymbolId(m),
        );
        setReady(true);
      });

      attachTooltip(m, "fill", (f) => [
        String(f.properties.buurtnaam),
        `${fmt1(Number(f.state.v ?? 0))} W/m²`,
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

  // Repaint on frame change and after the style reloads (which drops feature state).
  useEffect(() => {
    paint(frame);
  }, [frame, paint]);

  // Playback on rAF: no drift, and the browser stops it in a background tab.
  // User-initiated, so reduced-motion still allows it.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (t: number) => {
      if (t - last >= STEP_MS) {
        last = t;
        setFrame((f) => (f + 1) % FRAME_COUNT);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const label = frameLabel(frame, frames.months);
  const cityNow = cityMean[frame] ?? 0;

  return (
    <section id="year" aria-labelledby="year-h">
      <h2 id="year-h">A year of cooling, hour by hour</h2>
      <p className="lede">
        Every hour of a typical year, as 12 months by 24 hours. The cooling season is a shape: it
        opens in April, peaks through summer afternoons, and closes in October. Pick any cell to
        send the map to that hour.
      </p>

      <figure className="figure">
        <YearCarpet
          cityMean={cityMean}
          months={frames.months}
          vmax={vmax}
          frame={frame}
          onPick={(i) => {
            setPlaying(false);
            setFrame(i);
          }}
          fmt={fmt1}
        />

        <div className="player">
          <button
            type="button"
            className="iconbtn player__play"
            onClick={() => setPlaying((p) => !p)}
            aria-pressed={playing}
          >
            {playing ? "⏸ Pause" : "▶ Play"}
          </button>
          <div className="player__label" aria-hidden="true">
            <strong>{label}</strong>
            <span className="note"> · city avg {fmt1(cityNow)} W/m²</span>
          </div>
          <input
            className="player__slider"
            type="range"
            min={0}
            max={FRAME_COUNT - 1}
            step={1}
            value={frame}
            onChange={(e) => {
              setPlaying(false);
              setFrame(Number(e.target.value));
            }}
            aria-label="Time of year and hour of day"
            aria-valuetext={`${label}, city average ${fmt1(cityNow)} watts per square metre`}
          />
        </div>

        {/* biome-ignore lint/a11y/useSemanticElements: a slippy map is not a fieldset; group labels the interactive region */}
        <div
          className="map"
          ref={container}
          role="group"
          aria-label={`Map of The Hague showing cooling intensity per neighbourhood at ${label}. Use the slider to change the time.`}
        />
        <Legend items={heatLegend(vmax, HEAT_RAMP, fmt1)} title="Cooling intensity (W/m²)" />
        <figcaption>
          Cooling intensity per neighbourhood · {label}. Carpet above shows the city average for
          every hour, on the same scale. {frames.meta.weather_years} typical year.
        </figcaption>
      </figure>
    </section>
  );
}
