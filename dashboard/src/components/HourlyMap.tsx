import maplibregl from "maplibre-gl";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { num } from "../lib/format";
import { FRAME_COUNT, frameLabel, heatColor, heatLegend } from "../lib/frames";
import { bbox } from "../lib/geo";
import { HEAT_RAMP, type Palette } from "../lib/palette";
import type { BuurtCollection, HourlyFrames } from "../lib/types";
import { Legend } from "./Legend";

interface Props {
  buurten: BuurtCollection | null;
  frames: HourlyFrames | null;
  palette: Palette;
}

const STEP_MS = 150; // ~6.7 frames/second when playing
const fmt1 = (n: number) => num(n, 1);

export function HourlyMap({ buurten, frames, palette }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);
  const [frame, setFrame] = useState(6 * 24 + 15); // open on a July afternoon
  const [playing, setPlaying] = useState(false);

  // buurtcode -> index into a frame's value array
  const codeIndex = useMemo(() => {
    const m = new Map<string, number>();
    frames?.buurtcodes.forEach((c, i) => {
      m.set(c, i);
    });
    return m;
  }, [frames]);

  // geometry for buurten that have frame data, as a mutable working collection
  const work = useMemo(() => {
    if (!(buurten && frames)) return null;
    const features = buurten.features
      .filter((f) => codeIndex.has(f.properties.buurtcode))
      .map((f) => ({ ...f, properties: { ...f.properties, __c: palette.grid, __v: 0 } }));
    return { type: "FeatureCollection", features } as unknown as GeoJSON.FeatureCollection & {
      features: Array<{ properties: { buurtcode: string; __c: string; __v: number } }>;
    };
  }, [buurten, frames, codeIndex, palette.grid]);

  const vmax = frames?.meta.vmax ?? 1;

  // paint one frame: recolour every buurt from its cooling intensity, push to the map
  const paint = useCallback(
    (idx: number) => {
      const m = map.current;
      if (!(m && ready && work && frames)) return;
      const row = frames.frames[idx];
      for (const f of work.features) {
        const v = row[codeIndex.get(f.properties.buurtcode) ?? -1] ?? 0;
        f.properties.__v = v;
        f.properties.__c = heatColor(v, vmax, HEAT_RAMP);
      }
      (m.getSource("frames") as maplibregl.GeoJSONSource | undefined)?.setData(work);
    },
    [ready, work, frames, codeIndex, vmax],
  );

  // init map once
  // biome-ignore lint/correctness/useExhaustiveDependencies: mount-only setup
  useEffect(() => {
    if (!(container.current && buurten && frames)) return;
    const m = new maplibregl.Map({
      container: container.current,
      style: {
        version: 8,
        sources: {},
        layers: [{ id: "bg", type: "background", paint: { "background-color": palette.page } }],
      },
      attributionControl: false,
      dragRotate: false,
      canvasContextAttributes: { preserveDrawingBuffer: true },
    });
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    m.on("load", () => {
      m.addSource("frames", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      m.addLayer({
        id: "fill",
        type: "fill",
        source: "frames",
        paint: { "fill-color": ["get", "__c"], "fill-opacity": 0.92 },
      });
      m.addLayer({
        id: "outline",
        type: "line",
        source: "frames",
        paint: { "line-color": palette.baseline, "line-width": 0.4 },
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
        .setHTML(`<div class="tooltip"><strong>${p.buurtnaam}</strong>${num(p.__v, 1)} W/m²</div>`)
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

  // repaint on frame change (and once ready) and keep chrome colours in sync
  useEffect(() => {
    const m = map.current;
    if (m && ready) {
      m.setPaintProperty("bg", "background-color", palette.page);
      m.setPaintProperty("outline", "line-color", palette.baseline);
    }
    paint(frame);
  }, [frame, paint, ready, palette]);

  // playback: advance frames on a timer (user-initiated, so reduced-motion still allows it)
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setFrame((f) => (f + 1) % FRAME_COUNT), STEP_MS);
    return () => clearInterval(id);
  }, [playing]);

  const months = frames?.months ?? [];
  const label = frames ? frameLabel(frame, months) : "";
  const cityMean = frames
    ? frames.frames[frame].reduce((s, v) => s + v, 0) / frames.frames[frame].length
    : 0;

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

  return (
    <section id="year" aria-labelledby="year-h">
      <h2 id="year-h">A year of cooling, hour by hour</h2>
      <p className="lede">
        Scrub through a typical year and watch cooling demand flush across the city — pale on winter
        nights, deep red on summer afternoons. Colour is cooling intensity (W per m² of floor area),
        on a fixed scale so seasons compare directly.
      </p>

      <figure className="figure">
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
            <span className="note"> · city avg {fmt1(cityMean)} W/m²</span>
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
            aria-valuetext={`${label}, city average ${fmt1(cityMean)} watts per square metre`}
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
          Cooling intensity per neighbourhood · {label}. {frames.meta.weather_years} typical year.
        </figcaption>
      </figure>
    </section>
  );
}
