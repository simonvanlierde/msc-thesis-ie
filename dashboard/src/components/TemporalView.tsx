import { type LineCustomSvgLayer, type LineSeries, ResponsiveLine } from "@nivo/line";
import { useMemo, useState } from "react";
import { num } from "../lib/format";
import type { Palette } from "../lib/palette";
import type { TemporalData } from "../lib/types";

interface Props {
  temporal: TemporalData;
  palette: Palette;
}

interface Series extends LineSeries {
  id: string;
  color: string;
  data: readonly { x: string; y: number }[];
}

const USE_LABEL: Record<string, string> = { residential: "Residential", office: "Office" };
const colorForUse = (p: Palette, use: string) =>
  use === "office" ? p.use.Office : p.use.Residential;

// Two series only: label the line ends and drop the legend, so reading a line costs no
// glance away from it. Draw order (bottom to top) is default layers plus ours.
const LAYERS = ["grid", "axes", "crosshair", "lines", "points", "mesh"] as const;
const LABEL_GAP = 16;

function nivoTheme(p: Palette) {
  return {
    text: { fill: p.textSecondary, fontFamily: "inherit" },
    axis: {
      ticks: { text: { fill: p.muted }, line: { stroke: p.grid } },
      legend: { text: { fill: p.textSecondary } },
    },
    grid: { line: { stroke: p.grid } },
    crosshair: { line: { stroke: p.muted } },
    tooltip: { container: { background: "transparent", boxShadow: "none", padding: 0 } },
  };
}

function Tooltip({
  point,
}: {
  point: {
    seriesId: string | number;
    data: { xFormatted: string | number; yFormatted: string | number };
  };
}) {
  return (
    <div className="tooltip">
      <strong>{USE_LABEL[String(point.seriesId)] ?? point.seriesId}</strong>
      {point.data.xFormatted}: {point.data.yFormatted}
    </div>
  );
}

/** Series names at the right-hand line ends, nudged apart when the lines converge. */
const endLabels: LineCustomSvgLayer<Series> = ({ series }) => {
  const items = series
    .map((s) => {
      const last = s.data[s.data.length - 1];
      return { id: String(s.id), color: s.color, x: last.position.x, y: last.position.y };
    })
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < items.length; i++) {
    const overlap = items[i - 1].y + LABEL_GAP - items[i].y;
    if (overlap > 0) items[i].y += overlap;
  }
  return (
    <g>
      {items.map((it) => (
        <text
          key={it.id}
          x={it.x + 10}
          y={it.y + 4}
          fill={it.color}
          fontSize={13}
          fontWeight={600}
          fontFamily="inherit"
        >
          {USE_LABEL[it.id] ?? it.id}
        </text>
      ))}
    </g>
  );
};

/** One annotation, on the thing the eye lands on: the day's single highest point of
 *  cooling power, whichever series and season it falls in. Computed from the data, never
 *  hardcoded — the peak moves between morning (homes) and midday (offices) by season. */
function makePeakLayer(palette: Palette): LineCustomSvgLayer<Series> {
  return ({ series, innerWidth }) => {
    let peak: { x: string; y: number; px: number; py: number; id: string } | null = null;
    for (const s of series) {
      for (const d of s.data) {
        const y = Number(d.data.y);
        if (!peak || y > peak.y) {
          peak = { x: String(d.data.x), y, px: d.position.x, py: d.position.y, id: String(s.id) };
        }
      }
    }
    if (!peak) return null;

    const color = peak.id === "office" ? palette.use.Office : palette.use.Residential;
    const who = USE_LABEL[peak.id] ?? peak.id;
    const text = `${who} peak · ${peak.x}, ${num(peak.y, 2)} GW`;
    const tx = Math.min(Math.max(peak.px, 100), innerWidth - 100);
    return (
      <g pointerEvents="none">
        <circle
          cx={peak.px}
          cy={peak.py}
          r={4.5}
          fill={color}
          stroke={palette.page}
          strokeWidth={2}
        />
        <text
          x={tx}
          y={peak.py - 14}
          textAnchor="middle"
          fill={palette.textPrimary}
          stroke={palette.page}
          strokeWidth={3}
          paintOrder="stroke"
          fontSize={13}
          fontWeight={600}
          fontFamily="inherit"
        >
          {text}
        </text>
      </g>
    );
  };
}

export function TemporalView({ temporal, palette }: Props) {
  const [season, setSeason] = useState(temporal.seasons[0]);
  const uses = temporal.uses;
  const theme = nivoTheme(palette);
  const peakLayer = useMemo(() => makePeakLayer(palette), [palette]);

  const diurnal: Series[] = uses.map((use) => ({
    id: use,
    color: colorForUse(palette, use),
    data: temporal.hour_of_day.map((h) => ({
      x: `${String(h).padStart(2, "0")}:00`,
      y: temporal.diurnal_by_season[season][use][h],
    })),
  }));

  const monthly: Series[] = uses.map((use) => ({
    id: use,
    color: colorForUse(palette, use),
    data: temporal.months.map((mo, i) => ({ x: mo, y: temporal.monthly[use][i] })),
  }));

  return (
    <section id="when" aria-labelledby="when-h">
      <h2 id="when-h">When cooling is needed</h2>
      <p className="lede">
        Reconstructed from the thesis heat-balance model over {temporal.meta.weather_years} weather,
        calibrated to the published annual totals. Cooling concentrates in the warm months. Offices
        hold a steady plateau through working hours; homes swing higher, peaking in the morning and
        again in the early evening.
      </p>

      <fieldset className="segmented">
        <legend>Season (daily profile)</legend>
        <div className="segmented__row">
          {temporal.seasons.map((se) => (
            <label key={se}>
              <input
                type="radio"
                name="season"
                checked={season === se}
                onChange={() => setSeason(se)}
              />
              {se}
            </label>
          ))}
        </div>
      </fieldset>

      <figure className="figure">
        <div className="chart">
          <ResponsiveLine<Series>
            role="img"
            ariaLabel={`Average cooling power by hour of day in ${season}, residential versus office, in gigawatts.`}
            data={diurnal}
            theme={theme}
            colors={(s) => s.color}
            layers={[...LAYERS, endLabels, peakLayer]}
            margin={{ top: 20, right: 96, bottom: 50, left: 55 }}
            xScale={{ type: "point" }}
            yScale={{ type: "linear", min: 0, max: "auto" }}
            axisBottom={{
              legend: "Hour of day",
              legendOffset: 40,
              legendPosition: "middle",
              tickValues: diurnal[0].data.filter((_, i) => i % 3 === 0).map((d) => d.x),
            }}
            axisLeft={{
              legend: "Cooling power (GW)",
              legendOffset: -45,
              legendPosition: "middle",
            }}
            enablePoints={false}
            enableGridX={false}
            lineWidth={2}
            useMesh
            animate={false}
            yFormat={(v) => `${num(Number(v), 3)} GW`}
            tooltip={({ point }) => <Tooltip point={point} />}
          />
        </div>
        <figcaption>Average cooling power through the day · {season}.</figcaption>
      </figure>

      <figure className="figure">
        <div className="chart">
          <ResponsiveLine<Series>
            role="img"
            ariaLabel="Monthly cooling energy demand across the year, residential versus office, in gigawatt-hours."
            data={monthly}
            theme={theme}
            colors={(s) => s.color}
            layers={[...LAYERS, endLabels]}
            margin={{ top: 20, right: 96, bottom: 50, left: 55 }}
            xScale={{ type: "point" }}
            yScale={{ type: "linear", min: 0, max: "auto" }}
            axisBottom={{ legend: "Month", legendOffset: 40, legendPosition: "middle" }}
            axisLeft={{
              legend: "Cooling energy (GWh)",
              legendOffset: -45,
              legendPosition: "middle",
            }}
            enablePoints
            pointSize={8}
            enableGridX={false}
            lineWidth={2}
            useMesh
            animate={false}
            yFormat={(v) => `${num(Number(v), 1)} GWh`}
            tooltip={({ point }) => <Tooltip point={point} />}
          />
        </div>
        <figcaption>Cooling energy by month · typical year.</figcaption>
      </figure>

      <details className="datatable">
        <summary>Data table — monthly cooling energy (GWh)</summary>
        <table>
          <caption>Cooling energy by month and building use (GWh)</caption>
          <thead>
            <tr>
              <th scope="col">Month</th>
              {uses.map((u) => (
                <th scope="col" key={u}>
                  {USE_LABEL[u] ?? u}
                </th>
              ))}
              <th scope="col">Total</th>
            </tr>
          </thead>
          <tbody>
            {temporal.months.map((mo, i) => (
              <tr key={mo}>
                <td>{mo}</td>
                {uses.map((u) => (
                  <td key={u}>{num(temporal.monthly[u][i], 1)}</td>
                ))}
                <td>{num(temporal.monthly.total[i], 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}
