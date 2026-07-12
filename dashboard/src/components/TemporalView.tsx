import { type LineCustomSvgLayer, type LineSeries, ResponsiveLine } from "@nivo/line";
import { useMemo, useState } from "react";
import { num } from "../lib/format";
import type { Palette } from "../lib/palette";
import { scenarioLabel as rowLabel } from "../lib/scenarioMeta";
import type { ScenarioKey, TemporalData } from "../lib/types";
import { Legend } from "./Legend";
import { Segmented } from "./Segmented";

interface Props {
  temporal: TemporalData;
  scenario: ScenarioKey;
  /** kWh electricity per kWh cooling demand, per use (PUE × market penetration),
   *  for the selected scenario. */
  elec: { residential: number; office: number };
  palette: Palette;
}

interface Series extends LineSeries {
  id: string;
  color: string;
  data: readonly { x: string; y: number }[];
}

type Quantity = "cooling" | "electricity";

const QUANTITY_OPTIONS: { value: Quantity; label: string }[] = [
  { value: "cooling", label: "Cooling demand" },
  { value: "electricity", label: "Electricity use" },
];

const USE_LABEL: Record<string, string> = { residential: "Residential", office: "Office" };
const colorForUse = (p: Palette, use: string) =>
  use === "office" ? p.use.Office : p.use.Residential;

// Direct end-labels supplement the legend below the chart; they never replace it, because
// colour-matching alone is not a dependable identity channel. Draw order (bottom to top) is
// the default layers plus ours.
const LAYERS = ["grid", "axes", "crosshair", "lines", "points", "mesh"] as const;
const LABEL_GAP = 16;
const LEADER_X = 12; // horizontal run of the leader line before it meets the label
// The right margin has to clear LEADER_X plus the widest series name ("Residential"), or
// nivo clips the end-labels. Widen the two together.
const MARGIN = { top: 20, right: 124, bottom: 50, left: 55 };

/** "2023-06-09T14:00" → "9 Jun 14:00" (unique per hour, so it can be nivo's point x). */
function hourLabel(iso: string, months: string[]): string {
  return `${Number(iso.slice(8, 10))} ${months[Number(iso.slice(5, 7)) - 1]} ${iso.slice(11, 16)}`;
}

/** "9–15 June 2023", from the heatwave's first and last ISO hours. */
function weekLabel(dates: string[], months: string[]): string {
  const first = dates[0];
  const last = dates[dates.length - 1];
  return `${Number(first.slice(8, 10))}–${Number(last.slice(8, 10))} ${months[Number(first.slice(5, 7)) - 1]} ${first.slice(0, 4)}`;
}

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

/**
 * Series names at the right-hand line ends. When the lines converge, a nudged label floats
 * free of the line it names, so each label keeps a leader line back to its own line-end and
 * a ringed dot anchoring it there. The text itself wears an ink token, never the series
 * colour — a mid-blue or orange word on the surface is a contrast problem, and the coloured
 * dot beside it already carries the identity.
 */
function makeEndLabels(palette: Palette): LineCustomSvgLayer<Series> {
  return ({ series }) => {
    const items = series
      .map((s) => {
        const last = s.data[s.data.length - 1];
        return {
          id: String(s.id),
          color: s.color,
          x: last.position.x,
          y: last.position.y,
          labelY: last.position.y,
        };
      })
      .sort((a, b) => a.labelY - b.labelY);
    for (let i = 1; i < items.length; i++) {
      const overlap = items[i - 1].labelY + LABEL_GAP - items[i].labelY;
      if (overlap > 0) items[i].labelY += overlap;
    }
    return (
      <g pointerEvents="none">
        {items.map((it) => (
          <g key={it.id}>
            <polyline
              points={`${it.x},${it.y} ${it.x + LEADER_X},${it.y} ${it.x + LEADER_X},${it.labelY}`}
              fill="none"
              stroke={it.color}
              strokeWidth={1}
            />
            <circle
              cx={it.x}
              cy={it.y}
              r={4}
              fill={it.color}
              stroke={palette.page}
              strokeWidth={2}
            />
            <text
              x={it.x + LEADER_X + 6}
              y={it.labelY + 4}
              fill={palette.textPrimary}
              fontSize={13}
              fontWeight={600}
              fontFamily="inherit"
            >
              {USE_LABEL[it.id] ?? it.id}
            </text>
          </g>
        ))}
      </g>
    );
  };
}

/** One annotation, on the thing the eye lands on: the week's single highest point of
 *  power, whichever series it falls in. Computed from the data, never hardcoded. */
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
    const tx = Math.min(Math.max(peak.px, 130), innerWidth - 130);
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

export function TemporalView({ temporal, scenario, elec, palette }: Props) {
  const uses = temporal.uses;
  const profiles = temporal.by_scenario[scenario];
  const theme = nivoTheme(palette);
  const [quantity, setQuantity] = useState<Quantity>("cooling");
  const peakLayer = useMemo(() => makePeakLayer(palette), [palette]);
  const endLabels = useMemo(() => makeEndLabels(palette), [palette]);
  const useLegend = uses.map((u) => ({
    color: colorForUse(palette, u),
    label: USE_LABEL[u] ?? u,
  }));

  // Electricity = cooling × (PUE × market penetration), a constant per use in the model —
  // so the toggle rescales each series rather than needing a second dataset.
  const factor = (use: string) =>
    quantity === "electricity" ? (use === "office" ? elec.office : elec.residential) : 1;
  const isElec = quantity === "electricity";
  const powerLabel = isElec ? "Electric power (GW)" : "Cooling power (GW)";
  const energyLabel = isElec ? "Electricity (GWh)" : "Cooling energy (GWh)";

  const monthly: Series[] = uses.map((use) => ({
    id: use,
    color: colorForUse(palette, use),
    data: temporal.months.map((mo, i) => ({ x: mo, y: profiles.monthly[use][i] * factor(use) })),
  }));

  // The hottest week of the weather record, hour by hour — the stress case the summer
  // builds towards, and what the "sized for 98% of hours" systems must ride out.
  const hw = profiles.heatwave;
  const hwWeek = weekLabel(hw.dates, temporal.months);
  const heatwave: Series[] = uses.map((use) => ({
    id: use,
    color: colorForUse(palette, use),
    data: hw.dates.map((d, i) => ({
      x: hourLabel(d, temporal.months),
      y: hw.series[use][i] * factor(use),
    })),
  }));
  // One tick per midnight; the tick label drops the redundant "00:00".
  const hwTicks = heatwave[0].data.filter((_, i) => i % 24 === 0).map((d) => d.x);

  return (
    <section id="when" aria-labelledby="when-h">
      <h2 id="when-h">When cooling is needed</h2>
      <p className="lede">
        Hourly output of the thesis heat-balance model, run over {temporal.meta.weather_years}{" "}
        weather with the {rowLabel(scenario)} scenario's climate and comfort assumptions and
        calibrated to its annual totals. Cooling concentrates in the warm months — and within them, in a few extreme
        days: the second chart zooms into the hottest week of the record, {hwWeek}.
      </p>

      <div className="viewctl">
        <Segmented
          name="quantity"
          legend="Shown in the profiles"
          options={QUANTITY_OPTIONS}
          value={quantity}
          onChange={setQuantity}
        />
        <p className="scope-note">
          Electricity is what installed equipment draws to meet the demand: cooling demand ÷
          efficiency (SEER), scaled by the share of buildings that have cooling at all — both
          improve along the 2050 paths. Today's buildings set the profiles' shape; each scenario's
          magnitude is calibrated to its citywide totals, including the projected growth of the
          building stock.
        </p>
      </div>

      <figure className="figure">
        <div className="chart">
          <ResponsiveLine<Series>
            role="img"
            ariaLabel={`Monthly ${isElec ? "electricity use for cooling" : "cooling energy demand"} across the year, residential versus office, in gigawatt-hours.`}
            data={monthly}
            theme={theme}
            colors={(s) => s.color}
            layers={[...LAYERS, endLabels]}
            margin={MARGIN}
            xScale={{ type: "point" }}
            yScale={{ type: "linear", min: 0, max: "auto" }}
            axisBottom={{ legend: "Month", legendOffset: 40, legendPosition: "middle" }}
            axisLeft={{
              legend: energyLabel,
              legendOffset: -45,
              legendPosition: "middle",
            }}
            enablePoints
            pointSize={8}
            // 2px surface ring, so a marker stays legible where the two lines cross.
            pointBorderWidth={2}
            pointBorderColor={palette.page}
            enableGridX={false}
            lineWidth={2}
            useMesh
            animate={false}
            yFormat={(v) => `${num(Number(v), 1)} GWh`}
            tooltip={({ point }) => <Tooltip point={point} />}
          />
        </div>
        <Legend items={useLegend} title="Building use" />
        <figcaption>
          {isElec ? "Electricity for cooling" : "Cooling energy"} by month · typical year ·{" "}
          {rowLabel(scenario)}.
        </figcaption>
      </figure>

      <figure className="figure">
        <div className="chart">
          <ResponsiveLine<Series>
            role="img"
            ariaLabel={`Hourly ${isElec ? "electric" : "cooling"} power through the hottest week of the weather record, ${hwWeek}, residential versus office, in gigawatts.`}
            data={heatwave}
            theme={theme}
            colors={(s) => s.color}
            layers={[...LAYERS, endLabels, peakLayer]}
            margin={MARGIN}
            xScale={{ type: "point" }}
            yScale={{ type: "linear", min: 0, max: "auto" }}
            axisBottom={{
              legend: `Heatwave week · ${hwWeek}`,
              legendOffset: 40,
              legendPosition: "middle",
              tickValues: hwTicks,
              format: (v) => String(v).replace(" 00:00", ""),
            }}
            axisLeft={{
              legend: powerLabel,
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
        <Legend items={useLegend} title="Building use" />
        <figcaption>
          Hourly {isElec ? "electric" : "cooling"} power · heatwave of {hwWeek} ·{" "}
          {rowLabel(scenario)}.
        </figcaption>
      </figure>

      <details className="datatable">
        <summary>Data table — monthly {isElec ? "electricity" : "cooling energy"} (GWh)</summary>
        <table>
          <caption>
            {isElec ? "Electricity for cooling" : "Cooling energy"} by month and building use (GWh)
            · {rowLabel(scenario)}
          </caption>
          <thead>
            <tr>
              <th scope="col">Month</th>
              {uses.map((u) => (
                <th scope="col" key={u}>
                  {USE_LABEL[u] ?? u}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {temporal.months.map((mo, i) => (
              <tr key={mo}>
                <td>{mo}</td>
                {uses.map((u) => (
                  <td key={u}>{num(profiles.monthly[u][i] * factor(u), 1)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}
