import { ResponsiveBar } from "@nivo/bar";
import { num, pct } from "../lib/format";
import type { Palette } from "../lib/palette";
import { rollup, STAGE_ORDER } from "../lib/transform";
import type { GhgStages, ScenarioKey, ScenariosData } from "../lib/types";
import { Legend } from "./Legend";

interface Props {
  data: ScenariosData;
  scenario: ScenarioKey;
  palette: Palette;
}

function nivoTheme(p: Palette) {
  return {
    text: { fill: p.textSecondary, fontFamily: "inherit" },
    axis: {
      ticks: { text: { fill: p.muted }, line: { stroke: p.grid } },
      legend: { text: { fill: p.textSecondary } },
    },
    grid: { line: { stroke: p.grid } },
    tooltip: { container: { background: "transparent", boxShadow: "none", padding: 0 } },
  };
}

const CATEGORIES = [
  { key: "GHG_emissions_total_kgCO2eq" as const, label: "Climate change" },
  { key: "ADP_kgSbeq" as const, label: "Resource depletion" },
  { key: "CSI_kgSieq" as const, label: "Crustal scarcity" },
];

export function LcaView({ data, scenario, palette }: Props) {
  const s = data.scenarios[scenario];
  const theme = nivoTheme(palette);
  const stageLabels = data.meta.ghg_stages;

  // GHG by life-cycle stage, per building use (kt CO2-eq), stacked.
  const uses = ["Residential", "Office"] as const;
  const stageData = uses.map((use) => {
    const rows = s.archetypes.filter((a) => a.use === use);
    const sums = STAGE_ORDER.reduce(
      (o, st) => {
        o[st] = rows.reduce((t, a) => t + a.ghg[st], 0) / 1e6; // kt
        return o;
      },
      {} as Record<keyof GhgStages, number>,
    );
    return { use, ...sums };
  });
  const barTotal = new Map(
    stageData.map((d) => [d.use as string, STAGE_ORDER.reduce((t, st) => t + d[st], 0)]),
  );

  // Office's share of each impact category.
  const byUse = rollup(s.archetypes, "use");
  const shares = CATEGORIES.map((c) => {
    const res = byUse.find((r) => r.key === "Residential")?.[c.key] ?? 0;
    const off = byUse.find((r) => r.key === "Office")?.[c.key] ?? 0;
    return { label: c.label, share: (off / (res + off || 1)) * 100 };
  });

  const stageLegend = STAGE_ORDER.map((st) => ({
    color: palette.stage[st],
    label: stageLabels[st],
  }));

  return (
    <section id="impact" aria-labelledby="impact-h">
      <h2 id="impact-h">Life-cycle environmental impact</h2>
      <p className="lede">
        Greenhouse-gas emissions across the full life cycle of cooling — the electricity used to run
        it, refrigerant leaks, and making and disposing of the equipment. Operational electricity
        dominates, and offices carry most of it.
      </p>

      <figure className="figure">
        <div className="chart">
          <ResponsiveBar
            role="img"
            ariaLabel="Greenhouse-gas emissions by life-cycle stage, stacked, for residential versus office buildings, in kilotonnes CO2-equivalent."
            data={stageData}
            theme={theme}
            keys={STAGE_ORDER as unknown as string[]}
            indexBy="use"
            margin={{ top: 10, right: 20, bottom: 50, left: 60 }}
            padding={0.35}
            colors={(d) => palette.stage[d.id as keyof GhgStages]}
            innerPadding={2}
            // Only the dominant stage is tall enough to carry a label — which is the point
            // of the chart, so the reader gets it without decoding the stack.
            label={(d) => pct(d.value ?? 0, barTotal.get(String(d.indexValue)) ?? 1, 0)}
            labelSkipHeight={36}
            labelTextColor="#fff"
            borderRadius={2}
            axisBottom={{ legend: "Building use", legendOffset: 40, legendPosition: "middle" }}
            axisLeft={{
              legend: "Emissions (kt CO₂-eq)",
              legendOffset: -50,
              legendPosition: "middle",
            }}
            animate={false}
            tooltip={({ id, value, indexValue }) => (
              <div className="tooltip">
                <strong>{stageLabels[id as keyof GhgStages]}</strong>
                {num(value, 1)} kt CO₂-eq · {pct(value, barTotal.get(String(indexValue)) ?? 1, 0)}{" "}
                of {indexValue}
              </div>
            )}
          />
        </div>
        <Legend items={stageLegend} title="Life-cycle stage" />
        <figcaption>
          GHG emissions by life-cycle stage · scenario {scenario}. Labels show each stage's share of
          its own bar.
        </figcaption>
      </figure>

      <figure className="figure">
        <OfficeShareDots rows={shares} color={palette.use.Office} />
        <figcaption>
          Office share of each impact category · scenario {scenario}. Offices hold{" "}
          {pct(
            byUse.find((r) => r.key === "Office")?.floor_area_m2 ?? 0,
            byUse.reduce((t, r) => t + r.floor_area_m2, 0),
            0,
          )}{" "}
          of the floor area; the dashed rule is an even split.
        </figcaption>
      </figure>

      <details className="datatable">
        <summary>Data table — impact totals by building use</summary>
        <table>
          <caption>Environmental impact by building use · scenario {scenario}</caption>
          <thead>
            <tr>
              <th scope="col">Impact category</th>
              <th scope="col">Residential</th>
              <th scope="col">Office</th>
              <th scope="col">Office share</th>
            </tr>
          </thead>
          <tbody>
            {CATEGORIES.map((c) => {
              const res = byUse.find((r) => r.key === "Residential")?.[c.key] ?? 0;
              const off = byUse.find((r) => r.key === "Office")?.[c.key] ?? 0;
              return (
                <tr key={c.key}>
                  <td>
                    {c.label} ({data.meta.categories[c.key]?.unit ?? ""})
                  </td>
                  <td>{num(res, 1)}</td>
                  <td>{num(off, 1)}</td>
                  <td>{pct(off, res + off, 0)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </details>
    </section>
  );
}

const W = 1000;
const ROW_H = 52;
const TOP = 14;
const AXIS_H = 30;
const GUTTER_L = 210;
const PLOT_R = 60; // room for the value label right of the dot
const TICKS = [0, 25, 50, 75, 100];

/**
 * Three shares, three dots. As 100%-stacked bars these read as three near-identical
 * blocks; on a common axis against an even-split rule, the 10-point spread between them is
 * the first thing you see. The bar from parity to the dot is the excess.
 */
function OfficeShareDots({
  rows,
  color,
}: {
  rows: { label: string; share: number }[];
  color: string;
}) {
  const H = TOP + rows.length * ROW_H + AXIS_H;
  const x = (p: number) => GUTTER_L + (p / 100) * (W - GUTTER_L - PLOT_R);
  const rowY = (i: number) => TOP + i * ROW_H + ROW_H / 2;

  return (
    <div className="dotplot">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Office share of each impact category: ${rows.map((r) => `${r.label} ${r.share.toFixed(0)} percent`).join(", ")}. An even split would be 50 percent.`}
      >
        {TICKS.map((t) => (
          <text
            className="dotplot__axis"
            key={t}
            x={x(t)}
            y={H - 8}
            textAnchor={t === 0 ? "start" : t === 100 ? "end" : "middle"}
          >
            {t}%
          </text>
        ))}
        <line className="dotplot__ref" x1={x(50)} x2={x(50)} y1={TOP} y2={H - AXIS_H} />

        {rows.map((r, i) => (
          <g key={r.label}>
            <line className="dotplot__rule" x1={x(0)} x2={x(100)} y1={rowY(i)} y2={rowY(i)} />
            <text className="dotplot__label" x={GUTTER_L - 16} y={rowY(i) + 5} textAnchor="end">
              {r.label}
            </text>
            <line
              x1={x(50)}
              x2={x(r.share)}
              y1={rowY(i)}
              y2={rowY(i)}
              stroke={color}
              strokeWidth={4}
            />
            <circle cx={x(r.share)} cy={rowY(i)} r={8} fill={color} />
            <text className="dotplot__value" x={x(r.share) + 16} y={rowY(i) + 5}>
              {r.share.toFixed(0)}%
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
