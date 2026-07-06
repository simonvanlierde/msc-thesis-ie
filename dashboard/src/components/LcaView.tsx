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

  // Each impact category split into office vs residential share (%), 100%-stacked.
  const byUse = rollup(s.archetypes, "use");
  const catData = CATEGORIES.map((c) => {
    const res = byUse.find((r) => r.key === "Residential")?.[c.key] ?? 0;
    const off = byUse.find((r) => r.key === "Office")?.[c.key] ?? 0;
    const tot = res + off || 1;
    return { category: c.label, Residential: (res / tot) * 100, Office: (off / tot) * 100 };
  });

  const stageLegend = STAGE_ORDER.map((st) => ({
    color: palette.stage[st],
    label: stageLabels[st],
  }));
  const useLegend = uses.map((u) => ({ color: palette.use[u], label: u }));

  return (
    <section id="impact" aria-labelledby="impact-h">
      <h2 id="impact-h">Life-cycle environmental impact</h2>
      <p className="lede">
        Greenhouse-gas emissions across the full life cycle of cooling — the electricity used to run
        it, refrigerant leaks, and making and disposing of the equipment. Operational electricity
        dominates, and offices carry most of it.
      </p>

      <div style={{ display: "grid", gap: "1.5rem", gridTemplateColumns: "1fr" }}>
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
              borderRadius={2}
              enableLabel={false}
              axisBottom={{ legend: "Building use", legendOffset: 40, legendPosition: "middle" }}
              axisLeft={{
                legend: "Emissions (kt CO₂-eq)",
                legendOffset: -50,
                legendPosition: "middle",
              }}
              animate={false}
              tooltip={({ id, value }) => (
                <div className="tooltip">
                  <strong>{stageLabels[id as keyof GhgStages]}</strong>
                  {num(value, 1)} kt CO₂-eq
                </div>
              )}
            />
          </div>
          <Legend items={stageLegend} title="Life-cycle stage" />
          <figcaption>GHG emissions by life-cycle stage · scenario {scenario}.</figcaption>
        </figure>

        <figure className="figure">
          <div className="chart">
            <ResponsiveBar
              role="img"
              ariaLabel="Share of each environmental impact category attributable to residential versus office buildings, as a percentage."
              data={catData}
              theme={theme}
              keys={["Residential", "Office"]}
              indexBy="category"
              layout="horizontal"
              margin={{ top: 10, right: 20, bottom: 50, left: 130 }}
              padding={0.4}
              colors={(d) => palette.use[d.id as "Residential" | "Office"]}
              innerPadding={2}
              valueScale={{ type: "linear", min: 0, max: 100 }}
              enableLabel={false}
              axisBottom={{
                legend: "Share of impact (%)",
                legendOffset: 40,
                legendPosition: "middle",
              }}
              animate={false}
              tooltip={({ id, value, indexValue }) => (
                <div className="tooltip">
                  <strong>{indexValue}</strong>
                  {id}: {value.toFixed(0)}%
                </div>
              )}
            />
          </div>
          <Legend items={useLegend} title="Building use" />
          <figcaption>
            Share of each impact category by building use · scenario {scenario}.
          </figcaption>
        </figure>
      </div>

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
