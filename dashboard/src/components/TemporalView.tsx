import { ResponsiveLine } from "@nivo/line";
import { useState } from "react";
import { num } from "../lib/format";
import type { Palette } from "../lib/palette";
import type { TemporalData } from "../lib/types";
import { Legend } from "./Legend";

interface Props {
  temporal: TemporalData;
  palette: Palette;
}

const USE_LABEL: Record<string, string> = { residential: "Residential", office: "Office" };
const colorForUse = (p: Palette, use: string) =>
  use === "office" ? p.use.Office : p.use.Residential;

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

export function TemporalView({ temporal, palette }: Props) {
  const [season, setSeason] = useState(temporal.seasons[0]);
  const uses = temporal.uses;
  const theme = nivoTheme(palette);

  const diurnal = uses.map((use) => ({
    id: use,
    color: colorForUse(palette, use),
    data: temporal.hour_of_day.map((h) => ({
      x: `${String(h).padStart(2, "0")}:00`,
      y: temporal.diurnal_by_season[season][use][h],
    })),
  }));

  const monthly = uses.map((use) => ({
    id: use,
    color: colorForUse(palette, use),
    data: temporal.months.map((mo, i) => ({ x: mo, y: temporal.monthly[use][i] })),
  }));

  const legendItems = uses.map((u) => ({
    color: colorForUse(palette, u),
    label: USE_LABEL[u] ?? u,
  }));

  return (
    <section id="when" aria-labelledby="when-h">
      <h2 id="when-h">When cooling is needed</h2>
      <p className="lede">
        Reconstructed from the thesis heat-balance model over {temporal.meta.weather_years} weather,
        calibrated to the published annual totals. Cooling concentrates on summer afternoons —
        offices peak sharply during working hours, homes later in the evening.
      </p>

      <fieldset className="segmented" style={{ marginBottom: "0.9rem" }}>
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

      <div style={{ display: "grid", gap: "1.5rem", gridTemplateColumns: "1fr" }}>
        <figure className="figure">
          <div className="chart">
            <ResponsiveLine
              role="img"
              ariaLabel={`Average cooling power by hour of day in ${season}, residential versus office, in gigawatts.`}
              data={diurnal}
              theme={theme}
              colors={(s) => s.color as string}
              margin={{ top: 10, right: 20, bottom: 50, left: 55 }}
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
          <Legend items={legendItems} title="Building use" />
          <figcaption>Average cooling power through the day · {season}.</figcaption>
        </figure>

        <figure className="figure">
          <div className="chart">
            <ResponsiveLine
              role="img"
              ariaLabel="Monthly cooling energy demand across the year, residential versus office, in gigawatt-hours."
              data={monthly}
              theme={theme}
              colors={(s) => s.color as string}
              margin={{ top: 10, right: 20, bottom: 50, left: 55 }}
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
          <Legend items={legendItems} title="Building use" />
          <figcaption>Cooling energy by month · typical year.</figcaption>
        </figure>
      </div>

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
