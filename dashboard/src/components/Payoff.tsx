import { num } from "../lib/format";
import type { Palette } from "../lib/palette";
import { SCENARIO_META } from "../lib/scenarioMeta";
import type { ScenarioKey, ScenariosData, ScenarioTotals } from "../lib/types";
import { Act } from "./Act";

interface Props {
  data: ScenariosData;
  scenario: ScenarioKey;
  palette: Palette;
}

interface Metric {
  id: string;
  title: string;
  unit: string;
  noun: string;
  digits: number;
  value: (t: ScenarioTotals) => number;
}

const METRICS: Metric[] = [
  {
    id: "demand",
    title: "Cooling demand",
    unit: "GWh / year",
    noun: "demand",
    digits: 0,
    value: (t) => t.E_cooling_kWh / 1e6,
  },
  {
    id: "ghg",
    title: "Life-cycle emissions",
    unit: "kt CO₂-eq / year",
    noun: "emissions",
    digits: 1,
    value: (t) => t.GHG_emissions_total_kgCO2eq / 1e6,
  },
];

/** "Today" / "2030" / "2050 Low" — names a scenario for a chart row. */
function rowLabel(k: ScenarioKey): string {
  const m = SCENARIO_META[k];
  return m.kind === "future" ? `2050 ${m.short}` : m.short;
}

/** Ratio to today (SQ), e.g. "1.9× today" / "0.1× today". */
function vsToday(v: number, today: number): string {
  return `${(v / today).toFixed(1)}× today`;
}

/** Act 4 — the payoff. For the two metrics that diverge most, the whole fan of futures on
 *  one axis with the chosen path lit in accent blue; today and 2030 sit as faint references.
 *  Length carries the story ("nearly doubles" / "a tenth of today"); the section's warm/cool
 *  tint commits to the chosen path. Warm colour stays reserved for the heat ramp elsewhere. */
export function Payoff({ data, scenario, palette }: Props) {
  const order = data.meta.scenario_order;

  return (
    <Act
      id="payoff"
      variant="payoff"
      path={scenario}
      eyebrow="2050 · the difference"
      labelledBy="payoff-h"
    >
      <h2 id="payoff-h">The difference by 2050</h2>
      <p className="lede">
        You picked <strong>{rowLabel(scenario)}</strong> —{" "}
        {SCENARIO_META[scenario].tagline.toLowerCase()}. Here is where that path lands, against
        today and 2030 for scale.
      </p>

      <div className="payoff">
        {METRICS.map((m) => (
          <PayoffBars
            key={m.id}
            metric={m}
            order={order}
            data={data}
            active={scenario}
            palette={palette}
          />
        ))}
      </div>

      <details className="datatable">
        <summary>Data table — demand and emissions by scenario</summary>
        <table>
          <caption>Annual cooling demand and life-cycle emissions, by scenario</caption>
          <thead>
            <tr>
              <th scope="col">Scenario</th>
              <th scope="col">Demand (GWh)</th>
              <th scope="col">Emissions (kt CO₂-eq)</th>
            </tr>
          </thead>
          <tbody>
            {order.map((k) => {
              const t = data.scenarios[k].totals;
              return (
                <tr key={k}>
                  <th scope="row">{rowLabel(k)}</th>
                  <td>{num(t.E_cooling_kWh / 1e6, 0)}</td>
                  <td>{num(t.GHG_emissions_total_kgCO2eq / 1e6, 1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </details>
    </Act>
  );
}

const W = 1000;
const GUTTER_L = 150;
const PLOT_R = 195; // room for the value + "× today" on the active row (worst case: the longest bar)
const ROW_H = 46;
const TOP = 12;
const BAR_H = 20;

function PayoffBars({
  metric,
  order,
  data,
  active,
  palette,
}: {
  metric: Metric;
  order: ScenarioKey[];
  data: ScenariosData;
  active: ScenarioKey;
  palette: Palette;
}) {
  const vals = order.map((k) => ({ k, v: metric.value(data.scenarios[k].totals) }));
  const max = Math.max(...vals.map((r) => r.v));
  const today = metric.value(data.scenarios.SQ.totals);
  const accent = palette.sequential[4];
  const H = TOP + order.length * ROW_H + 4;
  const x = (v: number) => GUTTER_L + (v / max) * (W - GUTTER_L - PLOT_R);
  const rowY = (i: number) => TOP + i * ROW_H + ROW_H / 2;

  const summary = vals
    .map((r) => `${rowLabel(r.k)} ${num(r.v, metric.digits)} ${metric.unit}`)
    .join("; ");

  return (
    <figure className="figure payoff__chart">
      <div className="payoff__title">
        {metric.title} <span className="payoff__unit">· {metric.unit}</span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${metric.title} by scenario, in ${metric.unit}: ${summary}. Selected: ${rowLabel(active)}.`}
      >
        {vals.map((r, i) => {
          const isActive = r.k === active;
          return (
            <g key={r.k}>
              <text
                className={`payoff__label${isActive ? " payoff__label--active" : ""}`}
                x={GUTTER_L - 14}
                y={rowY(i) + 5}
                textAnchor="end"
              >
                {rowLabel(r.k)}
              </text>
              <line className="payoff__track" x1={x(0)} x2={x(max)} y1={rowY(i)} y2={rowY(i)} />
              <rect
                x={x(0)}
                y={rowY(i) - BAR_H / 2}
                width={Math.max(0, x(r.v) - x(0))}
                height={BAR_H}
                rx={4}
                fill={isActive ? accent : palette.baseline}
              />
              <text
                className={`payoff__value${isActive ? " payoff__value--active" : ""}`}
                x={x(r.v) + 10}
                y={rowY(i) + 5}
              >
                {num(r.v, metric.digits)}
                {isActive && r.k !== "SQ" && (
                  <tspan className="payoff__ratio"> · {vsToday(r.v, today)}</tspan>
                )}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
