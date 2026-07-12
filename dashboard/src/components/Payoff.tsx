import { useEffect, useState } from "react";
import { num } from "../lib/format";
import type { Palette } from "../lib/palette";
import { scenarioLabel as rowLabel, SCENARIO_META } from "../lib/scenarioMeta";
import type { ScenarioKey, ScenariosData, ScenarioTotals } from "../lib/types";
import { useInView } from "../lib/useInView";
import { Act } from "./Act";
import { PathSwitch } from "./PathSwitch";

interface Props {
  data: ScenariosData;
  scenario: ScenarioKey;
  onChange: (k: ScenarioKey) => void;
  palette: Palette;
}

interface Metric {
  id: string;
  title: string;
  unit: string;
  digits: number;
  value: (t: ScenarioTotals) => number;
}

const METRICS: Metric[] = [
  {
    id: "demand",
    title: "Cooling demand",
    unit: "GWh / year",
    digits: 0,
    value: (t) => t.E_cooling_kWh / 1e6,
  },
  {
    id: "electricity",
    title: "Electricity for cooling",
    unit: "GWh / year",
    digits: 0,
    value: (t) => t.electricity_kWh / 1e6,
  },
  {
    id: "ghg",
    title: "Life-cycle emissions",
    unit: "kt CO₂-eq / year",
    digits: 1,
    value: (t) => t.GHG_emissions_total_kgCO2eq / 1e6,
  },
];

const prefersReducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** Ratio to today (SQ), e.g. "1.9× today" / "0.1× today". */
function vsToday(v: number, today: number): string {
  return `${(v / today).toFixed(1)}× today`;
}

/** Counts up to `value` once `run` turns true; instant under reduced motion. Renders the
 *  final value on first paint so a non-scrolling capture (or no JS) shows the real number. */
function CountUp({ value, digits, run }: { value: number; digits: number; run: boolean }) {
  const [shown, setShown] = useState(value);
  useEffect(() => {
    if (!run) return;
    if (prefersReducedMotion()) {
      setShown(value);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const dur = 700;
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / dur);
      setShown(value * (1 - (1 - p) ** 3));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [run, value]);
  return <>{num(shown, digits)}</>;
}

/** Act 4 — the payoff. For the three metrics that diverge, the whole fan of futures on
 *  one axis with the chosen path lit in accent blue; today and 2030 sit as faint references.
 *  Length carries the story ("nearly doubles" / "a tenth of today"); the section's warm/cool
 *  tint commits to the chosen path. A sticky switcher keeps the choice in reach while the
 *  bars are on screen. Warm colour stays reserved for the heat ramp elsewhere. */
export function Payoff({ data, scenario, onChange, palette }: Props) {
  const order = data.meta.scenario_order;
  const [chartsRef, grow] = useInView<HTMLDivElement>();
  const meta = SCENARIO_META[scenario];

  return (
    <Act
      id="payoff"
      variant="payoff"
      path={scenario}
      eyebrow="2050 · the difference"
      labelledBy="payoff-h"
    >
      <h2 id="payoff-h">The difference by 2050</h2>

      {/* Sticky, so the path stays switchable while the bars are in view — the fork's choice,
          kept in reach. Its own radiogroup name so it doesn't fight the fork's native group. */}
      <PathSwitch name="payoff-path" scenario={scenario} onChange={onChange} />

      <p className="lede">
        You picked <strong>{rowLabel(scenario)}</strong> — {meta.tagline.toLowerCase()}. Here is
        where that path lands, with today and 2030 as faint reference rows for scale.
      </p>

      <div className={`payoff${grow ? " payoff--grow" : ""}`} ref={chartsRef}>
        {METRICS.map((m) => (
          <PayoffBars
            key={m.id}
            metric={m}
            order={order}
            data={data}
            active={scenario}
            grow={grow}
            palette={palette}
          />
        ))}
      </div>

      <details className="datatable">
        <summary>Data table — demand, electricity and emissions by scenario</summary>
        <table>
          <caption>
            Annual cooling demand, electricity use and life-cycle emissions, by scenario
          </caption>
          <thead>
            <tr>
              <th scope="col">Scenario</th>
              <th scope="col">Demand (GWh)</th>
              <th scope="col">Electricity (GWh)</th>
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
                  <td>{num(t.electricity_kWh / 1e6, 0)}</td>
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

/** One metric's fan of scenarios as HTML rows — real text, so the labels stay readable at
 *  any column width (the old SVG scaled its type down with the viewBox, which is why the
 *  axis text was illegible on mobile). */
function PayoffBars({
  metric,
  order,
  data,
  active,
  grow,
  palette,
}: {
  metric: Metric;
  order: ScenarioKey[];
  data: ScenariosData;
  active: ScenarioKey;
  grow: boolean;
  palette: Palette;
}) {
  const vals = order.map((k) => ({ k, v: metric.value(data.scenarios[k].totals) }));
  const max = Math.max(...vals.map((r) => r.v));
  const today = metric.value(data.scenarios.SQ.totals);
  const accent = palette.sequential[4];

  const summary = vals
    .map((r) => `${rowLabel(r.k)} ${num(r.v, metric.digits)} ${metric.unit}`)
    .join("; ");

  return (
    <figure
      className="figure payoff__chart"
      role="img"
      aria-label={`${metric.title} by scenario, in ${metric.unit}: ${summary}. Selected: ${rowLabel(active)}.`}
    >
      <div className="payoff__title">
        {metric.title} <span className="payoff__unit">· {metric.unit}</span>
      </div>
      <div className="payoff__rows" aria-hidden="true">
        {vals.map((r) => {
          const isActive = r.k === active;
          return (
            <div key={r.k} className={`payoff__row${isActive ? " payoff__row--active" : ""}`}>
              <span className="payoff__label">{rowLabel(r.k)}</span>
              <span className="payoff__track">
                <span
                  className="payoff__bar"
                  style={{
                    width: `${(r.v / max) * 100}%`,
                    background: isActive ? accent : palette.baseline,
                  }}
                />
              </span>
              <span className="payoff__value">
                <CountUp value={r.v} digits={metric.digits} run={grow} />
                {isActive && r.k !== "SQ" && (
                  <span className="payoff__ratio"> {vsToday(r.v, today)}</span>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </figure>
  );
}
