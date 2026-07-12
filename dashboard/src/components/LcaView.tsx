import { num, pct } from "../lib/format";
import type { Palette } from "../lib/palette";
import { scenarioLabel as rowLabel } from "../lib/scenarioMeta";
import { STAGE_ORDER } from "../lib/transform";
import type { ScenarioKey, ScenariosData } from "../lib/types";
import { Legend } from "./Legend";

interface Props {
  data: ScenariosData;
  scenario: ScenarioKey;
  palette: Palette;
}

/**
 * Life-cycle GHG by stage, one stacked bar per scenario — the comparison that matters
 * down here: how the chosen path's climate impact is composed, next to today and the
 * other futures. HTML rows (not SVG), so the labels hold text size on every screen.
 */
export function LcaView({ data, scenario, palette }: Props) {
  const order = data.meta.scenario_order;
  const stageLabels = data.meta.ghg_stages;

  const rows = order.map((k) => {
    const stages = data.scenarios[k].lca_by_stage;
    const total = STAGE_ORDER.reduce((t, st) => t + stages[st], 0) / 1e6; // kt
    return { k, stages, total };
  });
  const max = Math.max(...rows.map((r) => r.total));

  const stageLegend = STAGE_ORDER.map((st) => ({
    color: palette.stage[st],
    label: stageLabels[st],
  }));

  const summary = rows.map((r) => `${rowLabel(r.k)} ${num(r.total, 1)} kt CO₂-eq`).join("; ");

  return (
    <section id="impact" aria-labelledby="impact-h">
      <h2 id="impact-h">Life-cycle climate impact</h2>
      <p className="lede">
        Greenhouse-gas emissions across the full life cycle of cooling — the electricity used to run
        it, refrigerant leaks, and making and disposing of the equipment — for every path on one
        scale. Operational electricity dominates today; on the cleaner-grid paths the equipment
        itself becomes the bigger share. The thesis also assessed resource-depletion impacts; this
        page shows climate only.
      </p>

      <figure
        className="figure"
        role="img"
        aria-label={`Life-cycle greenhouse-gas emissions by scenario, in kilotonnes CO2-equivalent: ${summary}. Selected: ${rowLabel(scenario)}.`}
      >
        <div className="lca__rows" aria-hidden="true">
          {rows.map((r) => {
            const isActive = r.k === scenario;
            return (
              <div key={r.k} className={`lca__row${isActive ? " lca__row--active" : ""}`}>
                <span className="lca__label">
                  {rowLabel(r.k)}
                  {isActive && <span className="lca__chosen"> · your path</span>}
                </span>
                <span className="lca__track">
                  <span className="lca__stack" style={{ width: `${(r.total / max) * 100}%` }}>
                    {STAGE_ORDER.map((st) => (
                      <span
                        key={st}
                        className="lca__seg"
                        style={{ flexGrow: r.stages[st], background: palette.stage[st] }}
                        title={`${stageLabels[st]} — ${num(r.stages[st] / 1e6, 1)} kt CO₂-eq (${pct(r.stages[st] / 1e6, r.total, 0)})`}
                      />
                    ))}
                  </span>
                </span>
                <span className="lca__value">{num(r.total, 1)}</span>
              </div>
            );
          })}
        </div>
        <Legend items={stageLegend} title="Life-cycle stage" />
        <figcaption>
          Life-cycle GHG emissions by stage and scenario, in kt CO₂-eq per year. Hover a segment for
          its stage and share.
        </figcaption>
      </figure>

      <details className="datatable">
        <summary>Data table — emissions by life-cycle stage and scenario</summary>
        <table>
          <caption>Life-cycle GHG emissions by stage (kt CO₂-eq per year)</caption>
          <thead>
            <tr>
              <th scope="col">Scenario</th>
              {STAGE_ORDER.map((st) => (
                <th scope="col" key={st}>
                  {stageLabels[st]}
                </th>
              ))}
              <th scope="col">Total</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.k}>
                <th scope="row">{rowLabel(r.k)}</th>
                {STAGE_ORDER.map((st) => (
                  <td key={st}>{num(r.stages[st] / 1e6, 1)}</td>
                ))}
                <td>{num(r.total, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}
