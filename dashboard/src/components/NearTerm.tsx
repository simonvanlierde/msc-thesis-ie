import { gwh, ktCO2 } from "../lib/format";
import { SCENARIO_META } from "../lib/scenarioMeta";
import type { ScenariosData } from "../lib/types";
import { Act } from "./Act";

interface Props {
  data: ScenariosData;
}

/** Signed percentage change from a→b, e.g. "−49%" / "+0%". */
function delta(a: number, b: number): string {
  const p = Math.round(((b - a) / a) * 100);
  return `${p > 0 ? "+" : p < 0 ? "−" : "±"}${Math.abs(p)}%`;
}

/** Act 2 — a short beat. By 2030 a cleaner grid pulls emissions down even as demand holds:
 *  the near future improves on its own, which sets up why 2050 is the real fork. */
export function NearTerm({ data }: Props) {
  const sq = data.scenarios.SQ.totals;
  const y30 = data.scenarios["2030"].totals;

  return (
    <Act id="near" variant="near" eyebrow="Soon · 2030" labelledBy="near-h">
      <h2 id="near-h">The near future bends down on its own</h2>
      <p className="lede">{SCENARIO_META["2030"].blurb}</p>

      <div className="deltas">
        <div className="delta">
          <div className="delta__metric">Cooling demand</div>
          <div className="delta__row">
            <span className="delta__from">{gwh(sq.E_cooling_kWh)}</span>
            <span className="delta__arrow" aria-hidden="true">
              →
            </span>
            <span className="delta__to">{gwh(y30.E_cooling_kWh)}</span>
            <span className="delta__pct delta__pct--flat">
              {delta(sq.E_cooling_kWh, y30.E_cooling_kWh)}
            </span>
          </div>
          <div className="delta__note">Roughly flat — the city hasn't grown much yet.</div>
        </div>
        <div className="delta">
          <div className="delta__metric">Cooling-related emissions</div>
          <div className="delta__row">
            <span className="delta__from">{ktCO2(sq.GHG_emissions_total_kgCO2eq)}</span>
            <span className="delta__arrow" aria-hidden="true">
              →
            </span>
            <span className="delta__to">{ktCO2(y30.GHG_emissions_total_kgCO2eq)}</span>
            <span className="delta__pct delta__pct--down">
              {delta(sq.GHG_emissions_total_kgCO2eq, y30.GHG_emissions_total_kgCO2eq)}
            </span>
          </div>
          <div className="delta__note">
            A cleaner grid (262 → 159 g CO₂/kWh) does the work, not less cooling.
          </div>
        </div>
      </div>
    </Act>
  );
}
