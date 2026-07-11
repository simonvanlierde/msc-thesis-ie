import { gwh, ktCO2, pct } from "../lib/format";
import { SCENARIO_META } from "../lib/scenarioMeta";
import { officeHeadline } from "../lib/transform";
import type { ScenariosData } from "../lib/types";
import { Act } from "./Act";

interface Props {
  data: ScenariosData;
}

/** Act 1 — the anchor. Always the present-day (SQ) city, framed as the opening thesis:
 *  a small office stock carries an outsized share of the cooling burden. */
export function TodayHero({ data }: Props) {
  const s = data.scenarios.SQ;
  const h = officeHeadline(s.archetypes);

  return (
    <Act id="today" variant="today" eyebrow="Now · the city as it is" labelledBy="today-h">
      <h2 id="today-h" className="visually-hidden">
        Cooling in The Hague today
      </h2>
      <p className="hero__headline">
        Offices fill just <em>{pct(h.areaShare, 1, 0)}</em> of the floor area but drive{" "}
        <em>{pct(h.ghgShare, 1, 0)}</em> of the city's cooling-related greenhouse-gas emissions.
      </p>
      <p className="lede">{SCENARIO_META.SQ.blurb}</p>

      <div className="tiles">
        <div className="tile">
          <div className="tile__value">{gwh(s.totals.E_cooling_kWh)}</div>
          <div className="tile__label">Annual cooling demand</div>
        </div>
        <div className="tile">
          <div className="tile__value">{ktCO2(s.totals.GHG_emissions_total_kgCO2eq)}</div>
          <div className="tile__label">Cooling-related emissions / year</div>
        </div>
        <div className="tile">
          <div className="tile__value">{pct(h.demandShare, 1, 0)}</div>
          <div className="tile__label">Share of demand from offices</div>
        </div>
        <div className="tile">
          <div className="tile__value">{pct(h.ghgShare, 1, 0)}</div>
          <div className="tile__label">Share of emissions from offices</div>
        </div>
      </div>
    </Act>
  );
}
