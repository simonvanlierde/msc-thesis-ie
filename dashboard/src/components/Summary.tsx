import { gwh, ktCO2, pct } from "../lib/format";
import { officeHeadline } from "../lib/transform";
import type { ScenarioKey, ScenariosData } from "../lib/types";

interface Props {
  data: ScenariosData;
  scenario: ScenarioKey;
}

/** Plain-language framing so a non-expert grasps the point without reading a chart. */
export function Summary({ data, scenario }: Props) {
  const s = data.scenarios[scenario];
  const h = officeHeadline(s.archetypes);

  return (
    <section id="overview" className="summary" aria-labelledby="overview-h">
      <h2 id="overview-h" style={{ position: "absolute", left: "-999px" }}>
        Overview
      </h2>
      <p className="note">{s.label}</p>
      <p className="summary__headline">
        Offices fill just <em>{pct(h.areaShare, 1, 0)}</em> of the floor area but drive{" "}
        <em>{pct(h.ghgShare, 1, 0)}</em> of the city's cooling-related greenhouse-gas emissions.
      </p>
      <p className="lede">
        This dashboard maps how much cooling the buildings of The Hague need, when that demand peaks
        through the day and year, and what it costs the climate and material resources over the full
        life cycle of the cooling equipment. A small, energy-intensive office stock carries an
        outsized share of the burden — and under warmer futures, total cooling demand roughly
        doubles.
      </p>

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
    </section>
  );
}
