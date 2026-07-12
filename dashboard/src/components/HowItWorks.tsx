import { num, pct } from "../lib/format";
import type { ScenariosData } from "../lib/types";
import { Act } from "./Act";

interface Step {
  title: string;
  body: string;
  /** The step's one number, pulled out of the prose so the chain scans visually. */
  fig: string;
  figCap: string;
  /** What this step hands to the next one — the quantity flowing down the chain. */
  flow?: string;
}

/** The model chain, told as the six steps the thesis actually computes, in order.
 *  Static figures quoted here are model inputs (data/input/parameters/); the flow
 *  quantities between steps are computed live from scenarios.json in the component. */
function buildSteps(d: ScenariosData): Step[] {
  const t = d.scenarios.SQ.totals;
  const ghg = d.scenarios.SQ.lca_by_stage;
  const ghgTotal = t.GHG_emissions_total_kgCO2eq;
  return [
    {
      title: "Heat piles up",
      body: "Every building gains heat: sun through windows, warm outside air, people, appliances. The model runs an hourly heat balance per building; whatever pushes one past its comfort threshold — 25 °C today — becomes cooling demand, surplus heat that must be removed. The urban heat island makes it much worse: without it, the city's cooling demand would be roughly a third of what it is.",
      fig: "+8.6 °C",
      figCap: "central The Hague vs its rural surroundings, measured on a hot day",
      flow: `${num(t.E_cooling_kWh / 1e6, 0)} GWh of cooling demand a year`,
    },
    {
      title: "There are many ways to shed it",
      body: "Some heat never becomes a machine's problem — shading, ventilation and building fabric can avoid or dump part of it passively. The rest takes active cooling, from portable ACs to split units, chillers, and air-, ground- and water-source heat pumps.",
      fig: "6",
      figCap: "active cooling technologies tracked by the model",
      flow: "the active share, split across technologies",
    },
    {
      title: "Cooling costs electricity",
      body: "Each technology needs electricity to move heat out: demand divided by its seasonal efficiency (SEER), improving in every future scenario.",
      fig: "2.5–7.5",
      figCap: "SEER today, from portable AC to water-source heat pump",
      flow: "demand ÷ efficiency",
    },
    {
      title: "…but only where cooling is installed",
      body: "Market-penetration rates, per building type, scale the hypothetical electricity down to what is actually drawn from the grid.",
      fig: "15%",
      figCap: "of Hague homes have cooling, against roughly 75% of offices",
      flow: `${num(t.electricity_kWh / 1e6, 0)} GWh of electricity actually drawn a year`,
    },
    {
      title: "Electricity carries emissions",
      body: "Generating that electricity emits greenhouse gases in step with the grid's carbon intensity — the biggest lever between the 2050 paths. This page tracks climate only; the thesis also assessed resource depletion.",
      fig: pct(ghg.electricity, ghgTotal, 0),
      figCap: "of cooling's climate impact today is grid electricity",
      flow: `${num(ghg.electricity / 1e6, 1)} kt CO₂-eq from the grid`,
    },
    {
      title: "So does the equipment itself",
      body: `Making, installing and scrapping cooling machines has its own footprint, plus refrigerant leaks along the way (another ${pct(ghg.refrigerant_leaks, ghgTotal, 0)} of today's impact). Installed capacity scales with the peak: systems are sized to cover cooling demand for 98% of the year, riding out the hottest 2% of hours.`,
      fig: "98%",
      figCap: "of hours the installed cooling capacity is sized to cover",
    },
  ];
}

/** The model, in plain words — the chain from surplus heat to climate impact, drawn as
 *  one vertical computation: numbered nodes on a spine, the quantity that flows between
 *  steps written on the spine itself, ending in the number the rest of the page tracks. */
export function HowItWorks({ data }: { data: ScenariosData }) {
  const steps = buildSteps(data);
  const ghgTotal = data.scenarios.SQ.totals.GHG_emissions_total_kgCO2eq;
  return (
    <Act id="model" variant="near" eyebrow="Behind the numbers · the model" labelledBy="model-h">
      <h2 id="model-h">From heat to impact, in six steps</h2>
      <p className="lede">
        Every number on this page comes out of one chain, computed building by building and hour by
        hour for ~59,000 real buildings.
      </p>

      <ol className="chain">
        {steps.map((s, i) => (
          <li key={s.title} className="chain__step">
            <span className="chain__n" aria-hidden="true">
              {i + 1}
            </span>
            <div className="chain__text">
              <h3 className="chain__t">{s.title}</h3>
              <p className="chain__b">{s.body}</p>
            </div>
            <p className="chain__fig">
              <strong>{s.fig}</strong>
              <span>{s.figCap}</span>
            </p>
            {s.flow && (
              <p className="chain__flow">
                <span aria-hidden="true">↓</span> {s.flow}
              </p>
            )}
          </li>
        ))}
      </ol>

      <p className="chain__result">
        <strong>{num(ghgTotal / 1e6, 1)} kt CO₂-eq a year</strong>
        cooling's climate impact today — the number every chart below starts from
      </p>

      <p className="note steps__note">
        Full method, validation and sensitivity analyses: chapter 3 of the thesis (link in the
        footer).
      </p>
    </Act>
  );
}
