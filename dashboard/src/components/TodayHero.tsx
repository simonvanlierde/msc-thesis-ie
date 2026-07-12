import { useState } from "react";
import { gwh, ktCO2, num, pct } from "../lib/format";
import { SCENARIO_META } from "../lib/scenarioMeta";
import { officeHeadline } from "../lib/transform";
import type { ScenariosData } from "../lib/types";
import { Act } from "./Act";

interface Props {
  data: ScenariosData;
}

// Rough-scale anchors for the comparisons. Deliberately coarse — they answer
// "how much is a GWh?", not an accounting question.
const HOUSEHOLD_KWH = 2500; // average Dutch household electricity use per year (CBS)
const CAR_KG_CO2 = 2000; // average Dutch petrol car: ~12,000 km/yr at ~170 g CO2/km

interface Tile {
  id: string;
  value: string;
  label: string;
  more: string;
}

/** Act 1 — the anchor. Always the present-day (SQ) city, framed as the opening thesis:
 *  a small office stock carries an outsized share of the cooling burden. */
export function TodayHero({ data }: Props) {
  const s = data.scenarios.SQ;
  const h = officeHeadline(s.archetypes);
  const [open, setOpen] = useState<string | null>(null);

  const demandHouseholds = s.totals.E_cooling_kWh / HOUSEHOLD_KWH;
  const elecHouseholds = s.totals.electricity_kWh / HOUSEHOLD_KWH;
  const cars = s.totals.GHG_emissions_total_kgCO2eq / CAR_KG_CO2;

  const tiles: Tile[] = [
    {
      id: "demand",
      value: gwh(s.totals.E_cooling_kWh),
      label: "Cooling demand / year",
      more: `The heat the city's buildings would need to shed to stay comfortable — equal to the annual electricity use of about ${num(Math.round(demandHouseholds / 1000) * 1000)} households, more than every home in The Hague combined. Most of it is never met.`,
    },
    {
      id: "electricity",
      value: gwh(s.totals.electricity_kWh),
      label: "Electricity for cooling / year",
      more: `What the cooling equipment that does exist draws from the grid: the annual electricity of roughly ${num(Math.round(elecHouseholds / 1000) * 1000)} households. On a hot afternoon it peaks near 69 MW, about half the capacity of the nearby Luchterduinen offshore wind park — real load on a grid already running near its limits.`,
    },
    {
      id: "ghg",
      value: ktCO2(s.totals.GHG_emissions_total_kgCO2eq),
      label: "Life-cycle emissions / year",
      more: `Greenhouse gases from generating that electricity (88% of the total), refrigerant leaks, and making and scrapping the equipment — the yearly exhaust of about ${num(Math.round(cars / 1000) * 1000)} petrol cars, or ~90 kg CO₂-eq per resident.`,
    },
    {
      id: "unmet",
      value: "77%",
      label: "Of cooling demand goes unmet",
      more: "About 85% of Hague homes have no cooling at all, so the heat stays indoors as discomfort — worst in the lower-income neighbourhoods where the heat island runs strongest. That 860 GWh gap is where future growth sits: as cooling spreads, unmet demand becomes electricity and emissions.",
    },
  ];

  return (
    <Act id="today" variant="today" labelledBy="today-h">
      <p className="wordmark">
        <span>Cooling for Comfort,</span>
        <span className="wordmark__warm">Warming the World</span>
      </p>
      <p className="act__eyebrow">Now · the city as it is</p>
      <h2 id="today-h" className="visually-hidden">
        Cooling in The Hague today
      </h2>
      <p className="hero__headline">
        Offices fill just <em>{pct(h.areaShare, 1, 0)}</em> of the floor area but drive{" "}
        <em>{pct(h.ghgShare, 1, 0)}</em> of the city's cooling-related greenhouse-gas emissions.
      </p>
      <p className="lede">
        It is getting hotter, and the world is reaching for cooling: global cooling electricity is
        on course to triple by 2050. The Netherlands is warming at twice the global average, yet
        comes to this late — heat waves used to be rare here, and air-conditioning went from 1.5% of
        homes in 2000 to 30% in 2023. Today cooling's footprint is still modest; how big it gets is
        a choice.
      </p>
      <p className="lede">{SCENARIO_META.SQ.blurb}</p>
      <p className="note hero__now">
        "Now" is the model's status-quo baseline, centred on 2020: the city's real building stock,
        run hour by hour through the 2018–2022 weather, with the grid and cooling equipment of that
        moment.
      </p>

      <div className="tiles">
        {tiles.map((t) => {
          const isOpen = open === t.id;
          return (
            <button
              key={t.id}
              type="button"
              className="tile"
              aria-expanded={isOpen}
              onClick={() => setOpen(isOpen ? null : t.id)}
            >
              <span className="tile__value">{t.value}</span>
              <span className="tile__label">{t.label}</span>
              {isOpen && <span className="tile__more">{t.more}</span>}
              <span className="tile__hint">{isOpen ? "Show less ▴" : "How much is that? ▾"}</span>
            </button>
          );
        })}
      </div>
    </Act>
  );
}
