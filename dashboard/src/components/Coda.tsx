import { pct } from "../lib/format";
import { SCENARIO_META, scenarioLabel } from "../lib/scenarioMeta";
import { officeHeadline } from "../lib/transform";
import type { ScenarioKey, ScenariosData } from "../lib/types";
import { Act } from "./Act";

interface Props {
  data: ScenariosData;
  scenario: ScenarioKey;
}

// Which of the fork's levers each 2050 path pulls, phrased from that path's own
// assumptions (scenarioMeta) — no numbers that aren't already on the page.
const LEVERS: Partial<Record<ScenarioKey, string>> = {
  "2050_L":
    "the path that pulls all three levers: a near-clean grid, zero-GWP refrigerants, and comfort that adapts up to 26 °C",
  "2050_M":
    "the path that pulls two levers, the grid and the refrigerants, while comfort holds at 25 °C",
  "2050_H":
    "the path that pulls none: the grid stalls, refrigerants sit at the F-gas ceiling, and comfort expectations fall to 23 °C",
};

/** The coda — a closing beat after the detail act. It bookends the hero (same serif
 *  headline voice), keeps the chosen path's tint via Act's data-path, answers the
 *  reader's choice, and hands over the thesis and dataset as the next step. */
export function Coda({ data, scenario }: Props) {
  const h = officeHeadline(data.scenarios.SQ.archetypes);
  const lever = LEVERS[scenario];
  return (
    <Act id="coda" variant="coda" path={scenario} labelledBy="coda-h">
      <h2 id="coda-h" className="visually-hidden">
        How the story ends
      </h2>
      <p className="hero__headline">
        Every future is hotter. Which one arrives is <em>chosen</em>: by the grid, the refrigerant,
        and the thermostat.
      </p>
      <p className="lede">
        {lever ? (
          <>
            You picked <strong>{scenarioLabel(scenario)}</strong>: {lever}. The 18-fold spread
            between the 2050 paths (a tenth of today's emissions to 1.7 times them) is the distance
            those levers cover.
          </>
        ) : (
          <>
            You picked <strong>{scenarioLabel(scenario)}</strong>:{" "}
            {SCENARIO_META[scenario].tagline.toLowerCase()}. The 2050 fork above is still open: its
            paths span an 18-fold range in emissions, and the levers that separate them (the grid,
            the refrigerant, the thermostat) are chosen, not forecast.
          </>
        )}
      </p>
      <p className="lede">
        And where to start is already on the map: the offices that fill {pct(h.areaShare, 1, 0)} of
        the floor area drive {pct(h.ghgShare, 1, 0)} of today's cooling emissions.
      </p>
      <p className="coda__links">
        <a href="https://repository.tudelft.nl/record/uuid:32222863-536f-464a-b8c6-6c2283a7249a">
          Read the full thesis
        </a>
        <a href="https://doi.org/10.5281/zenodo.8344580">Explore the dataset on Zenodo</a>
      </p>
    </Act>
  );
}
