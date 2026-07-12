import { PATHS_2050, SCENARIO_META, scenarioLabel } from "../lib/scenarioMeta";
import type { ScenarioKey } from "../lib/types";

interface Props {
  /** Unique radio-group name; distinct per instance on the page. */
  name: string;
  scenario: ScenarioKey;
  onChange: (k: ScenarioKey) => void;
  /** Scenarios offered. Defaults to the three 2050 paths — in the impact act, Today and
   *  2030 stay visible as reference rows in every chart, so re-selecting them only muddied
   *  the "which future?" question. The detail act passes all five: its map and LCA views
   *  show one scenario at a time, so the present must be selectable there. */
  keys?: ScenarioKey[];
  label?: string;
}

/** The fork's choice, kept in reach: a sticky pill row for switching scenario. */
export function PathSwitch({
  name,
  scenario,
  onChange,
  keys = PATHS_2050,
  label = "2050 path",
}: Props) {
  // With Today/2030 in the row the futures need their year ("2050 Low"); in a pure
  // 2050 group the label already says so and the short form keeps the pill compact.
  const mixed = keys.some((k) => SCENARIO_META[k].kind !== "future");
  return (
    <div className="payoff__switch">
      <span className="payoff__switch-label">{label}</span>
      <fieldset className="payoff__pills">
        <legend className="visually-hidden">Scenario shown below</legend>
        {keys.map((k) => (
          <label key={k} className="payoff__pill" data-path={k}>
            <input
              type="radio"
              name={name}
              value={k}
              checked={scenario === k}
              onChange={() => onChange(k)}
            />
            {mixed ? scenarioLabel(k) : SCENARIO_META[k].short}
          </label>
        ))}
      </fieldset>
    </div>
  );
}
