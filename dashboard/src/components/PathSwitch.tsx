import { PATHS_2050, SCENARIO_META } from "../lib/scenarioMeta";
import type { ScenarioKey } from "../lib/types";

interface Props {
  /** Unique radio-group name; distinct per instance on the page. */
  name: string;
  scenario: ScenarioKey;
  onChange: (k: ScenarioKey) => void;
}

/** The fork's choice, kept in reach: a sticky pill row offering the three 2050 paths.
 *  Today and 2030 are not offered here — they stay visible as reference rows in every
 *  chart, so re-selecting them only muddied the "which future?" question. */
export function PathSwitch({ name, scenario, onChange }: Props) {
  return (
    <div className="payoff__switch">
      <span className="payoff__switch-label">2050 path</span>
      <fieldset className="payoff__pills">
        <legend className="visually-hidden">Scenario shown below</legend>
        {PATHS_2050.map((k) => (
          <label key={k} className="payoff__pill" data-path={k}>
            <input
              type="radio"
              name={name}
              value={k}
              checked={scenario === k}
              onChange={() => onChange(k)}
            />
            {SCENARIO_META[k].short}
          </label>
        ))}
      </fieldset>
    </div>
  );
}
