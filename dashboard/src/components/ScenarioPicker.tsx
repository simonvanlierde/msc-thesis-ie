import type { Scenario, ScenarioKey } from "../lib/types";

interface Props {
  order: ScenarioKey[];
  scenarios: Record<ScenarioKey, Scenario>;
  value: ScenarioKey;
  onChange: (k: ScenarioKey) => void;
}

/** Radio-group segmented control — keyboard-navigable, labelled, selection not colour-alone. */
export function ScenarioPicker({ order, scenarios, value, onChange }: Props) {
  return (
    <fieldset className="segmented">
      <legend>Scenario</legend>
      <div className="segmented__row" role="presentation">
        {order.map((key) => (
          <label key={key}>
            <input
              type="radio"
              name="scenario"
              value={key}
              checked={value === key}
              onChange={() => onChange(key)}
            />
            {scenarios[key].label}
          </label>
        ))}
      </div>
    </fieldset>
  );
}
