import { PATHS_2050, REFERENCE_STATES, SCENARIO_META } from "../lib/scenarioMeta";
import type { ScenarioKey } from "../lib/types";
import { Act } from "./Act";

interface Props {
  scenario: ScenarioKey;
  onChange: (k: ScenarioKey) => void;
}

const CHIP_ORDER: { key: keyof ScenarioMetaAssumptions; label: string }[] = [
  { key: "warming", label: "Warming" },
  { key: "comfort", label: "Comfort" },
  { key: "grid", label: "Grid" },
  { key: "refrigerant", label: "Refrigerant" },
];
type ScenarioMetaAssumptions = (typeof SCENARIO_META)[ScenarioKey]["assumptions"];

/** Act 3 — the signature choice. Three 2050 paths as cards you commit to; Today and 2030
 *  stay reachable as quiet reference states. One radiogroup underneath (the same `scenario`
 *  state the map and impact charts already read), so it is fully keyboard-operable and the
 *  selection is never colour-alone — the checked card names its own path and assumptions. */
export function Fork({ scenario, onChange }: Props) {
  return (
    <Act id="fork" variant="fork" eyebrow="2050 · your choice" labelledBy="fork-h">
      <h2 id="fork-h">Choose the path to 2050</h2>
      <p className="lede">
        By 2050 the same city's cooling emissions span an 18-fold range — from a tenth of today's to
        nearly double. Which future arrives is a set of choices: how hot we let it get, whether we
        adapt what "comfortable" means, how clean the grid stays, whether high-warming refrigerants
        are phased out.
      </p>

      <p className="fork__prompt">
        Choose a 2050 — the impact below and the map update to the path you pick.
      </p>

      <fieldset className="fork">
        <legend className="visually-hidden">Path to 2050</legend>
        <div className="fork__paths">
          {PATHS_2050.map((k) => {
            const m = SCENARIO_META[k];
            const checked = scenario === k;
            return (
              <label key={k} className="fork__card" data-path={k}>
                <input
                  type="radio"
                  name="scenario"
                  value={k}
                  checked={checked}
                  onChange={() => onChange(k)}
                />
                <span className="fork__head">
                  <span className="fork__radio" aria-hidden="true" />
                  <span className="fork__short">{m.short}</span>
                  <span className="fork__tag">{m.tagline}</span>
                </span>
                <span className="fork__blurb">{m.blurb}</span>
                <span className="fork__chips">
                  {CHIP_ORDER.map((c) => (
                    <span className="chip" key={c.key}>
                      <span className="chip__k">{c.label}</span>
                      {m.assumptions[c.key]}
                    </span>
                  ))}
                </span>
              </label>
            );
          })}
        </div>

        <div className="fork__ref">
          <span className="fork__ref-label">Or compare with the present and near term:</span>
          {REFERENCE_STATES.map((k) => (
            <label key={k} className="fork__refchip" data-path={k}>
              <input
                type="radio"
                name="scenario"
                value={k}
                checked={scenario === k}
                onChange={() => onChange(k)}
              />
              {SCENARIO_META[k].short}
            </label>
          ))}
        </div>
      </fieldset>
    </Act>
  );
}
