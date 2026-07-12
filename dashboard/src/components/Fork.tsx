import { PATHS_2050, SCENARIO_META } from "../lib/scenarioMeta";
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

/** Act 3 — the signature choice. Three 2050 paths as cards you commit to. One radiogroup
 *  (the same `scenario` state the map and impact charts already read), so it is fully
 *  keyboard-operable and the selection is never colour-alone — the checked card names its
 *  own path and assumptions. The full assumption list sits in a disclosure per card,
 *  outside the radio label so opening it never re-fires the selection. */
export function Fork({ scenario, onChange }: Props) {
  return (
    <Act id="fork" variant="fork" eyebrow="2050 · your choice" labelledBy="fork-h">
      <h2 id="fork-h">Choose the path to 2050</h2>
      <p className="lede">
        By 2050 the same city's cooling emissions span an 18-fold range — from a tenth of today's to
        1.7 times today's. Which future arrives is a set of choices: how hot we let it get, whether
        we adapt what "comfortable" means, how clean the grid stays, whether high-warming
        refrigerants are phased out.
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
              <div key={k} className="fork__card" data-path={k}>
                <label className="fork__pick">
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
                {m.details && (
                  <details className="fork__more">
                    <summary>All assumptions</summary>
                    <ul>
                      {m.details.map((d) => (
                        <li key={d}>{d}</li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            );
          })}
        </div>
      </fieldset>
    </Act>
  );
}
