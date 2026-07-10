interface Option<T extends string> {
  value: T;
  label: string;
}

interface Props<T extends string> {
  /** Unique radio-group name; distinct per control on the page. */
  name: string;
  legend: string;
  options: Option<T>[];
  value: T;
  onChange: (v: T) => void;
  /** What this control scopes, e.g. "the map". Read out to assistive tech. */
  scope?: string;
}

/** Radio-group segmented control — keyboard-navigable, labelled, selection not colour-alone. */
export function Segmented<T extends string>({
  name,
  legend,
  options,
  value,
  onChange,
  scope,
}: Props<T>) {
  return (
    <fieldset className="segmented">
      <legend>
        {legend}
        {scope && <span className="segmented__scope"> · {scope}</span>}
      </legend>
      <div className="segmented__row" role="presentation">
        {options.map((o) => (
          <label key={o.value}>
            <input
              type="radio"
              name={name}
              value={o.value}
              checked={value === o.value}
              onChange={() => onChange(o.value)}
            />
            {o.label}
          </label>
        ))}
      </div>
    </fieldset>
  );
}
