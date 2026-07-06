interface Item {
  color: string;
  label: string;
}

/** Shared legend — identity is always available as text, never colour-alone. */
export function Legend({ items, title }: { items: Item[]; title?: string }) {
  return (
    <ul className="legend" aria-label={title ?? "Legend"}>
      {items.map((it) => (
        <li className="legend__item" key={it.label}>
          <span className="legend__swatch" style={{ background: it.color }} aria-hidden="true" />
          {it.label}
        </li>
      ))}
    </ul>
  );
}
