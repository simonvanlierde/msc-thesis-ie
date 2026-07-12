interface Props {
  values: number[];
  breaks: number[];
  fmt: (n: number) => string;
  unit: string;
}

// User units ≈ rendered pixels at the page's content width, so the type reads at its
// nominal size rather than being scaled up by the viewBox.
const W = 1000;
const H = 28;
const BASE_Y = 16;

/**
 * One tick per neighbourhood along the value axis, with the quantile class breaks drawn as
 * rules. Quantile bins hold equal *counts*, so the swatch legend alone can never show how
 * the 112 values actually spread — this does. The ranges stay in the legend beside it.
 */
export function ValueStrip({ values, breaks, fmt, unit }: Props) {
  if (values.length === 0) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const x = (v: number) => ((v - min) / span) * W;

  return (
    <div className="valuestrip">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Distribution of ${values.length} neighbourhoods, from ${fmt(min)} to ${fmt(max)} ${unit}. Taller vertical lines mark the colour class breaks.`}
      >
        <line className="valuestrip__break" x1={0} x2={W} y1={BASE_Y} y2={BASE_Y} />
        {breaks.map((b) => (
          <line className="valuestrip__break" key={b} x1={x(b)} x2={x(b)} y1={0} y2={BASE_Y} />
        ))}
        {values.map((v, i) => (
          <line
            className="valuestrip__tick"
            // biome-ignore lint/suspicious/noArrayIndexKey: ticks are positional and values repeat
            key={i}
            x1={x(v)}
            x2={x(v)}
            y1={5}
            y2={BASE_Y}
          />
        ))}
        <text className="valuestrip__axis" x={0} y={H - 1} textAnchor="start">
          {fmt(min)}
        </text>
        <text className="valuestrip__axis" x={W} y={H - 1} textAnchor="end">
          {fmt(max)} {unit}
        </text>
      </svg>
    </div>
  );
}
