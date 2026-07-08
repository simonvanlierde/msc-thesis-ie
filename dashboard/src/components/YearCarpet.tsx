import { FRAMES_PER_DAY, frameParts, heatColor } from "../lib/frames";
import { HEAT_RAMP } from "../lib/palette";

interface Props {
  /** City-average cooling intensity for each of the 288 frames, month-major. */
  cityMean: number[];
  months: string[];
  vmax: number;
  frame: number;
  onPick: (frame: number) => void;
  fmt: (n: number) => string;
}

const HOURS = Array.from({ length: FRAMES_PER_DAY }, (_, i) => i);
const CELL_W = 36;
const CELL_H = 30;
const GUTTER_L = 44;
const GUTTER_B = 20;
const PAD_T = 2;
const W = GUTTER_L + FRAMES_PER_DAY * CELL_W;
const H = PAD_T + 12 * CELL_H + GUTTER_B;
const hh = (h: number) => String(h).padStart(2, "0");

/**
 * The whole year at once: 12 months by 24 hours, coloured on the same heat scale as the
 * map below it. Building-energy analysts read these — the shape of the cooling season is a
 * property of the city, and a 288-step slider hides it. Clicking a cell scrubs the map.
 *
 * The svg is role="img", so its cells are presentational to assistive tech; the labelled
 * range input beside it is the keyboard-operable equivalent for the same state.
 */
export function YearCarpet({ cityMean, months, vmax, frame, onPick, fmt }: Props) {
  const active = frameParts(frame);

  return (
    <div className="carpet">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Cooling intensity for every hour of a typical year, as a grid of 12 months by 24 hours. Palest on winter nights, deepest red on summer afternoons. Peak city average ${fmt(Math.max(...cityMean))} watts per square metre.`}
      >
        <title>City-average cooling intensity, month by hour</title>

        {months.map((mo, m) => (
          <text
            className="carpet__axis"
            key={mo}
            x={GUTTER_L - 8}
            y={PAD_T + m * CELL_H + CELL_H / 2 + 4}
            textAnchor="end"
          >
            {mo}
          </text>
        ))}

        {months.map((mo, m) =>
          HOURS.map((h) => {
            const idx = m * FRAMES_PER_DAY + h;
            const v = cityMean[idx];
            return (
              // biome-ignore lint/a11y/noStaticElementInteractions: cells are presentational inside role="img"; the range input below is the keyboard equivalent
              <rect
                className="carpet__cell"
                key={`${mo}-${h}`}
                x={GUTTER_L + h * CELL_W}
                y={PAD_T + m * CELL_H}
                width={CELL_W}
                height={CELL_H}
                fill={heatColor(v, vmax, HEAT_RAMP)}
                onClick={() => onPick(idx)}
              >
                <title>{`${mo} · ${hh(h)}:00 — ${fmt(v)} W/m²`}</title>
              </rect>
            );
          }),
        )}

        <rect
          className="carpet__cursor"
          x={GUTTER_L + active.hour * CELL_W}
          y={PAD_T + active.month * CELL_H}
          width={CELL_W}
          height={CELL_H}
        />

        {HOURS.filter((h) => h % 3 === 0).map((h) => (
          <text
            className="carpet__axis"
            key={h}
            x={GUTTER_L + h * CELL_W + CELL_W / 2}
            y={H - 6}
            textAnchor="middle"
          >
            {hh(h)}
          </text>
        ))}
      </svg>
    </div>
  );
}
