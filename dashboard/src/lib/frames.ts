// Helpers for the animated year time-lapse: turn a frame index into a label, and
// a cooling-intensity value into a colour on a fixed scale (so winter stays pale
// and summer afternoons saturate). Pure + unit-tested (frames.test.ts).

export const FRAMES_PER_DAY = 24;
export const FRAME_COUNT = 12 * FRAMES_PER_DAY;

/** Frame index (month-major, hour-minor) → { month 0-11, hour 0-23 }. */
export function frameParts(index: number): { month: number; hour: number } {
  const i = ((index % FRAME_COUNT) + FRAME_COUNT) % FRAME_COUNT;
  return { month: Math.floor(i / FRAMES_PER_DAY), hour: i % FRAMES_PER_DAY };
}

/** "Jul · 15:00" for the given frame. */
export function frameLabel(index: number, months: string[]): string {
  const { month, hour } = frameParts(index);
  return `${months[month]} · ${String(hour).padStart(2, "0")}:00`;
}

/** Bin a value onto [0, vmax] across `bins` classes (top bin catches the overflow). */
export function heatBin(value: number, vmax: number, bins: number): number {
  if (!(vmax > 0) || value <= 0) return 0;
  return Math.min(bins - 1, Math.floor((value / vmax) * bins));
}

/** Colour for a value on the fixed heat scale. */
export function heatColor(value: number, vmax: number, ramp: string[]): string {
  return ramp[heatBin(value, vmax, ramp.length)];
}

/** Legend rows [colour, "lo–hi"] for the fixed heat scale. */
export function heatLegend(
  vmax: number,
  ramp: string[],
  fmt: (n: number) => string,
): Array<{ color: string; label: string }> {
  const step = vmax / ramp.length;
  return ramp.map((color, i) => {
    const lo = step * i;
    const hi = i === ramp.length - 1 ? null : step * (i + 1);
    return { color, label: hi == null ? `≥ ${fmt(lo)}` : `${fmt(lo)}–${fmt(hi)}` };
  });
}
