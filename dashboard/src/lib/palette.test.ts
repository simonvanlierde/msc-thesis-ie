import { describe, expect, it } from "vitest";
import { getPalette } from "./palette";

/** WCAG relative-luminance contrast ratio between two opaque hexes. */
function contrast(a: string, b: string): number {
  const lum = (hex: string) => {
    const c = [1, 3, 5].map((i) => {
      const v = Number.parseInt(hex.slice(i, i + 2), 16) / 255;
      return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  };
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

describe("palette text tokens", () => {
  // Chart labels and values wear the ink tokens on the page surface; they owe 4.5:1.
  it("primary and secondary ink clear 4.5:1 on page and surface, in both modes", () => {
    for (const mode of ["light", "dark"] as const) {
      const p = getPalette(mode);
      for (const ink of [p.textPrimary, p.textSecondary]) {
        for (const bg of [p.page, p.surface]) {
          expect(contrast(ink, bg), `${mode} ${ink} on ${bg}`).toBeGreaterThanOrEqual(4.5);
        }
      }
    }
  });
});
