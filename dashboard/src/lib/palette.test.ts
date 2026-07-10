import { describe, expect, it } from "vitest";
import { getPalette, inkOn } from "./palette";

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

describe("inkOn", () => {
  // The stacked bar prints a percentage inside each life-cycle segment. The label is text,
  // so it owes 4.5:1 against the fill it sits on — axe cannot see SVG text, so this asserts
  // it. A hardcoded #fff used to fail here on five of the eight fills.
  it("clears 4.5:1 on every life-cycle stage fill, in both modes", () => {
    for (const mode of ["light", "dark"] as const) {
      const { stage } = getPalette(mode);
      for (const [name, fill] of Object.entries(stage)) {
        const ratio = contrast(inkOn(fill), fill);
        expect(ratio, `${mode}/${name} (${fill})`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("picks white on dark fills and black on light ones", () => {
    expect(inkOn("#000000")).toBe("#ffffff");
    expect(inkOn("#ffffff")).toBe("#000000");
  });
});
