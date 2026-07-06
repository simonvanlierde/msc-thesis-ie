// Number formatting for a mixed audience: compact, human units, no raw kWh strings.

/** Cooling energy: kWh -> GWh with one decimal (city totals are ~1000 GWh). */
export function gwh(kwh: number): string {
  return `${(kwh / 1e6).toLocaleString("en", { maximumFractionDigits: 1 })} GWh`;
}

/** GHG: kg CO2-eq -> kt CO2-eq. */
export function ktCO2(kg: number): string {
  return `${(kg / 1e6).toLocaleString("en", { maximumFractionDigits: 1 })} kt CO₂-eq`;
}

/** Compact SI-ish number with a unit, e.g. 1.2M, 34k. */
export function compact(n: number, unit = ""): string {
  const s = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(
    n,
  );
  return unit ? `${s} ${unit}` : s;
}

/** Percentage of a whole, rounded to the given precision. */
export function pct(part: number, whole: number, digits = 0): string {
  if (whole === 0) return "—";
  return `${((part / whole) * 100).toFixed(digits)}%`;
}

/** Plain number with thousands separators. */
export function num(n: number, digits = 0): string {
  return n.toLocaleString("en", { maximumFractionDigits: digits });
}
