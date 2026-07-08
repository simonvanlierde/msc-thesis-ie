import { useEffect, useState } from "react";
import type { Mode } from "./palette";

/** Must match the key and the fallback chain in the inline script in index.html. */
const KEY = "cooling-dashboard.theme";

function initialMode(): Mode {
  // index.html already resolved this before first paint — read it back rather than
  // recomputing, so React state and the painted theme can never disagree.
  const stamped = typeof document !== "undefined" ? document.documentElement.dataset.theme : null;
  if (stamped === "light" || stamped === "dark") return stamped;
  return typeof matchMedia !== "undefined" && matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

/** Theme state: respects prefers-color-scheme, allows a manual override, persists it,
 *  and stamps `data-theme` on <html> so CSS variables switch. */
export function useTheme(): [Mode, () => void] {
  const [mode, setMode] = useState<Mode>(initialMode);

  useEffect(() => {
    document.documentElement.dataset.theme = mode;
    localStorage.setItem(KEY, mode);
  }, [mode]);

  return [mode, () => setMode((m) => (m === "dark" ? "light" : "dark"))];
}
