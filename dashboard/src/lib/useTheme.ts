import { useEffect, useState } from "react";
import type { Mode } from "./palette";

const KEY = "cooling-dashboard.theme";

function initialMode(): Mode {
  const saved = typeof localStorage !== "undefined" ? localStorage.getItem(KEY) : null;
  if (saved === "light" || saved === "dark") return saved;
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
