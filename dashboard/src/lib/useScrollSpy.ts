import { useEffect, useState } from "react";

/**
 * Tracks which of the given section ids is the one currently under the top of the viewport,
 * for highlighting the masthead nav. Uses one IntersectionObserver with a band near the top
 * of the screen: whichever section last entered that band is active. Falls back to the first
 * id where IntersectionObserver is unavailable.
 */
export function useScrollSpy(ids: string[], enabled = true): string {
  const [active, setActive] = useState(ids[0] ?? "");

  useEffect(() => {
    // `enabled` gates until the sections exist in the DOM — they mount only after the data
    // loads, so an eager run on mount would observe nothing and never re-attach.
    if (!enabled || typeof IntersectionObserver === "undefined") return;
    const els = ids
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null);
    if (els.length === 0) return;

    // Track each section's visibility and pick the topmost one in document order that is
    // currently in the band — deterministic, where "whichever entry fired last" is not.
    const visible = new Map<string, boolean>();
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) visible.set(e.target.id, e.isIntersecting);
        const top = ids.find((id) => visible.get(id));
        if (top) setActive(top);
      },
      // A thin band across the viewport middle: the section crossing the centre line is
      // active. Robust where a band pinned to the top grazes section boundaries.
      { rootMargin: "-45% 0px -45% 0px", threshold: 0 },
    );
    for (const el of els) obs.observe(el);
    return () => obs.disconnect();
  }, [ids, enabled]);

  return active;
}
