import { useEffect, useRef, useState } from "react";

/**
 * Reveal-on-scroll: watches an element and flips to true the first time it enters the
 * viewport, then stops observing (the reveal is one-way). CSS does the actual animation
 * and honours prefers-reduced-motion, so this hook is motion-agnostic — it only reports
 * visibility. Falls back to visible-always where IntersectionObserver is missing.
 */
export function useInView<T extends Element>(rootMargin = "0px 0px -12% 0px") {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setInView(true);
          obs.disconnect();
        }
      },
      { rootMargin },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [rootMargin]);

  return [ref, inView] as const;
}
