import type { ReactNode } from "react";
import { useInView } from "../lib/useInView";

interface Props {
  id: string;
  /** Act variant, drives the cool→warm tint: today | near | fork | payoff | detail. */
  variant: string;
  /** For the payoff act: the active scenario key, so the tint commits to the chosen path. */
  path?: string;
  eyebrow?: string;
  labelledBy?: string;
  children: ReactNode;
}

/** A scroll act: a tinted section that reveals as it enters the viewport (CSS handles the
 *  motion and reduced-motion; useInView only reports visibility). */
export function Act({ id, variant, path, eyebrow, labelledBy, children }: Props) {
  const [ref, inView] = useInView<HTMLElement>();
  return (
    <section
      id={id}
      ref={ref}
      className={`act act--${variant} reveal${inView ? " in-view" : ""}`}
      data-path={path}
      aria-labelledby={labelledBy}
    >
      <div className="wrap act__inner">
        {eyebrow && <p className="act__eyebrow">{eyebrow}</p>}
        {children}
      </div>
    </section>
  );
}
