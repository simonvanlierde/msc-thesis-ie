import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { Act } from "./components/Act";
import { Fork } from "./components/Fork";
import { HowItWorks } from "./components/HowItWorks";
import { NearTerm } from "./components/NearTerm";
import { PathSwitch } from "./components/PathSwitch";
import { Payoff } from "./components/Payoff";
import { TodayHero } from "./components/TodayHero";
import { type Datasets, loadDatasets } from "./lib/data";
import { getPalette } from "./lib/palette";
import { elecFactors } from "./lib/transform";
import type { ScenarioKey } from "./lib/types";
import { useScrollSpy } from "./lib/useScrollSpy";
import { useTheme } from "./lib/useTheme";

// Split the heavy views (MapLibre, nivo) into their own chunks so the story and controls
// paint immediately while the maps and charts stream in. Each gets its own Suspense
// boundary: one shared boundary would hold the charts hostage to MapLibre's 1 MB.
const MapView = lazy(() => import("./components/MapView").then((m) => ({ default: m.MapView })));
const TemporalView = lazy(() =>
  import("./components/TemporalView").then((m) => ({ default: m.TemporalView })),
);
const LcaView = lazy(() => import("./components/LcaView").then((m) => ({ default: m.LcaView })));

function ViewFallback({ label }: { label: string }) {
  return (
    <p className="loading" role="status">
      Loading {label}…
    </p>
  );
}

// Nav targets, in scroll order. Module-scoped so the id list is a stable ref for the spy.
const NAV = [
  { id: "today", label: "Now" },
  { id: "fork", label: "2050" },
  { id: "payoff", label: "Impact" },
  { id: "detail", label: "Detail" },
];
const NAV_IDS = NAV.map((n) => n.id);

/** Two-way visibility for the hero wordmark: the masthead title shows only once the
 *  wordmark has scrolled out, so the name appears to hand off into the header. */
function useHeroPassed(active: boolean) {
  const [passed, setPassed] = useState(false);
  useEffect(() => {
    if (!active || typeof IntersectionObserver === "undefined") return;
    const el = document.querySelector(".wordmark");
    if (!el) return;
    const obs = new IntersectionObserver(([e]) => setPassed(!e.isIntersecting));
    obs.observe(el);
    return () => obs.disconnect();
  }, [active]);
  return passed;
}

export function App() {
  const [data, setData] = useState<Datasets | null>(null);
  const [failed, setFailed] = useState(false);
  // Opens on the middle 2050 path so the payoff and detail land on a future, not the present —
  // the fork is pre-set, and choosing another path is the page's central interaction.
  const [scenario, setScenario] = useState<ScenarioKey>("2050_M");
  const [mode, toggleTheme] = useTheme();
  const palette = useMemo(() => getPalette(mode), [mode]);
  const activeSection = useScrollSpy(NAV_IDS, data !== null);
  const heroPassed = useHeroPassed(data !== null);
  const elec = useMemo(
    () => (data ? elecFactors(data.scenarios.scenarios.SQ.archetypes) : null),
    [data],
  );

  useEffect(() => {
    loadDatasets()
      .then(setData)
      .catch((e: unknown) => {
        // biome-ignore lint/suspicious/noConsole: the only place a load failure surfaces for debugging
        console.error("Could not load the thesis data", e);
        setFailed(true);
      });
  }, []);

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className={`masthead${heroPassed ? " masthead--titled" : ""}`}>
        <div className="wrap masthead__row">
          <h1 className="masthead__title">
            <span className="masthead__name">Cooling for Comfort</span>
            <span className="masthead__sep" aria-hidden="true">
              ·
            </span>
            <button type="button" className="masthead__city" popoverTarget="city-pop">
              The Hague
              <span aria-hidden="true" className="masthead__caret">
                ▾
              </span>
            </button>
          </h1>
          <div id="city-pop" popover="auto" className="citypop">
            <strong>More cities are coming.</strong> The model behind this page is being extended to
            estimate cooling demand and its impacts for every municipality in the Netherlands. The
            Hague is the pilot.
          </div>
          <nav aria-label="Sections">
            {NAV.map((n) => (
              <a
                key={n.id}
                href={`#${n.id}`}
                aria-current={activeSection === n.id ? "true" : undefined}
              >
                {n.label}
              </a>
            ))}
          </nav>
          <button type="button" className="iconbtn" onClick={toggleTheme}>
            {mode === "dark" ? "☀ Light" : "☾ Dark"}
          </button>
        </div>
      </header>

      <main id="main">
        {failed && (
          <p className="errbox wrap" role="alert">
            The thesis data didn't load. Check your connection and reload the page.
          </p>
        )}
        {!(data || failed) && <p className="loading wrap">Loading thesis results…</p>}

        {data && (
          <>
            <TodayHero data={data.scenarios} />
            <HowItWorks />
            <NearTerm data={data.scenarios} />
            <Fork scenario={scenario} onChange={setScenario} />
            <Payoff
              data={data.scenarios}
              scenario={scenario}
              onChange={setScenario}
              palette={palette}
            />

            <Act
              id="detail"
              variant="detail"
              eyebrow="The detail behind the story"
              labelledBy="detail-h"
            >
              <h2 id="detail-h">Where, when, and what it costs</h2>
              <p className="lede">
                The full picture behind the headline: where cooling concentrates across the city,
                how demand moves through the day and year, and the life-cycle breakdown of the
                climate impact — all for the path you chose above.
              </p>

              {/* The fork's choice, kept switchable while deep in the detail views. */}
              <PathSwitch name="detail-path" scenario={scenario} onChange={setScenario} />

              <Suspense fallback={<ViewFallback label="the map" />}>
                <MapView buurten={data.buurten} scenario={scenario} palette={palette} />
              </Suspense>
              <Suspense fallback={<ViewFallback label="the profiles" />}>
                {elec && <TemporalView temporal={data.temporal} elec={elec} palette={palette} />}
              </Suspense>
              <Suspense fallback={<ViewFallback label="the impact charts" />}>
                <LcaView data={data.scenarios} scenario={scenario} palette={palette} />
              </Suspense>
            </Act>

            <footer className="colophon wrap">
              <p>
                Data: bottom-up cooling-demand and life-cycle model of The Hague building stock (BAG
                geospatial data · hourly heat-balance · LCA). MSc Industrial Ecology thesis, Leiden
                University &amp; TU Delft.{" "}
                <a href="https://doi.org/10.5281/zenodo.8344580">Dataset (Zenodo)</a> ·{" "}
                <a href="https://repository.tudelft.nl/record/uuid:32222863-536f-464a-b8c6-6c2283a7249a">
                  Thesis
                </a>
                .
              </p>
              <p className="note">{data.scenarios.meta.source}</p>
            </footer>
          </>
        )}
      </main>
    </>
  );
}
