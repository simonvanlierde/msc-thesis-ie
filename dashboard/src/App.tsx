import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { ScenarioPicker } from "./components/ScenarioPicker";
import { Summary } from "./components/Summary";
import { type Datasets, loadDatasets } from "./lib/data";
import { getPalette } from "./lib/palette";
import type { ScenarioKey } from "./lib/types";
import { useTheme } from "./lib/useTheme";

// Split the heavy views (MapLibre, nivo) into their own chunks so the summary and controls
// paint immediately while the maps and charts stream in. Each gets its own Suspense
// boundary: one shared boundary would hold the charts hostage to MapLibre's 1 MB.
const MapView = lazy(() => import("./components/MapView").then((m) => ({ default: m.MapView })));
const HourlyMap = lazy(() =>
  import("./components/HourlyMap").then((m) => ({ default: m.HourlyMap })),
);
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

export function App() {
  const [data, setData] = useState<Datasets | null>(null);
  const [failed, setFailed] = useState(false);
  const [scenario, setScenario] = useState<ScenarioKey>("SQ");
  const [mode, toggleTheme] = useTheme();
  const palette = useMemo(() => getPalette(mode), [mode]);

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
      <header className="masthead">
        <div className="wrap masthead__row">
          <h1>Cooling for Comfort · The Hague</h1>
          <nav aria-label="Sections">
            <a href="#map">Map</a>
            <a href="#year">Time-lapse</a>
            <a href="#when">Profiles</a>
            <a href="#impact">Impact</a>
          </nav>
          <button type="button" className="iconbtn" onClick={toggleTheme}>
            {mode === "dark" ? "☀ Light" : "☾ Dark"}
          </button>
        </div>
      </header>

      <main id="main" className="wrap">
        {failed && (
          <p className="errbox" role="alert">
            The thesis data didn't load. Check your connection and reload the page.
          </p>
        )}
        {!(data || failed) && <p className="loading">Loading thesis results…</p>}

        {data && (
          <>
            <Summary data={data.scenarios} scenario={scenario} />

            <div className="card" style={{ marginTop: "1.5rem" }}>
              <ScenarioPicker
                order={data.scenarios.meta.scenario_order}
                scenarios={data.scenarios.scenarios}
                value={scenario}
                onChange={setScenario}
              />
              <p className="scope-note">
                Changes the map and the impact charts. The time-lapse and the daily profiles always
                show the present-day building stock.
              </p>
            </div>

            <Suspense fallback={<ViewFallback label="the map" />}>
              <MapView buurten={data.buurten} scenario={scenario} palette={palette} />
            </Suspense>
            <Suspense fallback={<ViewFallback label="the time-lapse" />}>
              <HourlyMap buurten={data.buurten} palette={palette} />
            </Suspense>
            <Suspense fallback={<ViewFallback label="the profiles" />}>
              <TemporalView temporal={data.temporal} palette={palette} />
            </Suspense>
            <Suspense fallback={<ViewFallback label="the impact charts" />}>
              <LcaView data={data.scenarios} scenario={scenario} palette={palette} />
            </Suspense>

            <footer className="colophon">
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
