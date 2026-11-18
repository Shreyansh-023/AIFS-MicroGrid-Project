import { useCallback, useEffect, useState } from "react";
import Nav from "./components/Nav";
import Hero from "./sections/Hero";
import Fleet from "./sections/Fleet";
import Replay from "./sections/Replay";
import Comparison from "./sections/Comparison";
import Robustness from "./sections/Robustness";
import System from "./sections/System";
import { useGridIndex, useSimulation } from "./data/useSimulation";
import { BandsContext, TOU_BANDS } from "./data/agents";
import type { SimulationData, Source } from "./data/types";
import "./App.css";

/** ?grid=G3 selects a grid; no parameter (or ?grid=home) is the original study. */
function initialSource(): Source {
  const p = new URLSearchParams(window.location.search).get("grid");
  return p && /^G\d+$/i.test(p) ? p.toUpperCase() : "base";
}

export default function App() {
  const [source, setSource] = useState<Source>(initialSource);
  const index = useGridIndex();
  const state = useSimulation(source);

  const choose = useCallback((s: Source) => setSource(s), []);

  useEffect(() => {
    const url = new URL(window.location.href);
    if (source === "base") url.searchParams.delete("grid");
    else url.searchParams.set("grid", source);
    window.history.replaceState(null, "", url);
  }, [source]);

  if (state.status === "loading") return <Booting />;
  if (state.status === "error")
    return <Failed message={state.message} source={source} onReset={() => choose("base")} />;

  const { data } = state;
  const shown = state.source;

  return (
    <BandsContext.Provider value={data.profiles.bands ?? TOU_BANDS}>
      <div id="top" className={state.switching ? "is-switching" : undefined}>
        <Nav source={source} onSource={choose} index={index} switching={state.switching} />
        <Hero data={data} />
        <main>
          {index && <Fleet index={index} source={source} onSource={choose} />}
          {/* keyed so per-section state (selected agent, hour, noise) resets per grid */}
          <div key={shown}>
            <Replay data={data} />
            <Comparison data={data} />
            <Robustness data={data} />
            <System data={data} />
          </div>
        </main>
        <Footer data={data} />
      </div>
    </BandsContext.Provider>
  );
}

function Booting() {
  return (
    <div className="boot">
      <div className="boot-bars" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
      </div>
      <p className="boot-text mono">Loading simulation results…</p>
    </div>
  );
}

function Failed({
  message,
  source,
  onReset,
}: {
  message: string;
  source: Source;
  onReset: () => void;
}) {
  const multi = source !== "base";
  return (
    <div className="boot">
      <p className="boot-title">No simulation data</p>
      <p className="boot-text">
        Could not read{" "}
        <code className="mono">
          public/data/{multi ? `grids/${source}.json` : "simulation.json"}
        </code>
        {" — "}
        <span className="mono">{message}</span>
      </p>
      <div className="boot-fix">
        <span className="boot-fix-label mono">Generate it:</span>
        {multi ? (
          <>
            <code className="mono">python run_multigrid.py</code>
            <code className="mono">python export_multigrid_web.py</code>
          </>
        ) : (
          <>
            <code className="mono">python main.py</code>
            <code className="mono">python export_web_data.py</code>
          </>
        )}
      </div>
      {multi && (
        <button className="boot-back" onClick={onReset}>
          Back to the original study
        </button>
      )}
    </div>
  );
}

function Footer({ data }: { data: SimulationData }) {
  const trained = data.meta.agentsTrained;
  const g = data.meta.grid;
  return (
    <footer className="foot">
      <div className="shell foot-inner">
        <div className="foot-col">
          <p className="foot-title">Microgrid energy management with RL</p>
          <p className="foot-body">
            {g ? (
              <>
                {g.id} · {g.name}: SAC, PPO and their variants dispatching a{" "}
                {data.meta.config.batteryCapacityKwh} kWh battery against a
                perfect-foresight LP and hourly MPC, scored on {g.nTestDays}{" "}
                held-out test days of 2018 data.
              </>
            ) : (
              <>
                SAC, PPO and domain-randomised SAC dispatching a{" "}
                {data.meta.config.batteryCapacityKwh} kWh battery against a
                perfect-foresight linear program, evaluated over{" "}
                {data.meta.evalEpisodes} episodes and a five-point
                forecast-error sweep.
              </>
            )}
          </p>
        </div>

        <div className="foot-col foot-col--meta">
          <FootRow label="Agents trained" value={trained.join(", ") || "none"} />
          <FootRow
            label={g ? "Test days" : "Eval episodes"}
            value={String(data.meta.evalEpisodes)}
          />
          <FootRow
            label="Observation"
            value={`${g ? g.obsDim : 131} dims · ${data.meta.config.forecastHorizonH} h horizon`}
          />
          <FootRow
            label="Stack"
            value={
              g
                ? "PyTorch GPU-batched env · custom SAC/PPO · SciPy HiGHS"
                : "Gymnasium · Stable-Baselines3 · SciPy HiGHS"
            }
          />
        </div>
      </div>

      <div className="shell foot-rule">
        <span className="mono">
          {g
            ? `Battery dispatch across six microgrid archetypes · ${g.seeds} seeds`
            : `Battery dispatch under forecast uncertainty · ${data.meta.evalEpisodes}-episode evaluation`}
        </span>
        <span className="mono">
          {data.robustness
            ? `0–30% forecast error · ${g ? `${g.robustnessRuns} runs` : "10 scenarios"} per level`
            : "robustness sweep pending"}
        </span>
      </div>
    </footer>
  );
}

function FootRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="foot-row">
      <span className="foot-row-label">{label}</span>
      <span className="foot-row-value mono">{value}</span>
    </div>
  );
}
