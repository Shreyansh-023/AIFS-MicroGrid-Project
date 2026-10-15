import Nav from "./components/Nav";
import Hero from "./sections/Hero";
import Replay from "./sections/Replay";
import Comparison from "./sections/Comparison";
import Robustness from "./sections/Robustness";
import System from "./sections/System";
import { useSimulation } from "./data/useSimulation";
import type { SimulationData } from "./data/types";
import "./App.css";

export default function App() {
  const state = useSimulation();

  if (state.status === "loading") return <Booting />;
  if (state.status === "error") return <Failed message={state.message} />;

  const { data } = state;

  return (
    <div id="top">
      <Nav />
      <Hero data={data} />
      <main>
        <Replay data={data} />
        <Comparison data={data} />
        <Robustness data={data} />
        <System data={data} />
      </main>
      <Footer data={data} />
    </div>
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

function Failed({ message }: { message: string }) {
  return (
    <div className="boot">
      <p className="boot-title">No simulation data</p>
      <p className="boot-text">
        Could not read <code className="mono">public/data/simulation.json</code>
        {" — "}
        <span className="mono">{message}</span>
      </p>
      <div className="boot-fix">
        <span className="boot-fix-label mono">Generate it:</span>
        <code className="mono">python main.py</code>
        <code className="mono">python export_web_data.py</code>
      </div>
    </div>
  );
}

function Footer({ data }: { data: SimulationData }) {
  const trained = data.meta.agentsTrained;
  return (
    <footer className="foot">
      <div className="shell foot-inner">
        <div className="foot-col">
          <p className="foot-title">Microgrid energy management with RL</p>
          <p className="foot-body">
            SAC, PPO and domain-randomised SAC dispatching a{" "}
            {data.meta.config.batteryCapacityKwh} kWh battery against a
            perfect-foresight linear program, evaluated over{" "}
            {data.meta.evalEpisodes} episodes and a five-point forecast-error
            sweep.
          </p>
        </div>

        <div className="foot-col foot-col--meta">
          <FootRow label="Agents trained" value={trained.join(", ") || "none"} />
          <FootRow label="Eval episodes" value={String(data.meta.evalEpisodes)} />
          <FootRow
            label="Observation"
            value={`131 dims · ${data.meta.config.forecastHorizonH} h horizon`}
          />
          <FootRow
            label="Stack"
            value="Gymnasium · Stable-Baselines3 · SciPy HiGHS"
          />
        </div>
      </div>

      <div className="shell foot-rule">
        <span className="mono">
          Battery dispatch under forecast uncertainty · {data.meta.evalEpisodes}
          -episode evaluation
        </span>
        <span className="mono">
          {data.robustness
            ? "0–30% forecast error · 10 scenarios per level"
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
