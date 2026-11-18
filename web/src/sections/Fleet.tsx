import { useState } from "react";
import type { GridIndex, Source } from "../data/types";
import { agentColor } from "../data/agents";
import "./Fleet.css";

type View = "clean" | "noisy" | "transfer";

const VIEWS: { id: View; label: string; note: string }[] = [
  {
    id: "clean",
    label: "Clean forecast",
    note: "Specialists, generalist and zero-shot transfer on held-out test days.",
  },
  {
    id: "noisy",
    label: "30% forecast error",
    note: "Correlated AR(1) error on load and PV. LP = plan on the forecast, executed open loop.",
  },
  {
    id: "transfer",
    label: "Cross-grid transfer",
    note: "SAC specialist trained on the row grid, tested unchanged on the column grid.",
  },
];

/** Diverging fill: blue toward 100 (optimal), red below 0 (worse than idle). */
function cellFill(v: number | undefined) {
  if (v === undefined || Number.isNaN(v)) return "transparent";
  if (v >= 0) {
    const a = 0.08 + Math.min(v, 100) / 100 * 0.5;
    return `rgba(46, 125, 209, ${a.toFixed(3)})`;
  }
  const a = 0.08 + Math.min(-v, 200) / 200 * 0.5;
  return `rgba(232, 87, 107, ${a.toFixed(3)})`;
}

const fmtCap = (v: number | undefined) =>
  v === undefined || Number.isNaN(v) ? "—" : `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(0)}`;

export default function Fleet({
  index,
  source,
  onSource,
}: {
  index: GridIndex;
  source: Source;
  onSource: (s: Source) => void;
}) {
  const [view, setView] = useState<View>("clean");
  const grids = index.grids;

  const rows: { label: string; color?: string; values: (number | undefined)[] }[] =
    view === "clean"
      ? index.methods
          .filter((m) => grids.some((g) => m in g.capture))
          .map((m) => ({ label: m, color: agentColor(m), values: grids.map((g) => g.capture[m]) }))
      : view === "noisy"
      ? index.robustMethods.map((m) => ({
          label: m === "LP" ? "LP (open loop)" : m,
          color: agentColor(m),
          values: grids.map((g) => g.capture30[m]),
        }))
      : grids.map((tr) => ({
          label: `trained on ${tr.id}`,
          values: grids.map((te) => index.transfer[tr.id]?.[te.id]),
        }));

  return (
    <section className="section" id="grids">
      <div className="shell">
        <div className="sec-head">
          <span className="sec-index">00 / SIX MICROGRIDS</span>
          <h2 className="sec-title">One method, six very different sites</h2>
          <p className="sec-sub">
            The same learning pipeline runs on six microgrid archetypes, each
            with its own battery, solar array, tariff and grid limits, and a
            full year of real 2018 load, weather and price data. Pick a grid
            here or in the top bar, and every section below switches to it.
          </p>
        </div>

        <div className="fl-cards" role="radiogroup" aria-label="Choose a microgrid">
          <button
            role="radio"
            aria-checked={source === "base"}
            className={`fl-card fl-card--base${source === "base" ? " is-on" : ""}`}
            onClick={() => onSource("base")}
          >
            <span className="fl-card-top">
              <span className="fl-card-id mono">HOME</span>
              <span className="fl-card-name">Original single-home study</span>
            </span>
            <span className="fl-card-desc">
              One Pecan Street day, Stable-Baselines3 agents and the forecast-error
              stress test from the first version of the project.
            </span>
          </button>
          {grids.map((g) => {
            const sac = g.capture.SAC;
            return (
              <button
                key={g.id}
                role="radio"
                aria-checked={source === g.id}
                className={`fl-card${source === g.id ? " is-on" : ""}`}
                onClick={() => onSource(g.id)}
              >
                <span className="fl-card-top">
                  <span className="fl-card-id mono">{g.id}</span>
                  <span className="fl-card-name">{g.name}</span>
                </span>
                <span className="fl-card-desc">{g.description}</span>
                <span className="fl-card-specs mono">
                  <span>{g.batteryKwh} kWh / {g.batteryKw} kW</span>
                  <span>{g.pvKwp} kWp PV</span>
                  <span>{g.tariff}</span>
                </span>
                <span className="fl-card-score">
                  <span className="fl-card-score-val num">{fmtCap(sac)}%</span>
                  <span className="fl-card-score-label">of optimal savings · SAC</span>
                </span>
              </button>
            );
          })}
        </div>

        <div className="panel fl-panel">
          <div className="panel-head fl-head">
            <span className="panel-title">Share of the optimal savings captured</span>
            <div className="fl-tabs" role="tablist">
              {VIEWS.map((v) => (
                <button
                  key={v.id}
                  role="tab"
                  aria-selected={view === v.id}
                  className={`fl-tab${view === v.id ? " is-on" : ""}`}
                  onClick={() => setView(v.id)}
                >
                  {v.label}
                </button>
              ))}
            </div>
          </div>
          <div className="fl-scroll">
            <table className="fl-table">
              <thead>
                <tr>
                  <th className="fl-rowhead">{view === "transfer" ? "Policy" : "Method"}</th>
                  {grids.map((g) => (
                    <th key={g.id} className={source === g.id ? "is-on" : undefined}>
                      <button className="fl-colbtn" onClick={() => onSource(g.id)} title={g.name}>
                        <span className="mono">{g.id}</span>
                        <small>{g.name}</small>
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.label}>
                    <th className="fl-rowhead" scope="row">
                      <span className="fl-method">
                        {r.color && <i style={{ background: r.color }} />}
                        {r.label}
                      </span>
                    </th>
                    {r.values.map((v, j) => (
                      <td
                        key={grids[j].id}
                        className={`num${source === grids[j].id ? " is-on" : ""}`}
                        style={{ background: cellFill(v) }}
                        title={`${r.label} on ${grids[j].id} (${grids[j].name}): ${fmtCap(v)}% of LP savings`}
                      >
                        {fmtCap(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="fl-foot">
            <div className="fl-scale" aria-hidden="true">
              <span className="mono">−200</span>
              <i />
              <span className="mono">0</span>
              <i className="fl-scale-pos" />
              <span className="mono">100</span>
            </div>
            <p className="fl-note">
              {VIEWS.find((v) => v.id === view)!.note} {index.note}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
