import { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RobustnessPoint, SimulationData } from "../data/types";
import { agentColor, agentDash, orderAgents } from "../data/agents";
import { AXIS, GRID, Legend, Tip } from "../components/ChartKit";
import "./Robustness.css";

const NOISE_STEPS = [0, 5, 10, 20, 30];

type Rob = NonNullable<SimulationData["robustness"]>;

export default function Robustness({ data }: { data: SimulationData }) {
  const rob = data.robustness;

  if (!rob) {
    return (
      <section className="section" id="robustness">
        <div className="shell">
          <div className="sec-head">
            <span className="sec-index">03 / ROBUSTNESS</span>
            <h2 className="sec-title">Breaking the forecast</h2>
            <p className="sec-sub">
              The robustness sweep has not been exported yet. Train the agents
              with <code className="mono">python main.py</code>, then re-run{" "}
              <code className="mono">python export_web_data.py</code>.
            </p>
          </div>
        </div>
      </section>
    );
  }
  return <RobustnessView data={data} rob={rob} />;
}

function RobustnessView({ data, rob }: { data: SimulationData; rob: Rob }) {
  const [noise, setNoise] = useState(0);

  // Multi-grid exports carry the paper's metric (% of the 0%-error LP's
  // savings captured); the original study plots $ profit.
  const g = data.meta.grid;
  const useCap = !!(g && rob.capture);
  const curves: Record<string, RobustnessPoint[]> = useCap
    ? Object.fromEntries(
        Object.entries(rob.capture!).map(([a, pts]) => [
          a,
          pts.map((p) => ({
            noise: p.noise,
            meanProfit: p.capture,
            stdProfit: p.std ?? 0,
            minProfit: p.capture,
            maxProfit: p.capture,
          })),
        ])
      )
    : rob.curves;
  const fmtV = (v: number) =>
    useCap ? `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(0)}%` : money(v);

  const agents = orderAgents(Object.keys(curves));

  // Recharts has no native band mark: an Area whose dataKey is a [lo, hi]
  // tuple renders the ±1σ ribbon directly, with no stacking tricks.
  const curveRows = useMemo(() => {
    const noises = curves[agents[0]].map((p) => p.noise);
    return noises.map((n, i) => {
      const row: Record<string, number | [number, number]> = { noise: n };
      agents.forEach((a) => {
        const pt = curves[a][i];
        if (pt) {
          row[a] = pt.meanProfit;
          row[`${a}__band`] = [
            pt.meanProfit - pt.stdProfit,
            pt.meanProfit + pt.stdProfit,
          ];
        }
      });
      return row;
    });
  }, [curves, agents]);

  // snapshot at the selected noise level
  const snapshot = useMemo(() => {
    return agents
      .map((a) => {
        const pts = curves[a];
        const pt =
          pts.find((p) => Math.abs(p.noise - noise) < 0.01) ??
          pts.reduce((best, p) =>
            Math.abs(p.noise - noise) < Math.abs(best.noise - noise) ? p : best
          );
        const base = pts.find((p) => p.noise === 0) ?? pt;
        const delta = useCap
          ? pt.meanProfit - base.meanProfit
          : Math.abs(base.meanProfit) > 1e-6
          ? ((pt.meanProfit - base.meanProfit) / Math.abs(base.meanProfit)) * 100
          : 0;
        return { agent: a, ...pt, baseProfit: base.meanProfit, delta };
      })
      .sort((a, b) => b.meanProfit - a.meanProfit);
  }, [curves, agents, noise, useCap]);

  const leader = snapshot[0];
  const lpRank = snapshot.findIndex((s) => s.agent === "LP");

  const fcRows = useMemo(() => {
    const { solar, load, forecastDemo } = data.profiles;
    const k = noise / forecastDemo.noiseLevel;
    return data.profiles.hours.map((h) => ({
      h,
      fSolar: solar[h],
      fLoad: load[h],
      // interpolate between clean forecast and the exported 30% realisation
      aSolar: solar[h] + (forecastDemo.actualSolar[h] - solar[h]) * k,
      aLoad: load[h] + (forecastDemo.actualLoad[h] - load[h]) * k,
    }));
  }, [data.profiles, noise]);

  return (
    <section className="section" id="robustness">
      <div className="shell">
        <div className="sec-head">
          <span className="sec-index">03 / ROBUSTNESS</span>
          <h2 className="sec-title">Breaking the forecast</h2>
          {g ? (
            <p className="sec-sub">
              Perfect foresight is a fiction. Here every controller gets
              corrupted forecasts of load and solar, with correlated AR(1)
              error (ρ = 0.9) like real weather models make. The LP commits to
              a plan before the day starts, MPC re-plans every hour, and the RL
              agents re-decide every hour. Scores are the share of the
              clean-forecast LP's savings each method keeps. Drag the dial.
            </p>
          ) : (
            <p className="sec-sub">
              Perfect foresight is a fiction. Here both sides get corrupted
              forecasts — correlated AR(1) error, the kind real weather models
              make — and the LP has to live with a plan it committed to before
              the day started. Drag the dial.
            </p>
          )}
        </div>

        {/* ---- noise dial ---- */}
        <div className="rb-dial-bar">
          <div className="rb-dial-label">
            <span className="rb-dial-caption">Forecast error</span>
            <span className="rb-dial-read num">
              {noise}
              <small>%</small>
            </span>
          </div>

          <div className="rb-dial-track">
            <input
              type="range"
              min={0}
              max={30}
              step={1}
              value={noise}
              onChange={(e) => setNoise(Number(e.target.value))}
              style={{ ["--fill" as string]: `${(noise / 30) * 100}%` }}
            />
            <div className="rb-dial-stops">
              {NOISE_STEPS.map((s) => (
                <button
                  key={s}
                  className={`rb-stop${noise === s ? " is-on" : ""}`}
                  style={{ left: `${(s / 30) * 100}%` }}
                  onClick={() => setNoise(s)}
                >
                  <i />
                  <span className="mono">{s}%</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* ---- live verdict ---- */}
        <div className="rb-verdict" data-flipped={lpRank > 0}>
          <span className="rb-verdict-dot" />
          <p className="rb-verdict-text">
            {noise === 0 ? (
              g ? (
                <>
                  With a clean forecast the LP and MPC are unbeatable — they
                  solve the whole day exactly. The learned agents pay a
                  learning penalty.
                </>
              ) : (
                <>
                  With a clean forecast the LP is unbeatable — it knows the whole
                  day and solves it exactly. Everything else pays a learning
                  penalty.
                </>
              )
            ) : g && leader.agent !== "LP" ? (
              <>
                At <strong>{noise}% error</strong>,{" "}
                <strong style={{ color: agentColor(leader.agent) }}>
                  {leader.agent}
                </strong>{" "}
                leads with {fmtV(leader.meanProfit)} of the optimal savings, and
                the open-loop LP is at rank <strong>#{lpRank + 1}</strong>
                {leader.agent === "MPC"
                  ? ". Re-planning every hour on the latest forecast is enough here."
                  : ". The learned policy has absorbed how this grid punishes mistakes, which a plan built on a wrong forecast cannot do."}
              </>
            ) : lpRank > 0 ? (
              <>
                At <strong>{noise}% error</strong> the LP has dropped to rank{" "}
                <strong>#{lpRank + 1}</strong>.{" "}
                <strong style={{ color: agentColor(leader.agent) }}>
                  {leader.agent}
                </strong>{" "}
                now leads — it re-decides every hour, so a wrong forecast costs
                it one step instead of the whole schedule.
              </>
            ) : (
              <>
                At <strong>{noise}% error</strong> the LP still holds the lead,
                but its margin is shrinking as the plan it committed to drifts
                from reality.
              </>
            )}
          </p>
        </div>

        <div className="rb-grid">
          {/* ---- degradation curves ---- */}
          <div className="panel rb-wide">
            <div className="panel-head">
              <span className="panel-title">
                {useCap ? "Savings kept as the forecast degrades" : "Profit as the forecast degrades"}
              </span>
              <span className="panel-note">
                bands show ±1σ over {g ? `${g.robustnessRuns} runs` : "10 scenarios"}
              </span>
            </div>
            <div className="panel-body">
              <div className="rb-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={curveRows}
                    margin={{ top: 6, right: 12, bottom: 0, left: -10 }}
                  >
                    <CartesianGrid {...GRID} />
                    <XAxis
                      dataKey="noise"
                      {...AXIS}
                      tickLine={false}
                      unit="%"
                      type="number"
                      domain={[0, 30]}
                      ticks={NOISE_STEPS}
                    />
                    <YAxis {...AXIS} tickLine={false} width={46} unit={useCap ? "%" : undefined} />
                    <Tooltip
                      content={({ active, payload, label }) =>
                        active && payload?.length ? (
                          <Tip
                            title={`${label}% forecast error`}
                            rows={agents
                              .map((a) => ({
                                a,
                                v: payload[0].payload[a] as number,
                              }))
                              .filter((r) => r.v !== undefined)
                              .sort((x, y) => y.v - x.v)
                              .map((r) => ({
                                label: r.a,
                                value: useCap ? fmtV(r.v) : `${r.v < 0 ? "−" : ""}$${Math.abs(r.v).toFixed(3)}`,
                                color: agentColor(r.a),
                              }))}
                          />
                        ) : null
                      }
                    />
                    {useCap && <ReferenceLine y={0} stroke="#2f4055" />}
                    {useCap && (
                      <ReferenceLine y={100} stroke="#2f4055" strokeDasharray="2 4" />
                    )}
                    <ReferenceLine
                      x={noise}
                      stroke="#e4ebf2"
                      strokeOpacity={0.32}
                      strokeWidth={1.5}
                    />
                    {agents.map((a) => (
                      <Area
                        key={`${a}-band`}
                        dataKey={`${a}__band`}
                        stroke="none"
                        fill={agentColor(a)}
                        fillOpacity={0.13}
                        isAnimationActive={false}
                        legendType="none"
                        activeDot={false}
                      />
                    ))}
                    {agents.map((a) => (
                      <Line
                        key={a}
                        dataKey={a}
                        stroke={agentColor(a)}
                        strokeWidth={a === "LP" ? 2.4 : 2}
                        strokeDasharray={agentDash(a)}
                        dot={{ r: 2.8, strokeWidth: 0, fill: agentColor(a) }}
                        activeDot={{ r: 4.5, stroke: "#0b0f14", strokeWidth: 2 }}
                        isAnimationActive={false}
                      />
                    ))}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <Legend
                items={agents.map((a) => ({
                  color: agentColor(a),
                  label: a === "LP" && g ? "LP (open loop)" : a,
                  dashed: !!agentDash(a),
                }))}
              />
            </div>
          </div>

          {/* ---- ranking at this noise ---- */}
          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">Standing at {noise}% error</span>
              <span className="panel-note">
                {useCap ? "change in points vs 0% error" : "vs own clean-forecast baseline"}
              </span>
            </div>
            <div className="rb-rank">
              {snapshot.map((s, i) => {
                // Profits are negative (they are costs), so rank by how close
                // to zero: the best agent earns the longest bar, not the worst.
                const vals = snapshot.map((x) => x.meanProfit);
                const hi = Math.max(...vals);
                const lo = Math.min(...vals);
                const span = hi - lo;
                const w =
                  span > 1e-9 ? 12 + ((s.meanProfit - lo) / span) * 88 : 100;
                return (
                  <div className="rb-rank-row" key={s.agent}>
                    <span className="rb-rank-pos mono">#{i + 1}</span>
                    <span className="rb-rank-name">
                      <i style={{ background: agentColor(s.agent) }} />
                      {s.agent === "LP" && g ? "LP (open loop)" : s.agent}
                    </span>
                    <div className="rb-rank-bar">
                      <div
                        className="rb-rank-fill"
                        style={{
                          width: `${w}%`,
                          background: agentColor(s.agent),
                        }}
                      />
                    </div>
                    <span className="rb-rank-val num">{fmtV(s.meanProfit)}</span>
                    <span
                      className={`rb-rank-delta mono ${
                        s.delta < -0.5 ? "neg" : s.delta > 0.5 ? "pos" : "dim"
                      }`}
                    >
                      {noise === 0
                        ? "base"
                        : `${s.delta > 0 ? "+" : ""}${s.delta.toFixed(0)}${useCap ? "" : "%"}`}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ---- forecast vs actual ---- */}
          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">What the agent was told</span>
              <span className="panel-note">forecast vs realised</span>
            </div>
            <div className="panel-body">
              <div className="rb-chart rb-chart--short">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={fcRows} margin={{ top: 6, right: 10, bottom: 0, left: -14 }}>
                    <CartesianGrid {...GRID} />
                    <XAxis dataKey="h" {...AXIS} tickLine={false} interval={5} />
                    <YAxis {...AXIS} tickLine={false} width={40} unit="kW" />
                    <Tooltip
                      content={({ active, payload, label }) =>
                        active && payload?.length ? (
                          <Tip
                            title={`${String(label).padStart(2, "0")}:00`}
                            rows={payload.map((p) => ({
                              label: String(p.name),
                              value: `${Number(p.value).toFixed(2)} kW`,
                              color: p.color,
                              dim: String(p.dataKey).startsWith("f"),
                            }))}
                          />
                        ) : null
                      }
                    />
                    <Line
                      dataKey="fSolar"
                      name="Solar forecast"
                      stroke="#f5c242"
                      strokeWidth={1.3}
                      strokeDasharray="4 3"
                      strokeOpacity={0.55}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      dataKey="aSolar"
                      name="Solar actual"
                      stroke="#f5c242"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      dataKey="fLoad"
                      name="Load forecast"
                      stroke="#ff7a66"
                      strokeWidth={1.3}
                      strokeDasharray="4 3"
                      strokeOpacity={0.55}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      dataKey="aLoad"
                      name="Load actual"
                      stroke="#ff7a66"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <Legend
                compact
                items={[
                  { label: "Solar actual", color: "#f5c242" },
                  { label: "Solar forecast", color: "#f5c242", dashed: true },
                  { label: "Load actual", color: "#ff7a66" },
                  { label: "Load forecast", color: "#ff7a66", dashed: true },
                ]}
              />
              <p className="rb-read">
                Error is correlated hour to hour (ρ = 0.9), not independent
                noise — a forecast that is wrong at noon is usually still wrong
                at 13:00. That is what makes a committed schedule dangerous.
              </p>
            </div>
          </div>
        </div>

        {/* ---- summary table ---- */}
        <div className="panel rb-summary">
          <div className="panel-head">
            <span className="panel-title">Robustness scorecard</span>
            <span className="panel-note">0% → 30% forecast error</span>
          </div>
          <div className="rb-summary-scroll">
            <table className="cmp-table">
              <thead>
                <tr>
                  <th>Agent</th>
                  <th className="ta-r">Profit @ 0% noise</th>
                  <th className="ta-r">Profit @ 30% noise</th>
                  {useCap && <th className="ta-r">Savings kept @ 30%</th>}
                  <th className="ta-r">Degradation</th>
                  <th className="ta-r">Robustness score</th>
                </tr>
              </thead>
              <tbody>
                {rob.summary.map((r) => {
                  const a = String(r.Agent);
                  // Degradation is (baseline − worst)/|baseline|: positive means
                  // the agent lost ground as the forecast got worse.
                  const deg = Number(r["Degradation (%)"]);
                  return (
                    <tr key={a} className={a === "LP" ? "is-bench" : undefined}>
                      <td>
                        <span className="cmp-table-agent">
                          <i style={{ background: agentColor(a) }} />
                          {a === "LP" && g ? "LP (open loop)" : a}
                        </span>
                      </td>
                      <td className="ta-r num">
                        {money(Number(r["Profit @ 0% Noise"]))}
                      </td>
                      <td className="ta-r num">
                        {money(Number(r["Profit @ 30% Noise"]))}
                      </td>
                      {useCap && (
                        <td className="ta-r num">{fmtV(Number(r["Capture @ 30% (%)"]))}</td>
                      )}
                      <td className="ta-r num">
                        <span className={deg > 5 ? "neg" : deg < 0 ? "pos" : undefined}>
                          {deg > 0 ? "−" : "+"}
                          {Math.abs(deg).toFixed(1)}%
                        </span>
                      </td>
                      <td className="ta-r num">
                        {Number(r["Robustness Score"]).toFixed(1)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}

const money = (v: number) =>
  `${v < 0 ? "−" : ""}$${Math.abs(v).toFixed(Math.abs(v) >= 100 ? 0 : 2)}`;
