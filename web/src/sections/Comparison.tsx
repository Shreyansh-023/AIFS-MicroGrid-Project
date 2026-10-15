import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend as RLegend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SimulationData } from "../data/types";
import { AGENT_BLURB, AGENT_KIND, agentColor, orderAgents } from "../data/agents";
import { AXIS, GRID, Tip, TouBands } from "../components/ChartKit";
import "./Comparison.css";

export default function Comparison({ data }: { data: SimulationData }) {
  const agents = orderAgents(
    data.summary.map((r) => String(r.Agent)).filter((a) => a in data.trajectories)
  );

  const [shown, setShown] = useState<Set<string>>(
    () => new Set(agents.filter((a) => a !== "NoOp"))
  );

  const toggle = (a: string) =>
    setShown((prev) => {
      const next = new Set(prev);
      if (next.has(a)) {
        if (next.size > 1) next.delete(a);
      } else next.add(a);
      return next;
    });

  const visible = agents.filter((a) => shown.has(a));

  const socRows = useMemo(
    () =>
      data.profiles.hours.map((h) => {
        const row: Record<string, number> = { h };
        agents.forEach((a) => {
          row[a] = (data.trajectories[a].soc[h] ?? 0) * 100;
        });
        return row;
      }),
    [data, agents]
  );

  const profitRows = useMemo(
    () =>
      data.profiles.hours.map((h) => {
        const row: Record<string, number | null> = { h };
        agents.forEach((a) => {
          const cp = data.trajectories[a].cumulativeProfit;
          row[a] = cp ? cp[h] ?? null : null;
        });
        return row;
      }),
    [data, agents]
  );

  const barRows = useMemo(
    () =>
      orderAgents(data.summary.map((r) => String(r.Agent))).map((a) => {
        const row = data.summary.find((r) => String(r.Agent) === a)!;
        return {
          agent: a,
          profit: Number(row["Mean Profit ($)"]),
          err: Number(row["Std Profit ($)"] ?? 0),
        };
      }),
    [data]
  );

  const breakdownRows = useMemo(
    () =>
      agents
        .filter((a) => data.trajectories[a].energyProfit !== undefined)
        .map((a) => {
          const t = data.trajectories[a];
          return {
            agent: a,
            energy: t.energyProfit ?? 0,
            peak: -(t.peakPenalty ?? 0),
            degradation: -(t.degradationCost ?? 0),
          };
        }),
    [data, agents]
  );

  return (
    <section className="section" id="compare">
      <div className="shell">
        <div className="sec-head">
          <span className="sec-index">02 / HEAD TO HEAD</span>
          <h2 className="sec-title">Same day, five strategies</h2>
          <p className="sec-sub">
            Every agent sees the identical {data.meta.evalEpisodes}-episode
            sweep, with starting charge spread from 20% to 80%. Click a name to
            add or remove it from the charts.
          </p>
        </div>

        <div className="cmp-toggles">
          {agents.map((a) => (
            <button
              key={a}
              className={`cmp-tog${shown.has(a) ? " is-on" : ""}`}
              style={{ ["--c" as string]: agentColor(a) } as React.CSSProperties}
              onClick={() => toggle(a)}
            >
              <i />
              <span className="cmp-tog-name">{a}</span>
              <span className="cmp-tog-kind mono">{AGENT_KIND[a]}</span>
            </button>
          ))}
        </div>

        <div className="cmp-grid">
          <div className="panel cmp-wide">
            <div className="panel-head">
              <span className="panel-title">State of charge over 24 h</span>
              <span className="panel-note">
                start {(data.meta.trajectoryInitialSoc * 100).toFixed(0)}% · shaded by TOU band
              </span>
            </div>
            <div className="panel-body">
              <div className="cmp-chart cmp-chart--tall">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={socRows} margin={{ top: 4, right: 10, bottom: 0, left: -12 }}>
                    <CartesianGrid {...GRID} />
                    <TouBands />
                    <XAxis dataKey="h" {...AXIS} tickLine={false} />
                    <YAxis domain={[0, 100]} {...AXIS} tickLine={false} width={44} unit="%" />
                    <ReferenceLine
                      y={data.meta.config.minSocRatio * 100}
                      stroke="#e8576b"
                      strokeDasharray="3 3"
                      strokeOpacity={0.45}
                      label={{
                        value: "floor",
                        position: "insideBottomLeft",
                        fill: "#e8576b",
                        fontSize: 10,
                      }}
                    />
                    <Tooltip
                      content={({ active, payload, label }) =>
                        active && payload?.length ? (
                          <Tip
                            title={`${String(label).padStart(2, "0")}:00`}
                            rows={payload
                              .slice()
                              .sort((a, b) => Number(b.value) - Number(a.value))
                              .map((p) => ({
                                label: String(p.dataKey),
                                value: `${Number(p.value).toFixed(1)}%`,
                                color: p.color,
                              }))}
                          />
                        ) : null
                      }
                    />
                    {visible.map((a) => (
                      <Line
                        key={a}
                        dataKey={a}
                        stroke={agentColor(a)}
                        strokeWidth={a === "LP" ? 1.6 : 2}
                        strokeDasharray={a === "LP" ? "5 3" : undefined}
                        dot={false}
                        isAnimationActive={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="cmp-read">
                The LP dashed line is the optimum. Watch where the learned
                policies diverge from it — usually at the on-peak boundary,
                where committing early costs less than being caught short.
              </p>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">Cumulative profit</span>
              <span className="panel-note">$ over the day</span>
            </div>
            <div className="panel-body">
              <div className="cmp-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={profitRows} margin={{ top: 4, right: 10, bottom: 0, left: -12 }}>
                    <CartesianGrid {...GRID} />
                    <TouBands />
                    <XAxis dataKey="h" {...AXIS} tickLine={false} interval={3} />
                    <YAxis {...AXIS} tickLine={false} width={44} />
                    <ReferenceLine y={0} stroke="#2f4055" />
                    <Tooltip
                      content={({ active, payload, label }) =>
                        active && payload?.length ? (
                          <Tip
                            title={`${String(label).padStart(2, "0")}:00`}
                            rows={payload
                              .slice()
                              .sort((a, b) => Number(b.value) - Number(a.value))
                              .map((p) => ({
                                label: String(p.dataKey),
                                value: `${Number(p.value) < 0 ? "−" : ""}$${Math.abs(
                                  Number(p.value)
                                ).toFixed(2)}`,
                                color: p.color,
                              }))}
                          />
                        ) : null
                      }
                    />
                    {visible.map((a) => (
                      <Line
                        key={a}
                        dataKey={a}
                        stroke={agentColor(a)}
                        strokeWidth={a === "LP" ? 1.6 : 2}
                        strokeDasharray={a === "LP" ? "5 3" : undefined}
                        dot={false}
                        connectNulls
                        isAnimationActive={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">Mean profit across episodes</span>
              <span className="panel-note">
                n = {data.meta.evalEpisodes} · bars show ±1σ
              </span>
            </div>
            <div className="panel-body">
              <div className="cmp-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barRows} margin={{ top: 4, right: 10, bottom: 0, left: -12 }}>
                    <CartesianGrid {...GRID} />
                    <XAxis dataKey="agent" {...AXIS} tickLine={false} interval={0} />
                    <YAxis {...AXIS} tickLine={false} width={44} />
                    <ReferenceLine y={0} stroke="#2f4055" />
                    <Tooltip
                      cursor={{ fill: "rgba(255,255,255,0.03)" }}
                      content={({ active, payload }) =>
                        active && payload?.length ? (
                          <Tip
                            title={String(payload[0].payload.agent)}
                            rows={[
                              {
                                label: "Mean profit",
                                value: `${
                                  payload[0].payload.profit < 0 ? "−" : ""
                                }$${Math.abs(payload[0].payload.profit).toFixed(3)}`,
                              },
                              {
                                label: "Std dev",
                                value: `$${payload[0].payload.err.toFixed(3)}`,
                                dim: true,
                              },
                            ]}
                          />
                        ) : null
                      }
                    />
                    <Bar dataKey="profit" radius={[3, 3, 0, 0]} maxBarSize={44}>
                      {barRows.map((r) => (
                        <Cell key={r.agent} fill={agentColor(r.agent)} fillOpacity={0.85} />
                      ))}
                      <ErrorBar
                        dataKey="err"
                        width={4}
                        strokeWidth={1.2}
                        stroke="#9eb0c2"
                      />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {breakdownRows.length > 0 && (
            <div className="panel cmp-wide">
              <div className="panel-head">
                <span className="panel-title">Where the money goes</span>
                <span className="panel-note">
                  single episode @ {(data.meta.trajectoryInitialSoc * 100).toFixed(0)}% start
                </span>
              </div>
              <div className="panel-body">
                <div className="cmp-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={breakdownRows}
                      margin={{ top: 4, right: 10, bottom: 0, left: -12 }}
                    >
                      <CartesianGrid {...GRID} />
                      <XAxis dataKey="agent" {...AXIS} tickLine={false} interval={0} />
                      <YAxis {...AXIS} tickLine={false} width={44} />
                      <ReferenceLine y={0} stroke="#2f4055" />
                      <Tooltip
                        cursor={{ fill: "rgba(255,255,255,0.03)" }}
                        content={({ active, payload }) =>
                          active && payload?.length ? (
                            <Tip
                              title={String(payload[0].payload.agent)}
                              rows={payload.map((p) => ({
                                label: String(p.name),
                                value: `${Number(p.value) < 0 ? "−" : "+"}$${Math.abs(
                                  Number(p.value)
                                ).toFixed(3)}`,
                                color: p.color,
                              }))}
                            />
                          ) : null
                        }
                      />
                      <RLegend
                        wrapperStyle={{ fontSize: 11.5, paddingTop: 6 }}
                        iconType="square"
                        iconSize={9}
                      />
                      <Bar
                        dataKey="energy"
                        name="Energy arbitrage"
                        stackId="a"
                        fill="#4da3ff"
                        fillOpacity={0.82}
                        radius={[3, 3, 0, 0]}
                        maxBarSize={52}
                      />
                      <Bar
                        dataKey="degradation"
                        name="Degradation"
                        stackId="a"
                        fill="#f0a23c"
                        fillOpacity={0.82}
                        maxBarSize={52}
                      />
                      <Bar
                        dataKey="peak"
                        name="Peak penalty"
                        stackId="a"
                        fill="#e8576b"
                        fillOpacity={0.82}
                        maxBarSize={52}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="cmp-read">
                  Degradation is the cost of cycling the pack — an agent that
                  trades aggressively pays for it here. Peak penalty only bites
                  when grid draw crosses{" "}
                  {data.meta.config.peakThresholdKw} kW.
                </p>
              </div>
            </div>
          )}
        </div>

        <ScoreTable data={data} />
      </div>
    </section>
  );
}

function ScoreTable({ data }: { data: SimulationData }) {
  const rows = orderAgents(data.summary.map((r) => String(r.Agent))).map(
    (a) => data.summary.find((r) => String(r.Agent) === a)!
  );

  const num = (r: Record<string, string | number>, k: string) =>
    r[k] === undefined || r[k] === null ? null : Number(r[k]);

  return (
    <div className="panel cmp-table-panel">
      <div className="panel-head">
        <span className="panel-title">Full scoreboard</span>
        <span className="panel-note">
          averaged over {data.meta.evalEpisodes} episodes
        </span>
      </div>
      <div className="cmp-table-scroll">
        <table className="cmp-table">
          <thead>
            <tr>
              <th>Agent</th>
              <th>Kind</th>
              <th className="ta-r">Mean profit</th>
              <th className="ta-r">σ</th>
              <th className="ta-r">Gap vs LP</th>
              <th className="ta-r">Throughput</th>
              <th className="ta-r">Final SOC</th>
              <th className="ta-r">Eval time</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const a = String(r.Agent);
              const gap = num(r, "Gap vs LP (%)");
              return (
                <tr key={a} className={a === "LP" ? "is-bench" : undefined}>
                  <td>
                    <span className="cmp-table-agent">
                      <i style={{ background: agentColor(a) }} />
                      {a}
                    </span>
                  </td>
                  <td className="cmp-table-kind">{AGENT_KIND[a]}</td>
                  <td className="ta-r num">
                    {fmtMoney(num(r, "Mean Profit ($)"))}
                  </td>
                  <td className="ta-r num dim">
                    {fmtPlain(num(r, "Std Profit ($)"))}
                  </td>
                  <td className="ta-r num">
                    {a === "LP" ? (
                      <span className="dim">benchmark</span>
                    ) : gap === null ? (
                      "—"
                    ) : (
                      <span className={gap > 1000 ? "neg" : undefined}>
                        {gap.toFixed(1)}%
                      </span>
                    )}
                  </td>
                  <td className="ta-r num dim">
                    {fmtPlain(num(r, "Mean Throughput (kWh)"))}
                  </td>
                  <td className="ta-r num dim">
                    {(() => {
                      const v = num(r, "Mean Final SOC");
                      return v === null ? "—" : `${(v * 100).toFixed(0)}%`;
                    })()}
                  </td>
                  <td className="ta-r num dim">
                    {(() => {
                      const v = num(r, "Eval Time (s)");
                      return v === null ? "—" : `${v.toFixed(2)}s`;
                    })()}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="cmp-table-foot">
        {orderAgents(Object.keys(AGENT_BLURB))
          .filter((a) => data.summary.some((r) => String(r.Agent) === a))
          .map((a) => (
            <div className="cmp-note" key={a}>
              <span className="cmp-note-head">
                <i style={{ background: agentColor(a) }} />
                {a}
              </span>
              <span className="cmp-note-body">{AGENT_BLURB[a]}</span>
            </div>
          ))}
      </div>
    </div>
  );
}

const fmtMoney = (v: number | null) =>
  v === null ? "—" : `${v < 0 ? "−" : ""}$${Math.abs(v).toFixed(3)}`;

const fmtPlain = (v: number | null) => (v === null ? "—" : v.toFixed(2));
