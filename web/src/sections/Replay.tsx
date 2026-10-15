import { useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import type { SimulationData } from "../data/types";
import { agentColor, orderAgents, touAt } from "../data/agents";
import { AXIS, GRID, TouBands, TouKey } from "../components/ChartKit";
import "./Replay.css";

const SPEEDS = [
  { label: "0.5×", ms: 1300 },
  { label: "1×", ms: 680 },
  { label: "2×", ms: 330 },
  { label: "4×", ms: 160 },
];

export default function Replay({ data }: { data: SimulationData }) {
  const available = orderAgents(Object.keys(data.trajectories));
  const [agent, setAgent] = useState(
    available.includes("R-SAC") ? "R-SAC" : available[0]
  );
  const [hour, setHour] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedIdx, setSpeedIdx] = useState(1);
  const [ghost, setGhost] = useState(true);

  const traj = data.trajectories[agent];
  const lp = data.trajectories.LP;
  const { solar, load, price } = data.profiles;
  const cfg = data.meta.config;

  // advance the clock
  const raf = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setHour((h) => {
        if (h >= 23) {
          setPlaying(false);
          return 23;
        }
        return h + 1;
      });
    }, SPEEDS[speedIdx].ms);
    raf.current = id;
    return () => window.clearInterval(id);
  }, [playing, speedIdx]);

  const rows = useMemo(
    () =>
      data.profiles.hours.map((h) => ({
        h,
        soc: (traj.soc[h] ?? 0) * 100,
        lpSoc: (lp.soc[h] ?? 0) * 100,
        bat: traj.batteryPower[h] ?? 0,
        grid: traj.gridPower[h] ?? 0,
        profit: traj.cumulativeProfit?.[h] ?? 0,
        lpProfit: lp.cumulativeProfit?.[h] ?? null,
        solar: solar[h],
        load: load[h],
        price: price[h],
      })),
    [traj, lp, solar, load, price, data.profiles.hours]
  );

  const now = rows[hour];
  const band = touAt(hour);
  const bat = now.bat;
  const mode =
    Math.abs(bat) < 0.05 ? "idle" : bat > 0 ? "charging" : "discharging";

  const replay = () => {
    setHour(0);
    setPlaying(true);
  };

  const togglePlay = () => {
    if (hour >= 23) replay();
    else setPlaying((p) => !p);
  };

  return (
    <section className="section" id="replay">
      <div className="shell">
        <div className="sec-head">
          <span className="sec-index">01 / REPLAY</span>
          <h2 className="sec-title">One day, hour by hour</h2>
          <p className="sec-sub">
            Press play and watch the policy work. The battery buys cheap energy
            overnight, soaks up the midday solar surplus, then sells it back
            into the on-peak window when grid power costs{" "}
            <span className="mono">${Math.max(...price).toFixed(2)}</span>/kWh.
            Switch agents to see how differently each one reasons about the
            same day.
          </p>
        </div>

        {/* ---- controls ---- */}
        <div className="rp-bar">
          <div className="rp-agents">
            {available.map((a) => (
              <button
                key={a}
                className={`rp-chip${a === agent ? " is-on" : ""}`}
                style={
                  a === agent
                    ? ({ ["--chip" as string]: agentColor(a) } as React.CSSProperties)
                    : undefined
                }
                onClick={() => setAgent(a)}
              >
                <i style={{ background: agentColor(a) }} />
                {a}
              </button>
            ))}
          </div>

          <div className="rp-transport">
            <button
              className="rp-play"
              onClick={togglePlay}
              aria-label={playing ? "Pause" : "Play"}
            >
              {playing ? (
                <svg width="13" height="14" viewBox="0 0 13 14" fill="currentColor">
                  <rect x="1" y="1" width="3.6" height="12" rx="1" />
                  <rect x="8.4" y="1" width="3.6" height="12" rx="1" />
                </svg>
              ) : (
                <svg width="13" height="14" viewBox="0 0 13 14" fill="currentColor">
                  <path d="M2 1.6a1 1 0 0 1 1.5-.87l7.2 4.4a1 1 0 0 1 0 1.74l-7.2 4.4A1 1 0 0 1 2 10.4V1.6Z" />
                </svg>
              )}
            </button>

            <button className="rp-reset" onClick={() => { setHour(0); setPlaying(false); }}>
              Reset
            </button>

            <div className="rp-speeds">
              {SPEEDS.map((s, i) => (
                <button
                  key={s.label}
                  className={`rp-speed${i === speedIdx ? " is-on" : ""}`}
                  onClick={() => setSpeedIdx(i)}
                >
                  {s.label}
                </button>
              ))}
            </div>

            <label className="rp-ghost">
              <input
                type="checkbox"
                checked={ghost}
                onChange={(e) => setGhost(e.target.checked)}
              />
              <span>LP ghost</span>
            </label>
          </div>
        </div>

        {/* ---- scrubber ---- */}
        <div className="rp-scrub">
          <input
            type="range"
            min={0}
            max={23}
            step={1}
            value={hour}
            onChange={(e) => {
              setHour(Number(e.target.value));
              setPlaying(false);
            }}
            style={{ ["--fill" as string]: `${(hour / 23) * 100}%` }}
          />
          <div className="rp-ticks" aria-hidden="true">
            {[0, 6, 12, 18, 23].map((t) => (
              <span key={t} className="mono" style={{ left: `${(t / 23) * 100}%` }}>
                {String(t).padStart(2, "0")}
              </span>
            ))}
          </div>
        </div>

        {/* ---- live readout ---- */}
        <div className="rp-stage">
          <div className="rp-dials">
            <div className="rp-clock">
              <span className="rp-clock-time num">
                {String(hour).padStart(2, "0")}
                <span className="rp-clock-colon">:</span>00
              </span>
              <span
                className="rp-clock-band"
                style={{ color: band.color, borderColor: band.color }}
              >
                {band.label} · ${band.price.toFixed(2)}
              </span>
            </div>

            <SocDial soc={now.soc} min={cfg.minSocRatio * 100} mode={mode} />

            <div className="rp-mode" data-mode={mode}>
              <span className="rp-mode-label">Battery</span>
              <span className="rp-mode-value num">
                {bat > 0.05 ? "+" : ""}
                {bat.toFixed(2)} <small>kW</small>
              </span>
              <span className="rp-mode-tag">{mode}</span>
            </div>

            <div className="rp-meters">
              <Meter
                label="Solar"
                value={now.solar}
                max={Math.max(...solar) * 1.08}
                color="var(--solar)"
                unit="kW"
              />
              <Meter
                label="Load"
                value={now.load}
                max={Math.max(...load) * 1.08}
                color="var(--load)"
                unit="kW"
              />
              <Meter
                label={now.grid >= 0 ? "Grid import" : "Grid export"}
                value={Math.abs(now.grid)}
                max={cfg.peakThresholdKw}
                color={now.grid >= 0 ? "var(--grid-import)" : "var(--grid-export)"}
                unit="kW"
                danger={Math.abs(now.grid) > cfg.peakThresholdKw}
              />
            </div>

            <div className="rp-pnl">
              <span className="rp-pnl-label">Cumulative P&L</span>
              <span
                className={`rp-pnl-value num ${
                  now.profit >= 0 ? "pos" : "neg"
                }`}
              >
                {now.profit < 0 ? "−" : ""}${Math.abs(now.profit).toFixed(2)}
              </span>
              {ghost && now.lpProfit !== null && (
                <span className="rp-pnl-ghost mono">
                  LP {now.lpProfit < 0 ? "−" : ""}$
                  {Math.abs(now.lpProfit).toFixed(2)}
                </span>
              )}
            </div>
          </div>

          <div className="rp-charts">
            <RpChart title="State of charge" unit="%">
              <AreaChart data={rows} margin={{ top: 6, right: 8, bottom: 0, left: -14 }}>
                <defs>
                  <linearGradient id="socFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={agentColor(agent)} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={agentColor(agent)} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid {...GRID} />
                <TouBands />
                <XAxis dataKey="h" {...AXIS} tickLine={false} interval={5} />
                <YAxis domain={[0, 100]} {...AXIS} tickLine={false} width={42} />
                <ReferenceLine
                  y={cfg.minSocRatio * 100}
                  stroke="#e8576b"
                  strokeDasharray="3 3"
                  strokeOpacity={0.5}
                />
                {ghost && (
                  <Line
                    dataKey="lpSoc"
                    stroke="#d8dee6"
                    strokeWidth={1.3}
                    strokeDasharray="4 3"
                    strokeOpacity={0.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                )}
                <Area
                  dataKey="soc"
                  stroke={agentColor(agent)}
                  strokeWidth={2}
                  fill="url(#socFill)"
                  dot={false}
                  isAnimationActive={false}
                />
                <ReferenceLine x={hour} stroke="#e4ebf2" strokeOpacity={0.3} />
                <ReferenceDot
                  x={hour}
                  y={now.soc}
                  r={4}
                  fill={agentColor(agent)}
                  stroke="#0b0f14"
                  strokeWidth={2}
                />
              </AreaChart>
            </RpChart>

            <RpChart title="Battery dispatch" unit="kW">
              <AreaChart data={rows} margin={{ top: 6, right: 8, bottom: 0, left: -14 }}>
                <CartesianGrid {...GRID} />
                <TouBands />
                <XAxis dataKey="h" {...AXIS} tickLine={false} interval={5} />
                <YAxis
                  domain={[-cfg.maxPowerKw, cfg.maxPowerKw]}
                  {...AXIS}
                  tickLine={false}
                  width={42}
                />
                <ReferenceLine y={0} stroke="#2f4055" />
                <Area
                  dataKey="bat"
                  stroke={agentColor(agent)}
                  strokeWidth={1.8}
                  fill={agentColor(agent)}
                  fillOpacity={0.14}
                  dot={false}
                  isAnimationActive={false}
                />
                <ReferenceLine x={hour} stroke="#e4ebf2" strokeOpacity={0.3} />
                <ReferenceDot
                  x={hour}
                  y={now.bat}
                  r={4}
                  fill={agentColor(agent)}
                  stroke="#0b0f14"
                  strokeWidth={2}
                />
              </AreaChart>
            </RpChart>

            <RpChart title="Cumulative profit" unit="$">
              <AreaChart data={rows} margin={{ top: 6, right: 8, bottom: 0, left: -14 }}>
                <CartesianGrid {...GRID} />
                <TouBands />
                <XAxis dataKey="h" {...AXIS} tickLine={false} interval={5} />
                <YAxis {...AXIS} tickLine={false} width={42} />
                <ReferenceLine y={0} stroke="#2f4055" />
                {ghost && (
                  <Line
                    dataKey="lpProfit"
                    stroke="#d8dee6"
                    strokeWidth={1.3}
                    strokeDasharray="4 3"
                    strokeOpacity={0.5}
                    dot={false}
                    isAnimationActive={false}
                    connectNulls
                  />
                )}
                <Area
                  dataKey="profit"
                  stroke={agentColor(agent)}
                  strokeWidth={2}
                  fill={agentColor(agent)}
                  fillOpacity={0.1}
                  dot={false}
                  isAnimationActive={false}
                />
                <ReferenceLine x={hour} stroke="#e4ebf2" strokeOpacity={0.3} />
                <ReferenceDot
                  x={hour}
                  y={now.profit}
                  r={4}
                  fill={agentColor(agent)}
                  stroke="#0b0f14"
                  strokeWidth={2}
                />
              </AreaChart>
            </RpChart>

            <div className="rp-chart-foot">
              <TouKey />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function RpChart({
  title,
  unit,
  children,
}: {
  title: string;
  unit: string;
  children: React.ReactElement;
}) {
  return (
    <div className="rp-chart">
      <div className="rp-chart-head">
        <span className="rp-chart-title">{title}</span>
        <span className="rp-chart-unit mono">{unit}</span>
      </div>
      <div className="rp-chart-body">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function SocDial({
  soc,
  min,
  mode,
}: {
  soc: number;
  min: number;
  mode: string;
}) {
  const R = 46;
  const C = 2 * Math.PI * R;
  const sweep = 0.75; // three-quarter ring
  const arc = C * sweep;
  const fill = (soc / 100) * arc;

  const color =
    mode === "charging"
      ? "var(--charge)"
      : mode === "discharging"
      ? "var(--discharge)"
      : "var(--text-2)";

  return (
    <div className="rp-dial">
      <svg viewBox="0 0 110 110" className="rp-dial-svg">
        <circle
          cx="55"
          cy="55"
          r={R}
          fill="none"
          stroke="var(--surface-3)"
          strokeWidth="8"
          strokeDasharray={`${arc} ${C}`}
          strokeLinecap="round"
          transform="rotate(135 55 55)"
        />
        <circle
          cx="55"
          cy="55"
          r={R}
          fill="none"
          stroke="#e8576b"
          strokeOpacity="0.45"
          strokeWidth="8"
          strokeDasharray={`${(min / 100) * arc} ${C}`}
          strokeLinecap="round"
          transform="rotate(135 55 55)"
        />
        <circle
          cx="55"
          cy="55"
          r={R}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={`${fill} ${C}`}
          strokeLinecap="round"
          transform="rotate(135 55 55)"
          style={{ transition: "stroke-dasharray 0.3s var(--ease), stroke 0.3s var(--ease)" }}
        />
      </svg>
      <div className="rp-dial-text">
        <span className="rp-dial-value num">{soc.toFixed(0)}</span>
        <span className="rp-dial-unit">% SOC</span>
      </div>
    </div>
  );
}

function Meter({
  label,
  value,
  max,
  color,
  unit,
  danger,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
  unit: string;
  danger?: boolean;
}) {
  const w = Math.min(100, (value / max) * 100);
  return (
    <div className="rp-meter">
      <div className="rp-meter-top">
        <span className="rp-meter-label">{label}</span>
        <span className={`rp-meter-value num${danger ? " neg" : ""}`}>
          {value.toFixed(2)} <small>{unit}</small>
        </span>
      </div>
      <div className="rp-meter-track">
        <div
          className="rp-meter-fill"
          style={{ width: `${w}%`, background: danger ? "var(--price-peak)" : color }}
        />
      </div>
    </div>
  );
}
