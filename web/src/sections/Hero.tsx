import type { SimulationData } from "../data/types";
import { touAt } from "../data/agents";
import "./Hero.css";

interface Props {
  data: SimulationData;
}

export default function Hero({ data }: Props) {
  const { summary, meta, robustness } = data;

  const lp = summary.find((r) => r.Agent === "LP");
  const learned = summary.filter((r) =>
    meta.agentsTrained.includes(String(r.Agent))
  );

  const bestLearned = learned.length
    ? learned.reduce((a, b) =>
        Number(a["Mean Profit ($)"]) > Number(b["Mean Profit ($)"]) ? a : b
      )
    : null;

  const mostRobust = robustness?.summary?.[0];

  const peakPrice = Math.max(...data.profiles.price);
  const offPrice = Math.min(...data.profiles.price);
  const spread = (peakPrice / offPrice).toFixed(1);

  return (
    <header className="hero">
      <div className="hero-grain" aria-hidden="true" />

      <div className="shell hero-inner">
        <div className="hero-kicker">
          <span className="hero-kicker-dot" />
          <span className="mono">
            Reinforcement learning · battery dispatch · thesis build
          </span>
        </div>

        <h1 className="hero-title">
          A battery doesn't know
          <br />
          what tomorrow costs.
        </h1>

        <p className="hero-lede">
          Linear programming solves the perfect day — if you already know the
          day. This project asks what happens when you don't: three RL agents
          dispatch a {meta.config.batteryCapacityKwh} kWh home battery against
          an LP that was handed the answer key, then both get their forecasts
          corrupted.
        </p>

        <div className="hero-stats">
          <HeroStat
            label="LP benchmark"
            value={lp ? fmt(Number(lp["Mean Profit ($)"])) : "—"}
            note="perfect foresight"
          />
          <HeroStat
            label={bestLearned ? `Best learned · ${bestLearned.Agent}` : "Best learned"}
            value={bestLearned ? fmt(Number(bestLearned["Mean Profit ($)"])) : "—"}
            note={
              bestLearned
                ? `${Number(bestLearned["Gap vs LP (%)"]).toFixed(1)}% gap`
                : "not yet trained"
            }
            accent
          />
          <HeroStat
            label="Most robust"
            value={mostRobust ? String(mostRobust.Agent) : "—"}
            note={
              mostRobust
                ? `${Math.abs(Number(mostRobust["Degradation (%)"])).toFixed(1)}% decay @30% noise`
                : "sweep pending"
            }
          />
          <HeroStat
            label="Price spread"
            value={`${spread}×`}
            note={`$${offPrice.toFixed(2)} → $${peakPrice.toFixed(2)}/kWh`}
          />
        </div>

        <Schematic data={data} />

        <a className="hero-scroll" href="#replay">
          <span>Replay a day, hour by hour</span>
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path
              d="M6.5 1v11M2 8l4.5 4.5L11 8"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </a>
      </div>
    </header>
  );
}

function fmt(v: number) {
  return `${v < 0 ? "−" : ""}$${Math.abs(v).toFixed(2)}`;
}

function HeroStat({
  label,
  value,
  note,
  accent,
}: {
  label: string;
  value: string;
  note: string;
  accent?: boolean;
}) {
  return (
    <div className={`hstat${accent ? " hstat--accent" : ""}`}>
      <span className="hstat-label">{label}</span>
      <span className="hstat-value num">{value}</span>
      <span className="hstat-note mono">{note}</span>
    </div>
  );
}

/**
 * Static system schematic. Flow widths/opacities are driven by the real
 * midday hour of the loaded profile so it is representative, not decorative.
 */
function Schematic({ data }: { data: SimulationData }) {
  const { solar, load } = data.profiles;

  // pick the hour with the most solar surplus — the interesting moment
  let h = 12;
  let best = -Infinity;
  solar.forEach((s, i) => {
    const surplus = s - load[i];
    if (surplus > best) {
      best = surplus;
      h = i;
    }
  });

  const s = solar[h];
  const l = load[h];
  const surplus = s - l;
  const band = touAt(h);

  return (
    <div className="schem">
      <div className="schem-head">
        <span className="schem-head-label mono">
          System at {String(h).padStart(2, "0")}:00 — peak surplus
        </span>
        <span className="schem-head-band" style={{ color: band.color }}>
          <i style={{ background: band.color }} />
          {band.label} · ${band.price.toFixed(2)}/kWh
        </span>
      </div>

      <div className="schem-rail">
        <Node
          kind="solar"
          title="Solar PV"
          value={`${s.toFixed(2)} kW`}
          icon={<SunIcon />}
        />
        <Flow
          magnitude={s}
          max={6}
          color="var(--solar)"
          label="generation"
        />
        <Node
          kind="bus"
          title="DC/AC bus"
          value="node"
          icon={<BusIcon />}
        />
        <Flow
          magnitude={Math.abs(surplus)}
          max={6}
          color={surplus > 0 ? "var(--charge)" : "var(--discharge)"}
          label={surplus > 0 ? "charge" : "discharge"}
          reverse={surplus < 0}
        />
        <Node
          kind="battery"
          title="BESS"
          value={`${data.meta.config.batteryCapacityKwh} kWh`}
          icon={<BatteryIcon />}
        />
      </div>

      <div className="schem-rail schem-rail--lower">
        <Node
          kind="grid"
          title="Utility grid"
          value={`cap ${data.meta.config.peakThresholdKw} kW`}
          icon={<GridIcon />}
        />
        <Flow
          magnitude={Math.max(0, l - s)}
          max={6}
          color="var(--grid-import)"
          label="import"
        />
        <Node
          kind="load"
          title="Household load"
          value={`${l.toFixed(2)} kW`}
          icon={<HomeIcon />}
        />
      </div>

      <div className="schem-foot">
        <span className="mono">
          Agent observes 131 features · acts once per hour · ±
          {data.meta.config.maxPowerKw} kW
        </span>
      </div>
    </div>
  );
}

function Node({
  kind,
  title,
  value,
  icon,
}: {
  kind: string;
  title: string;
  value: string;
  icon: React.ReactNode;
}) {
  return (
    <div className={`snode snode--${kind}`}>
      <span className="snode-icon">{icon}</span>
      <span className="snode-text">
        <span className="snode-title">{title}</span>
        <span className="snode-value mono">{value}</span>
      </span>
    </div>
  );
}

function Flow({
  magnitude,
  max,
  color,
  label,
  reverse,
}: {
  magnitude: number;
  max: number;
  color: string;
  label: string;
  reverse?: boolean;
}) {
  const t = Math.min(1, magnitude / max);
  const thickness = 1.5 + t * 3.5;

  return (
    <div className="sflow" style={{ ["--flow-color" as string]: color }}>
      <div className="sflow-track">
        <div
          className={`sflow-beam${reverse ? " sflow-beam--rev" : ""}`}
          style={{ height: thickness, opacity: 0.25 + t * 0.6 }}
        />
      </div>
      <span className="sflow-label mono">
        {label} · {magnitude.toFixed(2)} kW
      </span>
    </div>
  );
}

/* ---------- icons ---------- */

const SunIcon = () => (
  <svg width="17" height="17" viewBox="0 0 20 20" fill="none">
    <circle cx="10" cy="10" r="3.6" stroke="currentColor" strokeWidth="1.5" />
    <path
      d="M10 1.6v2.1M10 16.3v2.1M1.6 10h2.1M16.3 10h2.1M4.1 4.1l1.5 1.5M14.4 14.4l1.5 1.5M15.9 4.1l-1.5 1.5M5.6 14.4l-1.5 1.5"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </svg>
);

const BatteryIcon = () => (
  <svg width="17" height="17" viewBox="0 0 20 20" fill="none">
    <rect
      x="2.2"
      y="5.6"
      width="13.2"
      height="8.8"
      rx="1.8"
      stroke="currentColor"
      strokeWidth="1.5"
    />
    <path
      d="M17.4 8.6v2.8"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
    <rect x="4.4" y="7.8" width="6.2" height="4.4" rx="0.7" fill="currentColor" />
  </svg>
);

const GridIcon = () => (
  <svg width="17" height="17" viewBox="0 0 20 20" fill="none">
    <path
      d="M10 2.2v15.6M4.6 5.4h10.8M3.4 9.2h13.2M2.6 13h14.8"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </svg>
);

const HomeIcon = () => (
  <svg width="17" height="17" viewBox="0 0 20 20" fill="none">
    <path
      d="M3 8.6 10 3l7 5.6V16a1.4 1.4 0 0 1-1.4 1.4H4.4A1.4 1.4 0 0 1 3 16V8.6Z"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinejoin="round"
    />
  </svg>
);

const BusIcon = () => (
  <svg width="17" height="17" viewBox="0 0 20 20" fill="none">
    <circle cx="10" cy="10" r="2.4" fill="currentColor" />
    <circle cx="10" cy="10" r="6.4" stroke="currentColor" strokeWidth="1.4" opacity="0.5" />
  </svg>
);
