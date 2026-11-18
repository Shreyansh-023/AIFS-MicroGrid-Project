import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SimulationData } from "../data/types";
import { touAt, useBands } from "../data/agents";
import { AXIS, GRID, Legend, Tip, TouBands, TouKey } from "../components/ChartKit";
import "./System.css";

export default function System({ data }: { data: SimulationData }) {
  const cfg = data.meta.config;
  const { hours, solar, load, netLoad, price } = data.profiles;
  const bands = useBands();
  const g = data.meta.grid;
  const obsDim = g?.obsDim ?? 131;
  const priceMax = Math.max(0.34, Math.max(...price) * 1.15);
  const loadLabel = g?.loadLabel ?? "Household load";

  // the grid-specific cost term (peak cap, demand charge or import cap)
  const limitTerm =
    cfg.peakThresholdKw !== null
      ? {
          name: "Peak penalty",
          formula: `$${cfg.peakPenaltyRate.toFixed(2)} × max(0, P_grid − ${cfg.peakThresholdKw})`,
          note: "demand charge above the cap",
        }
      : g?.demandChargePerKwDay
      ? {
          name: "Demand charge",
          formula: `$${g.demandChargePerKwDay.toFixed(2)} × max_t P_import`,
          note: "billed on the day's peak import, in kW",
        }
      : g?.importCapKw != null
      ? {
          name: "Unserved energy",
          formula: `$${g.voll.toFixed(2)} × max(0, P_net − ${g.importCapKw})`,
          note: "load above the import cap is shed",
        }
      : { name: "Peak penalty", formula: "none on this tariff", note: "no demand or peak charge" };

  const exportNote = !g
    ? "exports earn 40% of retail"
    : g.exportCapKw === 0
    ? "export is not allowed"
    : `exports earn ${(g.exportRatio * 100).toFixed(0)}% of retail${
        g.exportCapKw != null ? `, capped at ${g.exportCapKw} kW` : ""
      }`;

  const rows = hours.map((h) => ({
    h,
    solar: solar[h],
    load: load[h],
    net: netLoad[h],
    price: price[h],
    band: touAt(h, bands).color,
  }));

  return (
    <section className="section" id="system">
      <div className="shell">
        <div className="sec-head">
          <span className="sec-index">04 / THE SYSTEM</span>
          <h2 className="sec-title">What the agent actually sees</h2>
          <p className="sec-sub">
            A {obsDim}-dimensional observation, one scalar action, and a cost
            function with competing terms.{" "}
            {g
              ? `${g.id} runs on ${g.tariff.toLowerCase()} with a ${g.pvKwp} kWp array. The chart shows the replayed day, ${g.replayDate}.`
              : "The whole problem is that solar peaks at noon while electricity is most expensive at six."}
          </p>
        </div>

        <div className="sys-grid">
          {/* ---- the duck ---- */}
          <div className="panel sys-wide">
            <div className="panel-head">
              <span className="panel-title">
                Generation, demand, and the price that rewards neither
              </span>
              <span className="panel-note">24 h · kW and $/kWh</span>
            </div>
            <div className="panel-body">
              <div className="sys-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={rows} margin={{ top: 6, right: 6, bottom: 0, left: -12 }}>
                    <defs>
                      <linearGradient id="solarF" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#f5c242" stopOpacity={0.28} />
                        <stop offset="100%" stopColor="#f5c242" stopOpacity={0.02} />
                      </linearGradient>
                      <linearGradient id="loadF" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#ff7a66" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#ff7a66" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid {...GRID} />
                    <TouBands opacity={0.07} />
                    <XAxis dataKey="h" {...AXIS} tickLine={false} />
                    <YAxis
                      yAxisId="kw"
                      {...AXIS}
                      tickLine={false}
                      width={44}
                      tickFormatter={(v: number) => (v >= 10 ? v.toFixed(0) : v.toFixed(1))}
                    />
                    <YAxis
                      yAxisId="pr"
                      orientation="right"
                      {...AXIS}
                      tickLine={false}
                      width={48}
                      domain={[0, Number(priceMax.toFixed(2))]}
                      tickFormatter={(v: number) => `$${v.toFixed(2)}`}
                    />
                    <Tooltip
                      content={({ active, payload, label }) =>
                        active && payload?.length ? (
                          <Tip
                            title={`${String(label).padStart(2, "0")}:00`}
                            rows={[
                              {
                                label: "Solar",
                                value: `${payload[0].payload.solar.toFixed(2)} kW`,
                                color: "#f5c242",
                              },
                              {
                                label: "Load",
                                value: `${payload[0].payload.load.toFixed(2)} kW`,
                                color: "#ff7a66",
                              },
                              {
                                label: "Net",
                                value: `${payload[0].payload.net.toFixed(2)} kW`,
                                dim: true,
                              },
                              {
                                label: "Price",
                                value: `$${payload[0].payload.price.toFixed(2)}/kWh`,
                                dim: true,
                              },
                            ]}
                            footer={touAt(Number(label), bands).label}
                          />
                        ) : null
                      }
                    />
                    <Area
                      yAxisId="kw"
                      dataKey="solar"
                      stroke="#f5c242"
                      strokeWidth={2}
                      fill="url(#solarF)"
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Area
                      yAxisId="kw"
                      dataKey="load"
                      stroke="#ff7a66"
                      strokeWidth={2}
                      fill="url(#loadF)"
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      yAxisId="pr"
                      dataKey="price"
                      stroke="#9eb0c2"
                      strokeWidth={1.5}
                      strokeDasharray="3 3"
                      dot={false}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <div className="sys-chart-legend">
                <Legend
                  items={[
                    { label: "Solar generation", color: "#f5c242" },
                    { label: loadLabel, color: "#ff7a66" },
                    { label: g ? "Import price" : "TOU price", color: "#9eb0c2", dashed: true },
                  ]}
                />
                <TouKey />
              </div>
            </div>
          </div>

          {/* ---- observation anatomy ---- */}
          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">Observation vector</span>
              <span className="panel-note">{obsDim} dims, normalised</span>
            </div>
            <div className="panel-body">
              <div className="obs-stack">
                <ObsBlock
                  n={7}
                  total={obsDim}
                  label="Current state"
                  color="#4da3ff"
                  items={[
                    "SOC (centred)",
                    "net power",
                    "price",
                    "sin/cos hour",
                    "previous action",
                    "last forecast error",
                  ]}
                />
                <ObsBlock
                  n={4}
                  total={obsDim}
                  label="Recent history"
                  color="#35d6a4"
                  items={["rolling net-power buffer, 4 steps"]}
                />
                <ObsBlock
                  n={120}
                  total={obsDim}
                  label="Forecast window"
                  color="#c48bff"
                  items={[
                    `${cfg.forecastHorizonH} h × 5 features`,
                    "net power, price, solar, sin/cos hour",
                  ]}
                />
                {g && (
                  <ObsBlock
                    n={obsDim - 131}
                    total={obsDim}
                    label="Grid descriptor + context"
                    color="#f0a23c"
                    items={[
                      "storage hours, ramp, PV and load vs inverter",
                      "export value, demand charge, import/export headroom",
                      "day's peak net load, peak price, running peak import",
                    ]}
                  />
                )}
              </div>
              <p className="sys-read">
                {g
                  ? "The descriptor is what lets one generalist policy tell the six grids apart. "
                  : ""}
                Most of the input is look-ahead. That is the whole bet: if
                the forecast is good, plan with it; if it is not, the agent has
                to learn to distrust it.
              </p>
            </div>
          </div>

          {/* ---- reward anatomy ---- */}
          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">Cost function</span>
              <span className="panel-note">reward = −cost</span>
            </div>
            <div className="panel-body">
              <div className="rw-list">
                <RwTerm
                  sign="−"
                  name="Energy cost"
                  formula="P_grid × price"
                  note={exportNote}
                  color="#4da3ff"
                />
                <RwTerm
                  sign="−"
                  name={limitTerm.name}
                  formula={limitTerm.formula}
                  note={limitTerm.note}
                  color="#e8576b"
                />
                <RwTerm
                  sign="−"
                  name="Degradation"
                  formula={`$${cfg.degradationCostPerKwh.toFixed(2)} × |throughput|`}
                  note="pack wear per kWh cycled"
                  color="#f0a23c"
                />
                <RwTerm
                  sign="−"
                  name="Terminal SOC"
                  formula={`|SOC_T − 0.5| × ${g ? g.terminalPenaltyPerUnitSoc.toFixed(0) : 50}`}
                  note="applied once, at hour 24"
                  color="#9eb0c2"
                />
              </div>
              <p className="sys-read">
                The terminal term is what stops an agent from simply draining
                the battery for a one-day payday. It has to hand the pack back
                half full.
              </p>
            </div>
          </div>

          {/* ---- hardware spec ---- */}
          <div className="panel sys-wide">
            <div className="panel-head">
              <span className="panel-title">Plant configuration</span>
              <span className="panel-note">{g ? `${g.id} · ${g.name}` : "Tesla Powerwall class"}</span>
            </div>
            <div className="spec-grid">
              <Spec label="Capacity" value={`${cfg.batteryCapacityKwh}`} unit="kWh" />
              <Spec label="Max power" value={`±${cfg.maxPowerKw}`} unit="kW" />
              <Spec
                label="Round-trip η"
                value={`${(cfg.efficiency * cfg.efficiency * 100).toFixed(0)}`}
                unit="%"
              />
              <Spec label="Ramp limit" value={`${cfg.rampRateKwPerH}`} unit="kW/h" />
              <Spec
                label={cfg.peakThresholdKw !== null ? "Grid cap" : "Import cap"}
                value={`${cfg.peakThresholdKw ?? g?.importCapKw ?? "none"}`}
                unit={(cfg.peakThresholdKw ?? g?.importCapKw) != null ? "kW" : ""}
              />
              {g && <Spec label="Solar PV" value={`${g.pvKwp}`} unit="kWp" />}
              <Spec
                label="SOC floor"
                value={`${(cfg.minSocRatio * 100).toFixed(0)}`}
                unit="%"
              />
              <Spec label="Horizon" value={`${cfg.forecastHorizonH}`} unit="h" />
              <Spec label="Decision step" value="1" unit="hour" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function ObsBlock({
  n,
  total,
  label,
  color,
  items,
}: {
  n: number;
  total: number;
  label: string;
  color: string;
  items: string[];
}) {
  const share = (n / total) * 100;
  return (
    <div className="obs-block">
      <div className="obs-block-head">
        <span className="obs-block-label">
          <i style={{ background: color }} />
          {label}
        </span>
        <span className="obs-block-n mono">
          {n} <span className="dim">· {share.toFixed(0)}%</span>
        </span>
      </div>
      <div className="obs-block-bar">
        <div
          className="obs-block-fill"
          style={{ width: `${share}%`, background: color }}
        />
      </div>
      <ul className="obs-block-items">
        {items.map((it) => (
          <li key={it}>{it}</li>
        ))}
      </ul>
    </div>
  );
}

function RwTerm({
  sign,
  name,
  formula,
  note,
  color,
}: {
  sign: string;
  name: string;
  formula: string;
  note: string;
  color: string;
}) {
  return (
    <div className="rw-term">
      <span className="rw-sign mono" style={{ color }}>
        {sign}
      </span>
      <span className="rw-body">
        <span className="rw-name">{name}</span>
        <span className="rw-formula mono">{formula}</span>
        <span className="rw-note">{note}</span>
      </span>
    </div>
  );
}

function Spec({
  label,
  value,
  unit,
}: {
  label: string;
  value: string;
  unit: string;
}) {
  return (
    <div className="spec">
      <span className="spec-label">{label}</span>
      <span className="spec-value num">
        {value}
        <small>{unit}</small>
      </span>
    </div>
  );
}
