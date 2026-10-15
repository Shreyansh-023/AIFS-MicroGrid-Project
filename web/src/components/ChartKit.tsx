import type { ReactNode } from "react";
import { ReferenceArea } from "recharts";
import { TOU_BANDS } from "../data/agents";
import "./ChartKit.css";

export const AXIS = {
  stroke: "#2f4055",
  tick: { fill: "#6b7f93", fontSize: 10.5 },
};

export const GRID = {
  stroke: "#1d2836",
  strokeDasharray: "2 4",
  vertical: false,
};

/**
 * Translucent TOU price bands for any hour-indexed chart.
 * Only the on-peak window is tinted — shading all four bands turns the
 * plot into an undifferentiated wash and hides the series.
 */
export function TouBands({ opacity = 0.09 }: { opacity?: number }) {
  const peak = TOU_BANDS.find((b) => b.label === "On-peak")!;
  return (
    <ReferenceArea
      x1={peak.from}
      x2={peak.to}
      fill={peak.color}
      fillOpacity={opacity}
      stroke="none"
      ifOverflow="hidden"
    />
  );
}

interface TipRow {
  label: string;
  value: string;
  color?: string;
  dim?: boolean;
}

export function Tip({
  title,
  rows,
  footer,
}: {
  title: string;
  rows: TipRow[];
  footer?: ReactNode;
}) {
  return (
    <div className="tip">
      <div className="tip-title mono">{title}</div>
      <div className="tip-rows">
        {rows.map((r, i) => (
          <div className={`tip-row${r.dim ? " tip-row--dim" : ""}`} key={i}>
            <span className="tip-key">
              {r.color && (
                <i className="tip-dot" style={{ background: r.color }} />
              )}
              {r.label}
            </span>
            <span className="tip-val num">{r.value}</span>
          </div>
        ))}
      </div>
      {footer && <div className="tip-foot">{footer}</div>}
    </div>
  );
}

export function Legend({
  items,
  compact = false,
}: {
  items: { label: string; color: string; dashed?: boolean }[];
  compact?: boolean;
}) {
  return (
    <div className={`lgnd${compact ? " lgnd--compact" : ""}`}>
      {items.map((it) => (
        <span className="lgnd-item" key={it.label}>
          <i
            className={`lgnd-swatch${it.dashed ? " lgnd-swatch--dashed" : ""}`}
            style={{
              [it.dashed ? "borderColor" : "background"]: it.color,
            } as React.CSSProperties}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}

export function TouKey() {
  const seen = new Set<string>();
  const bands = TOU_BANDS.filter((b) => {
    if (seen.has(b.label)) return false;
    seen.add(b.label);
    return true;
  });

  return (
    <div className="tou-key">
      {bands.map((b) => {
        const shaded = b.label === "On-peak";
        return (
          <span className="tou-key-item" key={b.label}>
            <i
              className={`tou-key-band${shaded ? " is-shaded" : ""}`}
              style={{ background: b.color }}
            />
            {b.label}
            <span className="tou-key-price mono">${b.price.toFixed(2)}</span>
            {shaded && <span className="tou-key-hint">shaded</span>}
          </span>
        );
      })}
    </div>
  );
}
