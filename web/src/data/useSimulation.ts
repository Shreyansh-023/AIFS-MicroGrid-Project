import { useEffect, useState } from "react";
import type { GridIndex, SimulationData, Source } from "./types";

type State =
  | { status: "loading" }
  | { status: "ready"; data: SimulationData; source: Source; switching: boolean }
  | { status: "error"; message: string };

const urlFor = (source: Source) =>
  `${import.meta.env.BASE_URL}data/${
    source === "base" ? "simulation.json" : `grids/${source}.json`
  }`;

const cache = new Map<Source, SimulationData>();

/**
 * Loads the dataset for one source. While a new grid is loading, the previous
 * data stays on screen (status "ready", switching = true) so the page does not
 * flash back to the boot screen.
 */
export function useSimulation(source: Source): State {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    let alive = true;
    const hit = cache.get(source);
    if (hit) {
      setState({ status: "ready", data: hit, source, switching: false });
      return;
    }
    setState((s) => (s.status === "ready" ? { ...s, switching: true } : s));

    fetch(urlFor(source))
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((data: SimulationData) => {
        cache.set(source, data);
        if (alive) setState({ status: "ready", data, source, switching: false });
      })
      .catch((e: Error) => {
        if (alive) setState({ status: "error", message: e.message });
      });

    return () => {
      alive = false;
    };
  }, [source]);

  return state;
}

/** Cross-grid index; null while loading or when the multi-grid export is absent. */
export function useGridIndex(): GridIndex | null {
  const [index, setIndex] = useState<GridIndex | null>(null);
  useEffect(() => {
    let alive = true;
    fetch(`${import.meta.env.BASE_URL}data/grids/index.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: GridIndex | null) => alive && setIndex(d))
      .catch(() => alive && setIndex(null));
    return () => {
      alive = false;
    };
  }, []);
  return index;
}

export const money = (v: number, dp = 2) =>
  `${v < 0 ? "−" : ""}$${Math.abs(v).toFixed(dp)}`;

export const signedMoney = (v: number, dp = 2) =>
  `${v < 0 ? "−" : "+"}$${Math.abs(v).toFixed(dp)}`;

export const kw = (v: number, dp = 2) => `${v.toFixed(dp)} kW`;

export const pct = (v: number, dp = 0) => `${v.toFixed(dp)}%`;

export const hourLabel = (h: number) =>
  `${String(h).padStart(2, "0")}:00`;

/** Decimal places that keep a dollar figure readable at any grid scale. */
export const moneyDp = (v: number) => (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2);
