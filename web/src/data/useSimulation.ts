import { useEffect, useState } from "react";
import type { SimulationData } from "./types";

type State =
  | { status: "loading" }
  | { status: "ready"; data: SimulationData }
  | { status: "error"; message: string };

export function useSimulation(): State {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    let alive = true;

    fetch(`${import.meta.env.BASE_URL}data/simulation.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((data: SimulationData) => {
        if (alive) setState({ status: "ready", data });
      })
      .catch((e: Error) => {
        if (alive) setState({ status: "error", message: e.message });
      });

    return () => {
      alive = false;
    };
  }, []);

  return state;
}

export const money = (v: number, dp = 2) =>
  `${v < 0 ? "−" : ""}$${Math.abs(v).toFixed(dp)}`;

export const signedMoney = (v: number, dp = 2) =>
  `${v < 0 ? "−" : "+"}$${Math.abs(v).toFixed(dp)}`;

export const kw = (v: number, dp = 2) => `${v.toFixed(dp)} kW`;

export const pct = (v: number, dp = 0) => `${v.toFixed(dp)}%`;

export const hourLabel = (h: number) =>
  `${String(h).padStart(2, "0")}:00`;
