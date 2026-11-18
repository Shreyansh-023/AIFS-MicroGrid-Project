# multigrid/lp.py
# Generalised battery-scheduling LP (perfect foresight) and shrinking-horizon
# MPC for every grid archetype. The objective is exactly the env's daily cost:
#
#   sum_t  p_imp*Gp - p_exp*Gn + voll*U + peak_rate*Pk + deg*(eta_c*Pc + Pd/eta_d)
#   + demand_rate * (M - M0) + terminal_rate * |E_T - E_target|
#
# subject to (per hour t, from the current hour h0 to 23)
#   Gp - Gn - Pc + Pd + U - C = load - pv             power balance
#   E_t = E0 + sum(eta_c*Pc - Pd/eta_d) in [Emin, Emax]
#   |(Pc-Pd)_t - (Pc-Pd)_{t-1}| <= ramp,  (Pc-Pd)_{h0-1} = prev battery power
#   Pk >= Gp - P_peak,  M >= Gp,  M >= M0 (running max import)
#   0<=Pc,Pd<=Pbat, 0<=Gp<=import_cap, 0<=Gn<=export_cap, 0<=U<=load, 0<=C<=pv
#
# Plans are executed in the batched env (multigrid/evaluate.py), so LP, MPC and
# RL are scored by the same cost code.

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from multigrid.grids import finite

BIG = 1e9


def grid_params(cfg):
    return {
        "e_max": cfg.e_max, "e_min": cfg.e_max * cfg.e_min_ratio, "p_bat": cfg.p_bat_max,
        "eta_c": cfg.eta_charge, "eta_d": cfg.eta_discharge, "ramp": cfg.ramp_rate,
        "deg": cfg.degradation_cost_per_kwh, "target": cfg.target_final_soc * cfg.e_max,
        "term": cfg.terminal_penalty_per_kwh, "dc": cfg.demand_charge_per_kw_day,
        "p_peak": finite(cfg.p_grid_peak, BIG), "peak_rate": cfg.peak_penalty_rate,
        "imp_cap": finite(cfg.import_cap_kw, BIG), "exp_cap": finite(cfg.export_cap_kw, BIG),
        "voll": cfg.voll,
    }


def solve_schedule(prm, load, pv, p_imp, p_exp, soc0, prev_bp=0.0, run_max=0.0,
                   hard_terminal=False, ramp_first=True, ramp=True):
    """Optimal battery power (kW, + = charge) for the given (remaining) hours.

    If the problem is infeasible (the env applies SOC limits *after* the ramp
    limit, so a state like "charging at full power with a full battery" has no
    ramp-feasible continuation), the link to the previous battery power is
    dropped, then the ramp limit altogether.
    """
    H = len(load)
    soc0 = min(max(soc0, prm["e_min"] / prm["e_max"]), 1.0)
    net = np.asarray(load, float) - np.asarray(pv, float)
    # variable blocks of length H
    PC, PD, GP, GN, U, C, PK = (np.arange(H) + k * H for k in range(7))
    M, DP, DN = 7 * H, 7 * H + 1, 7 * H + 2
    nv = 7 * H + 3

    c = np.zeros(nv)
    c[GP] = p_imp
    c[GN] = -np.asarray(p_exp, float)
    c[U] = prm["voll"]
    c[PK] = prm["peak_rate"]
    c[PC] = prm["deg"] * prm["eta_c"]
    c[PD] = prm["deg"] / prm["eta_d"]
    c[M] = prm["dc"]
    c[DP] = c[DN] = 0.0 if hard_terminal else prm["term"]

    rows, cols, vals, b_eq = [], [], [], []
    r = 0
    for t in range(H):                                   # power balance
        for col, v in ((GP[t], 1), (GN[t], -1), (PC[t], -1), (PD[t], 1), (U[t], 1), (C[t], -1)):
            rows.append(r); cols.append(col); vals.append(v)
        b_eq.append(net[t]); r += 1
    E0 = soc0 * prm["e_max"]
    # terminal: E_T - E_target = DP - DN
    for t in range(H):
        rows += [r, r]; cols += [PC[t], PD[t]]; vals += [prm["eta_c"], -1 / prm["eta_d"]]
    rows += [r, r]; cols += [DP, DN]; vals += [-1, 1]
    b_eq.append(prm["target"] - E0); r += 1
    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))

    rows, cols, vals, b_ub = [], [], [], []
    r = 0
    for t in range(H):                                   # SOC window (cumulative)
        for s in range(t + 1):
            rows += [r, r, r + 1, r + 1]; cols += [PC[s], PD[s], PC[s], PD[s]]
            vals += [prm["eta_c"], -1 / prm["eta_d"], -prm["eta_c"], 1 / prm["eta_d"]]
        b_ub += [prm["e_max"] - E0, E0 - prm["e_min"]]; r += 2
    for t in range(H if ramp else 0):                    # ramp
        if t == 0 and not ramp_first:
            continue
        for sgn in (1, -1):
            rows += [r, r]; cols += [PC[t], PD[t]]; vals += [sgn, -sgn]
            if t > 0:
                rows += [r, r]; cols += [PC[t - 1], PD[t - 1]]; vals += [-sgn, sgn]
                b_ub.append(prm["ramp"])
            else:
                b_ub.append(prm["ramp"] + sgn * prev_bp)
            r += 1
    for t in range(H):                                   # peak excess, running max
        rows += [r, r]; cols += [GP[t], PK[t]]; vals += [1, -1]; b_ub.append(prm["p_peak"]); r += 1
        if prm["dc"] > 0:
            rows += [r, r]; cols += [GP[t], M]; vals += [1, -1]; b_ub.append(0.0); r += 1
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))

    lo, hi = np.zeros(nv), np.full(nv, np.inf)
    hi[PC] = hi[PD] = prm["p_bat"]
    hi[GP] = prm["imp_cap"]; hi[GN] = prm["exp_cap"]
    hi[U] = np.maximum(np.asarray(load, float), 0.0)
    hi[C] = np.maximum(np.asarray(pv, float), 0.0)
    lo[M] = run_max
    if prm["dc"] == 0:
        hi[M] = run_max
    if hard_terminal:
        hi[DP] = hi[DN] = 0.0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=list(zip(lo, hi)), method="highs")
    if not res.success:
        args = (prm, load, pv, p_imp, p_exp, soc0, prev_bp, run_max)
        if hard_terminal:   # fall back to the soft terminal penalty
            return solve_schedule(*args, hard_terminal=False, ramp_first=ramp_first, ramp=ramp)
        if ramp_first:
            return solve_schedule(*args, ramp_first=False, ramp=ramp)
        if ramp:
            return solve_schedule(*args, ramp_first=False, ramp=False)
        raise RuntimeError("LP failed: %s" % res.message)
    x = res.x
    const = -prm["dc"] * run_max
    return {"battery_power": x[PC] - x[PD], "objective": res.fun + const,
            "grid": x[GP] - x[GN], "unserved": x[U], "curtailed": x[C]}


# ---- helpers for multiprocessing pools (top-level so they pickle on Windows)

def _plan_job(args):
    prm, load, pv, pi, pe, soc0, prev_bp, run_max = args
    return solve_schedule(prm, load, pv, pi, pe, soc0, prev_bp, run_max)["battery_power"]


def plan_many(pool, jobs):
    """Solve a list of jobs (tuples for _plan_job); returns list of plans."""
    if pool is None:
        return [_plan_job(j) for j in jobs]
    return pool.map(_plan_job, jobs, chunksize=max(1, len(jobs) // 48))
