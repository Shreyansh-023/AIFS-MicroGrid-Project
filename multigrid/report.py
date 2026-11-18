# multigrid/report.py
# Tables, statistics and figures from results/multigrid/daily_results.csv.
#
# Key metric: "% of LP savings captured"
#     capture = (cost_NoOp - cost_method) / (cost_NoOp - cost_LP_oracle) * 100
# computed per grid on identical days (100 = perfect-foresight optimum,
# 0 = no better than leaving the battery idle). Mean daily cost in $ is also
# reported. Uncertainty: 95% CI across training seeds (t-dist); paired
# Wilcoxon signed-rank test across days vs MPC.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from multigrid.grids import GRIDS, GRID_IDS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "multigrid"
PROC = ROOT / "data" / "processed"

COLORS = {"LP (oracle)": "#2ecc71", "LP (open loop)": "#d35400", "MPC": "#16a085",
          "SAC": "#3498db", "R-SAC": "#9b59b6", "PPO": "#e74c3c", "R-PPO": "#c0392b",
          "Generalist PPO": "#e67e22", "Generalist SAC": "#2980b9",
          "Zero-shot PPO": "#f1c40f", "Zero-shot SAC": "#1abc9c",
          "NoOp": "#95a5a6", "Rule-Based": "#f39c12", "Self-Cons.": "#7f8c8d"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(path=None):
    return pd.read_csv(path or RESULTS / "daily_results.csv")


def ci95(x):
    """Spread over training seeds: 95% t-interval with >= 3 seeds, otherwise
    the sample s.d. (a t-interval with 2 seeds is 12.7 s.d. wide)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    if len(x) < 3:
        return float(x.std(ddof=1))
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def _reference(df, exp_for_ref, noise=0.0, scen=0):
    """Per (grid, day) NoOp and LP-oracle costs from the same scenario."""
    ref = df[(df.experiment == exp_for_ref) & (df.noise == noise) & (df.scenario == scen)]
    noop = ref[ref.method == "NoOp"].set_index(["grid", "day"])["cost"]
    lp = ref[ref.method == "LP (oracle)"].set_index(["grid", "day"])["cost"]
    return noop, lp


def capture_table(df, experiments=("E1", "E2", "E3"), ref_exp="E1"):
    """Mean cost and LP-capture per (grid, method), seed-level CI."""
    noop, lp = _reference(df, ref_exp)
    d = df[df.experiment.isin(experiments) & (df.noise == 0)].copy()
    d = d[d.experiment != "E3T"]
    key = list(zip(d.grid, d.day))
    d["noop"] = noop.reindex(key).values
    d["lp"] = lp.reindex(key).values
    rows = []
    for (g, m), x in d.groupby(["grid", "method"]):
        per_seed = x.groupby("seed").agg(cost=("cost", "mean"), noop=("noop", "mean"), lp=("lp", "mean"))
        cap = (per_seed.noop - per_seed.cost) / (per_seed.noop - per_seed.lp) * 100
        rows.append({"grid": g, "method": m, "n_seeds": len(per_seed),
                     "cost": per_seed.cost.mean(), "cost_ci": ci95(per_seed.cost),
                     "capture": cap.mean(), "capture_ci": ci95(cap),
                     "unserved_kwh": x.unserved_kwh.mean(),
                     "throughput": x.throughput.mean()})
    return pd.DataFrame(rows)


def wilcoxon_vs(df, ref_method="MPC", exp="E1", noise=0.0):
    out = []
    d = df[(df.experiment == exp) & (df.noise == noise)]
    for g in sorted(d.grid.unique()):
        dg = d[d.grid == g]
        ref = dg[dg.method == ref_method].groupby("day").cost.mean()
        for m in sorted(dg.method.unique()):
            if m == ref_method:
                continue
            x = dg[dg.method == m].groupby("day").cost.mean().reindex(ref.index)
            diff = (x - ref).dropna()
            if len(diff) < 5 or np.allclose(diff, 0):
                continue
            p = stats.wilcoxon(diff).pvalue
            out.append({"grid": g, "method": m, "vs": ref_method, "mean_diff": diff.mean(),
                        "median_diff": diff.median(), "p_value": p, "n_days": len(diff)})
    return pd.DataFrame(out)


def robustness_table(df):
    """E4: capture vs the 0%-error LP oracle of the same day, per noise level."""
    d = df[df.experiment == "E4"].copy()
    noop, lp = _reference(df, "E4", 0.0, 0)
    key = list(zip(d.grid, d.day))
    d["noop0"] = noop.reindex(key).values
    d["lp0"] = lp.reindex(key).values
    rows = []
    for (g, m, nz), x in d.groupby(["grid", "method", "noise"]):
        per_seed = x.groupby("seed").agg(cost=("cost", "mean"), noop=("noop0", "mean"), lp=("lp0", "mean"))
        cap = (per_seed.noop - per_seed.cost) / (per_seed.noop - per_seed.lp) * 100
        rows.append({"grid": g, "method": m, "noise": nz, "cost": per_seed.cost.mean(),
                     "cost_ci": ci95(per_seed.cost), "capture": cap.mean(), "capture_ci": ci95(cap)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures --

def _grid_label(g):
    return "%s\n%s" % (g, GRIDS[g].name)


def fig_capture(tab, methods, path, title):
    grids = [g for g in GRID_IDS if g in set(tab.grid)]
    methods = [m for m in methods if m in set(tab.method)]
    fig, ax = plt.subplots(figsize=(1.9 * len(grids) + 2, 4))
    w = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        t = tab[tab.method == m].set_index("grid").reindex(grids)
        ax.bar(np.arange(len(grids)) + (i - (len(methods) - 1) / 2) * w, t.capture, w,
               yerr=t.capture_ci, capsize=2, label=m, color=COLORS.get(m))
    ax.axhline(100, color="k", lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(grids)), [_grid_label(g) for g in grids], fontsize=8)
    ax.set_ylabel("% of LP savings captured")
    ax.set_title(title, fontweight="bold")
    lo = np.nanmin(tab[tab.method.isin(methods)].capture)
    ax.set_ylim(max(min(lo - 10, -20), -150), 115)
    ax.legend(ncol=min(len(methods), 4), fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_transfer(df, path, algo="PPO"):
    d = df[(df.experiment == "E3T") & (df.method == algo)]
    if d.empty:
        return
    noop, lp = _reference(df, "E1")
    key = list(zip(d.grid, d.day))
    d = d.assign(noop=noop.reindex(key).values, lp=lp.reindex(key).values)
    grids = [g for g in GRID_IDS if g in set(d.grid)]
    M = np.full((len(grids), len(grids)), np.nan)
    for i, gi in enumerate(grids):
        for j, gj in enumerate(grids):
            x = d[(d.train_tag == gi) & (d.grid == gj)]
            if len(x):
                M[i, j] = ((x.noop.mean() - x.cost.mean()) / (x.noop.mean() - x.lp.mean())) * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.clip(M, -100, 100), cmap="RdYlGn", vmin=-100, vmax=100)
    for i in range(len(grids)):
        for j in range(len(grids)):
            ax.text(j, i, "%.0f" % M[i, j], ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(grids)), grids); ax.set_yticks(range(len(grids)), grids)
    ax.set_xlabel("evaluated on"); ax.set_ylabel("trained on")
    ax.set_title("%s cross-grid transfer (%% LP savings captured)" % algo, fontweight="bold", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_robustness(rob, path, methods):
    grids = [g for g in GRID_IDS if g in set(rob.grid)]
    n = len(grids)
    cols = min(3, n); rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(4.2 * cols, 3.2 * rows_), squeeze=False)
    for k, g in enumerate(grids):
        ax = axes[k // cols][k % cols]
        for m in methods:
            t = rob[(rob.grid == g) & (rob.method == m)].sort_values("noise")
            if t.empty:
                continue
            ls = "--" if m.startswith("LP") or m == "MPC" else "-"
            ax.plot(t.noise * 100, t.capture, "o" + ls, ms=4, label=m, color=COLORS.get(m))
            ax.fill_between(t.noise * 100, t.capture - t.capture_ci, t.capture + t.capture_ci,
                            alpha=0.15, color=COLORS.get(m))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(_grid_label(g).replace("\n", " - "), fontsize=9, fontweight="bold")
        ax.set_xlabel("forecast error (%)"); ax.set_ylabel("% LP(0%) savings")
        lo = rob[(rob.grid == g) & rob.method.isin(methods)].capture.min()
        ax.set_ylim(max(lo - 15, -250), 110)
    for k in range(n, rows_ * cols):
        axes[k // cols][k % cols].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 6), fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(path); plt.close(fig)


def fig_training(ckpt_dir, path):
    import torch
    files = sorted(Path(ckpt_dir).glob("*_s0.pt"))
    files = [f for f in files if f.stem.split("_")[1] in GRID_IDS + ["ALL"]]
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for f in files:
        st = torch.load(f, map_location="cpu", weights_only=False)
        h = [r for r in st.get("history", []) if "val" in r]
        if not h:
            continue
        algo, tag = f.stem.split("_")[:2]
        ax = axes[0] if st["algo"] == "ppo" else axes[1]
        steps = [r["steps"] for r in h]
        g0 = list(h[0]["val"].keys())[0]
        ax.plot(steps, [r["val"][g0] for r in h], label="%s %s (%s)" % (algo, tag, g0))
    for ax, t in zip(axes, ("PPO", "SAC")):
        ax.set_title("%s validation cost ($/day)" % t, fontweight="bold")
        ax.set_xlabel("env steps"); ax.set_yscale("symlog")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_dataset(path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for k, g in enumerate(GRID_IDS):
        f = PROC / ("%s.npz" % g)
        if not f.exists():
            continue
        d = np.load(f)
        ax = axes[k // 3][k % 3]
        doy = d["doy"]
        for season, (a, b), c in (("winter", (0, 59), "#3498db"), ("summer", (152, 243), "#e74c3c")):
            m = (doy >= a) & (doy <= b)
            ax.plot(d["load"][m].mean(0), color=c, label="load (%s)" % season)
            ax.plot(d["pv"][m].mean(0), color=c, ls="--", label="PV (%s)" % season)
        ax2 = ax.twinx()
        ax2.plot(d["price_import"].mean(0), color="k", lw=1, alpha=0.6, label="import price")
        ax2.set_ylim(0, max(0.35, d["price_import"].mean(0).max() * 1.2))
        ax2.grid(False); ax2.set_ylabel("$/kWh", fontsize=8)
        ax.set_title(_grid_label(g).replace("\n", " - "), fontsize=9, fontweight="bold")
        ax.set_xlabel("hour"); ax.set_ylabel("kW")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(path); plt.close(fig)


# ----------------------------------------------------------------- tables --

def latex_capture(tab, methods, path, caption, label):
    grids = [g for g in GRID_IDS if g in set(tab.grid)]
    methods = [m for m in methods if m in set(tab.method)]
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{%s}\\label{%s}" % (caption, label),
             "\\begin{tabular}{l" + "r" * len(grids) + "}", "\\toprule",
             "Method & " + " & ".join(grids) + " \\\\", "\\midrule"]
    for m in methods:
        cells = []
        for g in grids:
            r = tab[(tab.grid == g) & (tab.method == m)]
            if r.empty:
                cells.append("--"); continue
            r = r.iloc[0]
            ci = (" $\\pm$ %.1f" % r.capture_ci) if r.n_seeds > 1 else ""
            cells.append("%.1f%s" % (r.capture, ci))
        lines.append(m + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for m in methods[:1]:
        pass
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(path).write_text("\n".join(lines))


def latex_grids(path):
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Microgrid archetypes (all data: calendar year 2018).}\\label{tab:grids}",
             "\\begin{tabular}{llrrrll}", "\\toprule",
             "ID & Archetype & Battery & PV & Mean load & Tariff & Limits \\\\", "\\midrule"]
    for g in GRID_IDS:
        c = GRIDS[g]
        mf = PROC / ("%s_meta.json" % g)
        ml = json.loads(mf.read_text())["mean_load_kw"] if mf.exists() else float("nan")
        lim = []
        if c.import_cap_kw not in (None, float("inf")):
            lim.append("import $\\le$ %g kW" % c.import_cap_kw)
        if c.export_cap_kw not in (None, float("inf")):
            lim.append("export $\\le$ %g kW" % c.export_cap_kw)
        if c.demand_charge_per_kw_day:
            lim.append("DC \\$%.2f/kW-day" % c.demand_charge_per_kw_day)
        lines.append("%s & %s & %g kWh / %g kW & %g kWp & %.1f kW & %s & %s \\\\" % (
            g, c.name, c.e_max, c.p_bat_max, c.pv_kwp, ml, c.tariff.replace("_", "-"),
            ", ".join(lim) or "--"))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(path).write_text("\n".join(lines))


# ------------------------------------------------------------------- main --

def build_report(df=None, out=RESULTS):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = load() if df is None else df
    latex_grids(out / "table_grids.tex")
    fig_dataset(out / "fig_dataset_profiles.pdf")
    fig_training(out / "checkpoints", out / "fig_training_curves.pdf")
    written = []
    if "E1" in set(df.experiment):
        tab = capture_table(df)
        tab.to_csv(out / "summary_capture.csv", index=False)
        base = ["Rule-Based", "Self-Cons.", "MPC", "PPO", "SAC", "R-PPO", "R-SAC"]
        fig_capture(tab, base, out / "fig_e1_per_grid.pdf", "E1: per-grid benchmark (0% forecast error)")
        latex_capture(tab, ["NoOp", "Rule-Based", "Self-Cons.", "MPC", "LP (oracle)", "PPO", "SAC",
                            "R-PPO", "R-SAC", "Generalist PPO", "Generalist SAC",
                            "Zero-shot PPO", "Zero-shot SAC"], out / "table_capture.tex",
                      "Share of the perfect-foresight LP savings captured on held-out test days "
                      "(\\%, mean $\\pm$ 95\\% CI over seeds).", "tab:capture")
        written += ["summary_capture.csv", "fig_e1_per_grid.pdf", "table_capture.tex"]
        gen = [m for m in ["PPO", "Generalist PPO", "Zero-shot PPO", "SAC", "Generalist SAC", "Zero-shot SAC"]
               if m in set(tab.method)]
        if any(m.startswith("Generalist") or m.startswith("Zero") for m in gen):
            fig_capture(tab, gen, out / "fig_e2_e3_generalisation.pdf",
                        "E2/E3: specialist vs generalist vs zero-shot (unseen grid)")
            written.append("fig_e2_e3_generalisation.pdf")
        w = wilcoxon_vs(df)
        if not w.empty:
            w.to_csv(out / "stats_wilcoxon_vs_mpc.csv", index=False)
            written.append("stats_wilcoxon_vs_mpc.csv")
    if "E3T" in set(df.experiment):
        for algo in sorted(df[df.experiment == "E3T"].method.unique()):
            fig_transfer(df, out / ("fig_e3_transfer_%s.pdf" % algo.lower()), algo)
            written.append("fig_e3_transfer_%s.pdf" % algo.lower())
    if "E4" in set(df.experiment):
        rob = robustness_table(df)
        rob.to_csv(out / "summary_robustness.csv", index=False)
        fig_robustness(rob, out / "fig_e4_robustness.pdf",
                       ["LP (oracle)", "LP (open loop)", "MPC", "Self-Cons.", "PPO", "R-PPO", "SAC", "R-SAC"])
        written += ["summary_robustness.csv", "fig_e4_robustness.pdf"]
    print("report written to %s: %s" % (out, ", ".join(written)))
    return written


if __name__ == "__main__":
    build_report()
