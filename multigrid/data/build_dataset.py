# multigrid/data/build_dataset.py
# Build per-grid day banks: data/processed/<GID>.npz
#
# Arrays (D = n_profiles * 365 days):
#   load[D,24], pv[D,24]               kW (hourly mean)
#   price_import[D,24], price_export   $/kWh
#   profile[D] int, doy[D] int, weekday[D] int, split[D] int (0 train, 1 val, 2 test)
#   dates[D] str
# Plus <GID>_meta.json with data statistics used by the observation descriptor.
#
# Conventions: NREL timestamps are interval-ending in local standard time, so
# a 15-min stamp 00:15 belongs to hour 0 and 01:00 also belongs to hour 0.
# Split: for every month, days 8-14 are test, 22-28 validation, rest training.

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from multigrid.grids import GRIDS, GRID_IDS
from multigrid.tariffs import build_prices
from multigrid.data import download as dl
from multigrid.data.pv_model import pv_ac_per_kwp

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
YEAR = 2018
DATES = pd.date_range("%d-01-01" % YEAR, "%d-12-31" % YEAR, freq="D")


def split_of(dates):
    day = np.array([d.day for d in dates])
    s = np.zeros(len(dates), dtype=np.int64)
    s[(day >= 8) & (day <= 14)] = 2
    s[(day >= 22) & (day <= 28)] = 1
    return s


def hourly_matrix(ts, values, how="mean"):
    """Interval-ending timestamps -> [365, 24] hourly matrix for YEAR."""
    ts = pd.DatetimeIndex(ts) - pd.Timedelta(minutes=1)
    s = pd.Series(np.asarray(values, dtype=np.float64), index=ts)
    s = s.resample("h").mean() if how == "mean" else s.resample("h").sum()
    full = pd.date_range("%d-01-01 00:00" % YEAR, "%d-12-31 23:00" % YEAR, freq="h")
    s = s.reindex(full).interpolate(limit_direction="both")
    return s.values.reshape(365, 24)


# ---------------------------------------------------------------- loads ----

def resstock_load(cfg, rng):
    """[n_profiles, 365, 24] kW, each profile the sum of `homes_per_profile` homes."""
    keys = dl.resstock_home_keys(cfg.county)
    order = rng.permutation(len(keys))
    need = cfg.n_profiles * cfg.homes_per_profile
    lo, hi = cfg.annual_kwh_range
    homes = []
    for i in order:
        path = dl.resstock_home(cfg.county, keys[i])
        df = pd.read_parquet(path, columns=["timestamp", "out.electricity.total.energy_consumption"])
        kwh = df["out.electricity.total.energy_consumption"].values
        annual = kwh.sum()
        if not (lo <= annual <= hi):
            continue
        kw = kwh * 4.0                                   # 15-min kWh -> kW
        homes.append(hourly_matrix(df["timestamp"].values, kw))
        print("    home %-14s %7.0f kWh/yr  (%d/%d)" % (keys[i].rsplit("/", 1)[-1], annual, len(homes), need))
        if len(homes) == need:
            break
    if len(homes) < need:
        raise RuntimeError("%s: only %d homes in range %s" % (cfg.gid, len(homes), cfg.annual_kwh_range))
    homes = np.stack(homes).reshape(cfg.n_profiles, cfg.homes_per_profile, 365, 24)
    return homes.sum(axis=1)


def comstock_load(cfg):
    path = dl.comstock_aggregate(cfg.state, cfg.county, cfg.comstock_type)
    df = pd.read_csv(path, usecols=["timestamp", "floor_area_represented",
                                    "out.electricity.total.energy_consumption"])
    scale = cfg.floor_area_ft2 / df["floor_area_represented"].iloc[0]
    kw = df["out.electricity.total.energy_consumption"].values * 4.0 * scale
    return hourly_matrix(pd.to_datetime(df["timestamp"]).values, kw)[None]


def ev_load(cfg, rng):
    """Uncontrolled home EV charging, [365, 24] kW summed over `n_evs`.

    Arrival ~ N(18.0, 1.5) h on weekdays, N(15.0, 3.0) on weekends (INL EV
    Project / NREL EVI-Pro residential L2 patterns); daily energy ~ lognormal
    with median 8 kWh; 15% of EV-days have no charging session.
    """
    out = np.zeros((365, 24))
    for d, date in enumerate(DATES):
        weekend = date.weekday() >= 5
        for _ in range(cfg.n_evs):
            if rng.random() < 0.15:
                continue
            arr = rng.normal(15.0, 3.0) if weekend else rng.normal(18.0, 1.5)
            arr = float(np.clip(arr, 6.0, 23.5))
            energy = float(np.clip(rng.lognormal(np.log(8.0), 0.5), 2.0, 40.0))
            t, remaining = arr, energy
            while remaining > 1e-6 and t < 24.0 + 12.0:
                h = int(t)
                dt = min(h + 1 - t, remaining / cfg.ev_charger_kw)
                e = cfg.ev_charger_kw * dt
                if h < 24:
                    out[d, h] += e
                elif d + 1 < 365:                       # after midnight -> next day
                    out[d + 1, h - 24] += e
                remaining -= e
                t += dt
    return out


# ------------------------------------------------------------------ PV -----

def pv_matrix(cfg):
    path = dl.weather_file(cfg.county)
    w = pd.read_csv(path)
    w.columns = [c.split(" [")[0].strip() for c in w.columns]
    ts = pd.to_datetime(w["date_time"])
    mid = pd.DatetimeIndex(ts - pd.Timedelta(minutes=30))      # hour-ending -> mid-hour
    ac = pv_ac_per_kwp(mid, w["Global Horizontal Radiation"].values,
                       w["Direct Normal Radiation"].values,
                       w["Diffuse Horizontal Radiation"].values,
                       w["Dry Bulb Temperature"].values,
                       cfg.lat, cfg.lon, cfg.tz_offset, tilt=cfg.pv_tilt,
                       azimuth=cfg.pv_azimuth)
    return hourly_matrix(ts.values, ac) * cfg.pv_kwp


# --------------------------------------------------------------- prices ----

def nyiso_lbmp_matrix(zone="N.Y.C."):
    """NYISO DAM zonal LBMP for YEAR in $/kWh, local standard time [365,24]."""
    frames = []
    for m in range(1, 13):
        for f in dl.nyiso_month(YEAR, m):
            df = pd.read_csv(f)
            frames.append(df[df["Name"] == zone])
    df = pd.concat(frames)
    ts = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M")
    # stamps are Eastern prevailing time; rows of the DST fall-back hour repeat.
    loc = ts.dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
    std = loc.dt.tz_convert("Etc/GMT+5").dt.tz_localize(None)   # EST, no DST
    s = pd.Series(df["LBMP ($/MWHr)"].values / 1000.0, index=std).groupby(level=0).mean()
    full = pd.date_range("%d-01-01 00:00" % YEAR, "%d-12-31 23:00" % YEAR, freq="h")
    s = s.reindex(full).interpolate(limit_direction="both")
    return s.values.reshape(365, 24)


# ---------------------------------------------------------------- build ----

def build_grid(gid, seed=0):
    cfg = GRIDS[gid]
    rng = np.random.default_rng(seed + int(gid[1:]) * 1000)
    print("[%s] %s" % (gid, cfg.name))

    if cfg.load_kind == "resstock_homes":
        load = resstock_load(cfg, rng)
    elif cfg.load_kind == "comstock_aggregate":
        load = comstock_load(cfg)
    else:
        raise ValueError(cfg.load_kind)
    if cfg.n_evs:
        load = load + ev_load(cfg, rng)[None]

    pv = pv_matrix(cfg)
    lbmp = nyiso_lbmp_matrix() if cfg.tariff == "rtp_nyiso" else None
    imp, exp = build_prices(cfg, list(DATES), lbmp)

    P = load.shape[0]
    arrays = {
        "load": load.reshape(P * 365, 24).astype(np.float32),
        "pv": np.tile(pv, (P, 1)).astype(np.float32),
        "price_import": np.tile(imp, (P, 1)),
        "price_export": np.tile(exp, (P, 1)),
        "profile": np.repeat(np.arange(P), 365),
        "doy": np.tile(np.arange(365), P),
        "weekday": np.tile(np.array([d.weekday() for d in DATES]), P),
        "split": np.tile(split_of(DATES), P),
        "dates": np.tile(np.array([str(d.date()) for d in DATES]), P),
    }
    net = arrays["load"] - arrays["pv"]
    meta = {
        "gid": gid,
        "n_days": int(P * 365),
        "n_profiles": int(P),
        "mean_load_kw": float(arrays["load"].mean()),
        "peak_load_kw": float(arrays["load"].max()),
        "mean_daily_load_kwh": float(arrays["load"].sum(1).mean()),
        "mean_daily_pv_kwh": float(arrays["pv"].sum(1).mean()),
        "mean_import_price": float(arrays["price_import"].mean()),
        "max_import_price": float(arrays["price_import"].max()),
        "p95_daily_peak_net_kw": float(np.percentile(net.max(1), 95)),
        "p50_daily_peak_net_kw": float(np.percentile(net.max(1), 50)),
        "min_net_kw": float(net.min()),
        "pv_peak_hour_mean": float(arrays["pv"].mean(0).argmax()),
        "split_counts": {k: int((arrays["split"] == v).sum()) for k, v in
                         (("train", 0), ("val", 1), ("test", 2))},
        "config": cfg.to_dict(),
    }
    PROC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROC / ("%s.npz" % gid), **arrays)
    (PROC / ("%s_meta.json" % gid)).write_text(json.dumps(meta, indent=2))
    print("    load %.2f kW mean (peak %.1f) | PV %.1f kWh/day | price %.3f $/kWh | "
          "PV peak hour %d" % (meta["mean_load_kw"], meta["peak_load_kw"],
                                meta["mean_daily_pv_kwh"], meta["mean_import_price"],
                                meta["pv_peak_hour_mean"]))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="all")
    args = ap.parse_args()
    gids = GRID_IDS if args.grids == "all" else args.grids.split(",")
    for g in gids:
        build_grid(g)


if __name__ == "__main__":
    main()
