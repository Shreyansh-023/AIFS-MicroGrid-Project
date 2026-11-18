# multigrid/tariffs.py
# Hourly import/export price builders for each tariff family.
#
# Schedules are simplified versions of published tariffs:
#   tou_residential : original study TOU (see data_loader.get_tou_prices)
#   flat            : single energy rate (e.g. Xcel MN residential ~$0.14/kWh)
#   tou_commercial  : SCE TOU-GS style, on-peak 16-21 on weekdays
#   tou_industrial  : TX industrial TOU, on-peak 12-20 on weekdays
#   rtp_nyiso       : NYISO day-ahead LBMP pass-through + delivery adder
# Demand charges are expressed per day ($/kW of the day's max import), i.e. a
# monthly $/kW-month charge divided by ~30.

import numpy as np

from data_loader import get_tou_prices


def _weekday_mask(dates):
    return np.array([d.weekday() < 5 for d in dates])


def build_prices(cfg, dates, lbmp_kwh=None):
    """Return (import[D,24], export[D,24]) in $/kWh for the given dates."""
    D = len(dates)
    weekday = _weekday_mask(dates)

    if cfg.tariff == "tou_residential":
        imp = np.tile(get_tou_prices(24).astype(np.float64), (D, 1))

    elif cfg.tariff == "flat":
        imp = np.full((D, 24), cfg.flat_price)

    elif cfg.tariff == "tou_commercial":
        wk = np.full(24, 0.10)
        wk[8:16] = 0.15
        wk[16:21] = 0.32
        we = np.full(24, 0.10)
        we[8:21] = 0.13
        imp = np.where(weekday[:, None], wk[None, :], we[None, :])

    elif cfg.tariff == "tou_industrial":
        wk = np.full(24, 0.07)
        wk[12:20] = 0.14
        we = np.full(24, 0.07)
        imp = np.where(weekday[:, None], wk[None, :], we[None, :])

    elif cfg.tariff == "rtp_nyiso":
        if lbmp_kwh is None:
            raise ValueError("rtp_nyiso tariff needs LBMP data")
        imp = np.maximum(lbmp_kwh + cfg.rtp_adder, 0.01)

    else:
        raise ValueError("unknown tariff %s" % cfg.tariff)

    if cfg.tariff == "rtp_nyiso":
        exp = np.maximum(lbmp_kwh, 0.0) * (1.0 if cfg.export_ratio > 0 else 0.0)
    else:
        exp = imp * cfg.export_ratio
    return imp.astype(np.float32), exp.astype(np.float32)
