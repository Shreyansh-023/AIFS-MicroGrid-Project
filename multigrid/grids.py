# multigrid/grids.py
# Microgrid archetypes: hardware sizing, tariff, grid limits and data sources.
#
# Every archetype is a GridConfig. G1 reproduces the original single-grid setup
# (13.5 kWh Powerwall, TOU tariff, 40% feed-in, 10 kW peak threshold) so results
# stay comparable with the SB3 pipeline in main.py.

from dataclasses import dataclass, field, asdict
import math
import numpy as np

INF = float("inf")


@dataclass
class GridConfig:
    gid: str
    name: str
    description: str

    # --- data sources -----------------------------------------------------
    county: str                    # NREL county code, e.g. "G4804530"
    state: str                     # two-letter state used in OEDI paths
    lat: float
    lon: float
    tz_offset: float               # local standard time offset from UTC (h)
    load_kind: str                 # "resstock_homes" | "comstock_aggregate"
    homes_per_profile: int = 1     # homes summed into one load profile
    n_profiles: int = 1            # independent profiles (each 365 days)
    annual_kwh_range: tuple = (6000.0, 18000.0)   # per-home filter (ResStock)
    comstock_type: str = ""        # e.g. "mediumoffice"
    floor_area_ft2: float = 0.0    # ComStock scaling target
    n_evs: int = 0                 # uncontrolled EV chargers added to load
    ev_charger_kw: float = 7.2

    # --- PV ---------------------------------------------------------------
    pv_kwp: float = 6.0
    pv_tilt: float = 25.0
    pv_azimuth: float = 180.0      # degrees from north (180 = south)

    # --- battery ----------------------------------------------------------
    e_max: float = 13.5
    e_min_ratio: float = 0.1
    p_bat_max: float = 5.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    ramp_rate: float = 2.5
    degradation_cost_per_kwh: float = 0.02
    target_final_soc: float = 0.5
    # terminal penalty in $ per kWh of deviation from the target energy.
    # G1: 50 $ per unit SOC on 13.5 kWh, identical to MicrogridEnv.
    terminal_penalty_per_kwh: float = 50.0 / 13.5

    # --- tariff -----------------------------------------------------------
    tariff: str = "tou_residential"  # see multigrid/tariffs.py
    flat_price: float = 0.13
    export_ratio: float = 0.4        # export price = ratio * import price
    rtp_adder: float = 0.10          # $/kWh delivery adder on wholesale LBMP
    demand_charge_per_kw_day: float = 0.0
    p_grid_peak: float = INF         # soft peak threshold (original model)
    peak_penalty_rate: float = 0.0   # $/kW above p_grid_peak, per hour

    # --- grid connection limits -------------------------------------------
    import_cap_kw: float = INF       # weak grid: excess load is shed
    export_cap_kw: float = INF       # export limit: excess PV is curtailed
    voll: float = 5.0                # value of lost load, $/kWh

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float) and math.isinf(v):
                d[k] = None
        return d

    @property
    def hours_of_storage(self):
        return self.e_max / self.p_bat_max


GRIDS = {
    "G1": GridConfig(
        gid="G1", name="Residential (hot)",
        description="Single-family home in Austin TX, TOU tariff, 40% feed-in. "
                    "Same battery/tariff as the original single-grid study.",
        county="G4804530", state="TX", lat=30.33, lon=-97.78, tz_offset=-6,
        load_kind="resstock_homes", homes_per_profile=1, n_profiles=3,
        annual_kwh_range=(9000.0, 18000.0),
        pv_kwp=6.0, pv_tilt=25.0,
        e_max=13.5, p_bat_max=5.0, ramp_rate=2.5,
        tariff="tou_residential", export_ratio=0.4,
        p_grid_peak=10.0, peak_penalty_rate=0.50,
    ),
    "G2": GridConfig(
        gid="G2", name="Residential (cold)",
        description="Single-family home in Minneapolis MN, flat tariff with a low "
                    "export credit: value comes from PV self-consumption.",
        county="G2700530", state="MN", lat=44.98, lon=-93.47, tz_offset=-6,
        load_kind="resstock_homes", homes_per_profile=1, n_profiles=3,
        annual_kwh_range=(6000.0, 14000.0),
        pv_kwp=5.0, pv_tilt=40.0,
        e_max=10.0, p_bat_max=5.0, ramp_rate=2.5,
        terminal_penalty_per_kwh=50.0 / 13.5,
        tariff="flat", flat_price=0.14, export_ratio=0.25,
    ),
    "G3": GridConfig(
        gid="G3", name="Commercial office",
        description="Medium office in Los Angeles CA, commercial TOU with a "
                    "daily demand charge and weekday/weekend schedule.",
        county="G0600370", state="CA", lat=34.05, lon=-118.25, tz_offset=-8,
        load_kind="comstock_aggregate", comstock_type="mediumoffice",
        floor_area_ft2=50000.0,
        pv_kwp=120.0, pv_tilt=20.0,
        e_max=200.0, p_bat_max=60.0, ramp_rate=30.0,
        degradation_cost_per_kwh=0.02,
        terminal_penalty_per_kwh=50.0 / 13.5,
        tariff="tou_commercial", export_ratio=0.3,
        demand_charge_per_kw_day=0.60,
    ),
    "G4": GridConfig(
        gid="G4", name="Rural weak grid",
        description="20-home rural community in Apache County AZ on a weak feeder: "
                    "import and export are capped; shed load is priced at VoLL.",
        county="G0400010", state="AZ", lat=35.40, lon=-109.50, tz_offset=-7,
        load_kind="resstock_homes", homes_per_profile=20, n_profiles=1,
        annual_kwh_range=(3000.0, 20000.0),
        pv_kwp=80.0, pv_tilt=30.0,
        e_max=200.0, p_bat_max=40.0, ramp_rate=20.0,
        terminal_penalty_per_kwh=50.0 / 13.5,
        tariff="flat", flat_price=0.12, export_ratio=0.3,
        # 30 kW cap: without a battery 1/3 of days shed load (~5.6 kWh/day),
        # worst hour exceeds the cap by 21 kW (< 40 kW inverter).
        import_cap_kw=30.0, export_cap_kw=30.0, voll=5.0,
    ),
    "G5": GridConfig(
        gid="G5", name="EV community (RTP)",
        description="10 homes with 8 uncontrolled EV chargers in Queens NY, "
                    "real-time pricing on NYISO NYC day-ahead LBMP, no export.",
        county="G3600810", state="NY", lat=40.73, lon=-73.80, tz_offset=-5,
        load_kind="resstock_homes", homes_per_profile=10, n_profiles=1,
        annual_kwh_range=(4000.0, 16000.0),
        n_evs=8,
        pv_kwp=40.0, pv_tilt=30.0,
        e_max=60.0, p_bat_max=25.0, ramp_rate=12.5,
        terminal_penalty_per_kwh=50.0 / 13.5,
        tariff="rtp_nyiso", rtp_adder=0.10, export_ratio=0.0,
        export_cap_kw=0.0,
    ),
    "G6": GridConfig(
        gid="G6", name="Industrial warehouse",
        description="Warehouse in Houston TX, industrial TOU with a high daily "
                    "demand charge and a slow-ramping battery.",
        county="G4802010", state="TX", lat=29.86, lon=-95.39, tz_offset=-6,
        load_kind="comstock_aggregate", comstock_type="warehouse",
        floor_area_ft2=250000.0,
        pv_kwp=300.0, pv_tilt=20.0,
        e_max=500.0, p_bat_max=125.0, ramp_rate=40.0,
        degradation_cost_per_kwh=0.015,
        terminal_penalty_per_kwh=50.0 / 13.5,
        tariff="tou_industrial", export_ratio=0.3,
        demand_charge_per_kw_day=1.00,
    ),
}

GRID_IDS = list(GRIDS.keys())


def get_grid(gid):
    return GRIDS[gid]


def finite(x, big=1e9):
    """Replace None/inf by a big finite number (for tensors / LP bounds)."""
    if x is None or (isinstance(x, float) and math.isinf(x)):
        return big
    return float(x)


def grid_descriptor(cfg, meta=None):
    """Static per-grid features appended to every observation.

    All values are dimensionless so one policy can read any grid.
    `meta` holds data statistics computed at dataset build time.
    """
    meta = meta or {}
    mean_load = meta.get("mean_load_kw", cfg.p_bat_max)
    mean_price = meta.get("mean_import_price", 0.15)
    cap_imp = finite(cfg.import_cap_kw, 1e3 * cfg.p_bat_max)
    cap_exp = finite(cfg.export_cap_kw, 1e3 * cfg.p_bat_max)
    return np.array([
        cfg.hours_of_storage / 8.0,                       # storage duration
        cfg.ramp_rate / cfg.p_bat_max,                    # ramp agility
        min(cfg.pv_kwp / cfg.p_bat_max, 10.0) / 5.0,      # PV vs inverter
        min(mean_load / cfg.p_bat_max, 10.0) / 5.0,       # load vs inverter
        cfg.export_ratio,                                 # export value
        cfg.demand_charge_per_kw_day / max(mean_price, 1e-3) / 10.0,
        min(cap_imp / cfg.p_bat_max, 10.0) / 10.0,        # import headroom
        min(cap_exp / cfg.p_bat_max, 10.0) / 10.0,        # export headroom
        cfg.degradation_cost_per_kwh / max(mean_price, 1e-3),
    ], dtype=np.float32)


DESCRIPTOR_DIM = 9
