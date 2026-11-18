# multigrid/data/pv_model.py
# PVWatts-style PV output from hourly irradiance (GHI/DNI/DHI) and temperature.
#
# Steps: NOAA solar position -> plane-of-array irradiance (isotropic sky) ->
# NOCT cell temperature -> DC power with temperature coefficient -> fixed
# system losses -> inverter efficiency and AC clipping.
# Follows NREL PVWatts v5 defaults (Dobos 2014, NREL/TP-6A20-62641).

import numpy as np


def solar_position(times_local_std, lat, lon, tz_offset):
    """Zenith and azimuth (deg) for naive local-standard-time timestamps.

    `times_local_std` is a pandas DatetimeIndex (mid-interval times).
    """
    doy = times_local_std.dayofyear.values
    hour = (times_local_std.hour.values + times_local_std.minute.values / 60.0)
    gamma = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)

    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))

    time_offset = eqtime + 4 * lon - 60 * tz_offset          # minutes
    tst = hour * 60 + time_offset                             # true solar time
    ha = np.radians(tst / 4.0 - 180.0)                        # hour angle

    phi = np.radians(lat)
    cos_zen = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    cos_zen = np.clip(cos_zen, -1, 1)
    zen = np.arccos(cos_zen)

    sin_zen = np.maximum(np.sin(zen), 1e-6)
    cos_az = (np.sin(decl) - np.sin(phi) * cos_zen) / (np.cos(phi) * sin_zen)
    az = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
    az = np.where(ha > 0, 360.0 - az, az)                     # afternoon -> west
    return np.degrees(zen), az


def pv_ac_per_kwp(times_local_std, ghi, dni, dhi, temp_c, lat, lon, tz_offset,
                  tilt=25.0, azimuth=180.0, albedo=0.2, losses=0.14,
                  gamma_pdc=-0.0037, noct=45.0, inv_eff=0.96, dc_ac_ratio=1.2):
    """AC output (kW) of a 1 kWp(DC) system for each timestamp."""
    zen, az = solar_position(times_local_std, lat, lon, tz_offset)
    zr, tr = np.radians(zen), np.radians(tilt)
    cos_aoi = (np.cos(zr) * np.cos(tr)
               + np.sin(zr) * np.sin(tr) * np.cos(np.radians(az - azimuth)))
    cos_aoi = np.clip(cos_aoi, 0.0, 1.0)
    sun_up = zen < 90.0

    beam = np.where(sun_up, dni * cos_aoi, 0.0)
    sky = dhi * (1 + np.cos(tr)) / 2.0
    ground = ghi * albedo * (1 - np.cos(tr)) / 2.0
    poa = np.maximum(beam + sky + ground, 0.0)

    t_cell = temp_c + poa / 800.0 * (noct - 20.0)
    pdc = poa / 1000.0 * (1 + gamma_pdc * (t_cell - 25.0))
    pdc = np.maximum(pdc, 0.0) * (1 - losses)
    pac = np.minimum(pdc * inv_eff, 1.0 / dc_ac_ratio)
    return pac
