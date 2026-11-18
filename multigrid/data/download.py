# multigrid/data/download.py
# Cached downloads of the public datasets used by the multi-grid study.
#
#   NREL End-Use Load Profiles (OEDI, AMY2018 release 1)
#     - ResStock individual homes  (parquet, 15-min kWh)
#     - ComStock county aggregates (csv, 15-min kWh)
#     - county AMY2018 weather     (csv, hourly GHI/DNI/DHI/temperature)
#   NYISO day-ahead zonal LBMP (monthly zip of daily csv)
#
# No API keys or personal data are sent. Everything is cached in data/raw/.

import os
import re
import time
import zipfile
import urllib.request
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

OEDI = "https://oedi-data-lake.s3.amazonaws.com"
EULP = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021"
RES = EULP + "/resstock_amy2018_release_1"
COM = EULP + "/comstock_amy2018_release_1"
NYISO = "http://mis.nyiso.com/public/csv/damlbmp"

HEADERS = {"User-Agent": "multigrid-research-client/1.0"}


def fetch(url, dest, retries=4, timeout=120):
    """Download `url` to `dest` unless it is already cached."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # network hiccup -> back off and retry
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("download failed after %d tries: %s (%s)" % (retries, url, last))


def list_s3(prefix, limit=None):
    """List object keys under an OEDI S3 prefix (public bucket listing)."""
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = OEDI + "/?" + urllib.parse.urlencode(q)
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode("utf-8")
        keys += re.findall(r"<Key>([^<]+)</Key>", body)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if limit and len(keys) >= limit:
            return keys[:limit]
        if not m:
            return keys
        token = m.group(1)


def weather_file(county):
    name = "%s_2018.csv" % county
    return fetch("%s/%s/weather/amy2018/%s" % (OEDI, RES, name), RAW / "weather" / name)


def resstock_home_keys(county, limit=400):
    cache = RAW / "resstock" / county / "_keys.txt"
    if cache.exists():
        return cache.read_text().split()
    prefix = "%s/timeseries_individual_buildings/by_county/upgrade=0/county=%s/" % (RES, county)
    keys = sorted(list_s3(prefix, limit=limit))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(keys))
    return keys


def resstock_home(county, key):
    name = key.rsplit("/", 1)[-1]
    return fetch("%s/%s" % (OEDI, key), RAW / "resstock" / county / name)


def comstock_aggregate(state, county, btype):
    name = "%s-%s.csv" % (county.lower(), btype)
    url = "%s/%s/timeseries_aggregates/by_county/state=%s/%s" % (OEDI, COM, state, name)
    return fetch(url, RAW / "comstock" / name)


def nyiso_month(year, month):
    """Return list of extracted daily CSV paths for one month of DAM zonal LBMP."""
    stem = "%04d%02d01damlbmp_zone_csv.zip" % (year, month)
    z = fetch("%s/%s" % (NYISO, stem), RAW / "nyiso" / stem)
    out_dir = RAW / "nyiso" / ("%04d%02d" % (year, month))
    if not out_dir.exists() or not any(out_dir.iterdir()):
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out_dir)
    return sorted(out_dir.glob("*.csv"))
