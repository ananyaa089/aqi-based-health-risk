import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_master import PM25_X, PM25_Y, PM10_X, PM10_Y  

OFFICIAL_DEC = {2021: 336, 2022: 319, 2023: 348, 2024: 238}  


def main():
    m = pd.read_csv(sys.argv[1], parse_dates=["timestamp"])
    m["date"] = m["timestamp"].dt.normalize()
    m["year"], m["month"] = m["timestamp"].dt.year, m["timestamp"].dt.month
    nd = m[m["month"].isin([11, 12]) & m["year"].between(2021, 2025)].copy()

    agg = {"aqi": ("aqi", "mean"), "n": ("aqi", "count")}
    for c in ("pm2_5", "pm10"):
        if c in nd.columns:
            agg[c] = (c, "mean")
    sd = nd.groupby(["station", "year", "date"]).agg(**agg).reset_index()
    sd = sd[sd["n"] >= 12]

    t1 = sd.groupby(["station", "year"])["aqi"].mean().unstack("year").round(0)
    print("1. Mean AQI, Nov-Dec, by station and year (2021-23 reported; 2024-25 computed):\n", t1.to_string(), "\n")

    days = sd.groupby(["station", "year"])["date"].nunique().unstack("year")
    stable = days.index[(days >= 40).all(axis=1)]
    print("2. Stations with >=40 valid Nov-Dec days in every year 2021-25:", list(stable))
    if len(stable):
        c = sd[sd["station"].isin(stable)].groupby("year")["aqi"].mean().round(0)
        d = sd[sd["station"].isin(stable) & (sd["date"].dt.month == 12)].groupby("year")["aqi"].mean().round(0)
        out = pd.DataFrame({"Nov-Dec mean (stable stations)": c, "Dec only": d})
        out["Dec official (press)"] = [OFFICIAL_DEC.get(y, np.nan) for y in out.index]
        print(out.to_string(), "\n")

    if {"pm2_5", "pm10"} <= set(sd.columns):
        s = sd.dropna(subset=["pm2_5", "pm10"]).copy()
        s["sub25"] = np.interp(s["pm2_5"], PM25_X, PM25_Y)
        s["sub10"] = np.interp(s["pm10"], PM10_X, PM10_Y)
        s["pm10_drives"] = s["sub10"] > s["sub25"]
        t3 = s.groupby("year").agg(
            mean_pm25=("pm2_5", "mean"), mean_pm10=("pm10", "mean"),
            mean_sub_pm25=("sub25", "mean"), mean_sub_pm10=("sub10", "mean"),
            share_days_pm10_drives=("pm10_drives", "mean"),
        ).round(2)
        print("3. Computed-era raw levels (Nov-Dec, station-day means) and which sub-index drives AQI:\n", t3.to_string(), "\n")
    else:
        print("3. (no pm2_5 / pm10 columns in this master file, skipping)\n")

    print("How to read this:")
    print(" - If 2024-25 stays high even for the stable-station set, composition is not the cause.")
    print(" - If PM10 drives most days, check PM10 raw levels first (dust, sensor spikes, uncalibrated readings).")
    print(" - Either way, 2017-23 (reported) and 2024-25 (computed) should not be pooled as one AQI scale.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main()
