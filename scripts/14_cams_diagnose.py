import sys
import numpy as np
import pandas as pd

PM25_X, PM25_Y = [0, 30, 60, 90, 120, 250, 380], [0, 50, 100, 200, 300, 400, 500]
PM10_X, PM10_Y = [0, 50, 100, 250, 350, 430, 510], [0, 50, 100, 200, 300, 400, 500]


def main():
    m = pd.read_csv(sys.argv[1], parse_dates=["timestamp"])
    c = pd.read_csv(sys.argv[2], parse_dates=["date"])
    m["date"] = m["timestamp"].dt.normalize()
    agg = {"aqi": ("aqi", "mean"), "n": ("aqi", "count")}
    for col in ("pm2_5", "pm10"):
        if col in m.columns:
            agg[col] = (col, "mean")
    d = m.groupby(["station", "date"]).agg(**agg).reset_index()
    d = d[d["n"] >= 12]
    c["cams_aqi"] = np.maximum(np.interp(c["cams_pm25"], PM25_X, PM25_Y), np.interp(c["cams_pm10"], PM10_X, PM10_Y))
    x = d.merge(c, on=["station", "date"], how="inner")
    x["year"] = x["date"].dt.year
    x["winter"] = x["date"].dt.month.isin([10, 11, 12, 1, 2])

    def r(a, b):
        z = pd.concat([a, b], axis=1).dropna()
        return round(float(z.iloc[:, 0].corr(z.iloc[:, 1])), 3) if len(z) > 30 else np.nan

    rows = [dict(year=y, n=len(g), r_all=r(g.cams_aqi, g.aqi), r_winter=r(g[g.winter].cams_aqi, g[g.winter].aqi),
                 mean_station_aqi=round(g.aqi.mean(), 0), mean_cams_aqi=round(g.cams_aqi.mean(), 0)) for y, g in x.groupby("year")]
    print("A. CAMS-derived AQI vs station AQI, by year:\n", pd.DataFrame(rows).to_string(index=False))

    print("\nB. Correlation with CAMS shifted by k days (k=0 expected to peak):")
    for k in range(-2, 3):
        cs = c.copy()
        cs["date"] = cs["date"] + pd.Timedelta(days=k)
        z = d.merge(cs, on=["station", "date"])
        print(f"   shift {k:+d}: r = {r(z.cams_aqi, z.aqi)}  (n={len(z)})")

    if "pm2_5" in x.columns:
        z = x[x.pm2_5.notna()]
        print("\nC. Raw PM vs CAMS PM (2024-25 station sensors):")
        for a, b, nm in (("pm2_5", "cams_pm25", "PM2.5"), ("pm10", "cams_pm10", "PM10")):
            q = z[[a, b]].dropna()
            print(f"   {nm}: r = {q[a].corr(q[b]):.3f}, n = {len(q)}, median station/CAMS ratio = {(q[a] / q[b]).median():.2f}")

    if "pm2_5" in x.columns and "pm10" in x.columns:
        z = d.dropna(subset=["pm2_5", "pm10"]).copy()
        z["year"] = z["date"].dt.year
        z["pm10_below_pm25"] = z["pm10"] < z["pm2_5"]
        print("\nE. Station sensor sanity (PM10 can never be below PM2.5), share of station-days where it is:")
        print(z[z.year >= 2024].groupby(["station", "year"])["pm10_below_pm25"].mean().unstack("year").round(3).to_string())
        print(f"   station PM10 vs PM2.5 correlation 2024-25: {z[z.year >= 2024].pm10.corr(z[z.year >= 2024].pm2_5):.3f}")
        q = x[x.pm2_5.notna()].copy()
        q["ym"] = q["date"].dt.to_period("M")
        mo = q.groupby("ym").apply(lambda g: pd.Series({"median_station_over_cams_pm25": (g.pm2_5 / g.cams_pm25).median(),
                                                         "r_pm25": g.pm2_5.corr(g.cams_pm25)}), include_groups=False).round(2)
        print("\nF. Month by month, station PM2.5 vs CAMS PM2.5 (a sudden jump in the ratio = CAMS or sensor regime change):")
        print(mo.to_string())

    print("\nD. Per-station r (CAMS AQI vs station AQI):")
    t = x.assign(era=np.where(x.year <= 2023, "2023", "2024-25")).groupby(["station", "era"]).apply(lambda g: r(g.cams_aqi, g.aqi), include_groups=False).unstack("era")
    print(t.to_string())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main()
