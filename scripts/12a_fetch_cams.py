import argparse
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
COORDS = {
    "Anand Vihar": (28.6469, 77.3158), "Ashok Vihar": (28.6954, 77.1818), "Aya Nagar": (28.4706, 77.1099),
    "Bawana": (28.7762, 77.0511), "Burari Crossing": (28.7256, 77.2011), "CRRI Mathura Road": (28.5512, 77.2736),
    "Chandni Chowk": (28.6578, 77.2273), "DTU": (28.7500, 77.1112), "Dwarka Sector 8": (28.5710, 77.0719),
    "R K Puram": (28.5633, 77.1869),
}


def parse_hourly(js):
    h = js.get("hourly") or {}
    if "time" not in h:
        return pd.DataFrame(columns=["timestamp", "cams_pm25", "cams_pm10"])
    return pd.DataFrame({"timestamp": pd.to_datetime(h["time"]), "cams_pm25": h.get("pm2_5"), "cams_pm10": h.get("pm10")})


def fetch(lat, lon, start, end, tries=4):
    params = dict(latitude=lat, longitude=lon, hourly="pm2_5,pm10", start_date=start, end_date=end, timezone="Asia/Kolkata")
    for k in range(tries):
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code == 200:
            return parse_hourly(r.json())
        if r.status_code == 429:
            time.sleep(20 * (k + 1))
        else:
            print(f"    HTTP {r.status_code}: {r.text[:150]}")
            time.sleep(3)
    return pd.DataFrame(columns=["timestamp", "cams_pm25", "cams_pm10"])


def daily(df):
    df = df.dropna(subset=["cams_pm25", "cams_pm10"], how="all").copy()
    df["date"] = df["timestamp"].dt.normalize()
    g = df.groupby("date")
    d = g[["cams_pm25", "cams_pm10"]].mean()
    d["n_hours"] = g["cams_pm25"].count()
    return d[d["n_hours"] >= 18].reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv")
    ap.add_argument("--start", default="2022-08-01")
    ap.add_argument("--end", default="2025-12-31")
    a = ap.parse_args()
    edges = pd.date_range(a.start, a.end, freq="6MS").union(pd.DatetimeIndex([pd.Timestamp(a.start)]))
    chunks = []
    for i, s in enumerate(sorted(edges)):
        e = min((sorted(edges)[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(edges) else pd.Timestamp(a.end), pd.Timestamp(a.end))
        chunks.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    frames = []
    for st, (lat, lon) in COORDS.items():
        parts = []
        for s, e in chunks:
            parts.append(fetch(lat, lon, s, e))
            time.sleep(0.7)
        raw = pd.concat(parts, ignore_index=True)
        d = daily(raw)
        d.insert(0, "station", st)
        print(f"{st}: {len(d)} daily rows" + (f" ({d['date'].min().date()} to {d['date'].max().date()})" if len(d) else " (NO DATA)"))
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_csv, index=False)
    print(f"Saved {len(out)} rows to {a.out_csv}")


if __name__ == "__main__":
    main()
