import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_legacy_2017_2023 import parse_legacy_file  
from resample_2024_25 import resample_file  

STATIONS = {
    "Anand Vihar": "anand vihar",
    "Ashok Vihar": "ashok vihar",
    "Aya Nagar": "aya nagar",
    "Bawana": "bawana",
    "Burari Crossing": "burari crossing",
    "Chandni Chowk": "chandni chowk",
    "CRRI Mathura Road": "crri mathura road",
    "DTU": "dtu",
    "Dwarka Sector 8": "dwarka sector 8",
    "R K Puram": "r k puram",
}

PM25_X = [0, 30, 60, 90, 120, 250, 380]
PM25_Y = [0, 50, 100, 200, 300, 400, 500]
PM10_X = [0, 50, 100, 250, 350, 430, 510]
PM10_Y = [0, 50, 100, 200, 300, 400, 500]


def compute_aqi(df: pd.DataFrame) -> pd.Series:
    pm25 = df["pm2_5"].rolling(24, min_periods=16).mean()
    pm10 = df["pm10"].rolling(24, min_periods=16).mean()
    s25 = pd.Series(np.interp(pm25, PM25_X, PM25_Y), index=df.index).where(pm25.notna())
    s10 = pd.Series(np.interp(pm10, PM10_X, PM10_Y), index=df.index).where(pm10.notna())
    return np.fmax(s25, s10)


def find_files(data_dir: Path, prefix: str):
    legacy, recent = [], []
    for p in sorted(data_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not name.startswith(prefix + " "):
            continue
        (recent if "15 minute" in name else legacy).append(p)
    return legacy, recent


def build_station(station: str, legacy_files, recent_files) -> pd.DataFrame:
    parts = []

    for f in legacy_files:
        leg = parse_legacy_file(str(f), station)
        leg["aqi_source"] = "reported"
        parts.append(leg)

    for f in recent_files:
        rec = resample_file(str(f), station)
        rec = rec.sort_values("timestamp").set_index("timestamp")
        rec["aqi"] = compute_aqi(rec)
        rec = rec.reset_index()
        rec["aqi_source"] = "computed_pm_formula"
        parts.append(rec)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True, sort=False)
    df = df.sort_values("timestamp").drop_duplicates(["timestamp", "station"], keep="last")

    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="1h")
    df = (
        df.set_index("timestamp")
        .reindex(full)
        .rename_axis("timestamp")
        .reset_index()
    )
    df["station"] = station
    return df


def coverage_report(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["year"] = m["timestamp"].dt.year
    return (
        m.groupby(["station", "year"])["aqi"]
        .apply(lambda s: round(100 * s.notna().mean(), 1))
        .unstack("year")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("output_csv")
    ap.add_argument("--weather", help="optional hourly city-wide weather CSV to left-join on timestamp")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    frames = []
    for station, prefix in STATIONS.items():
        legacy, recent = find_files(data_dir, prefix)
        print(f"{station}: {len(legacy)} legacy file(s), {len(recent)} 2024-25 file(s)")
        if not legacy or not recent:
            print(f"  WARNING: expected 1 of each for {station}; check filenames in {data_dir}")
        if not legacy and not recent:
            continue
        frames.append(build_station(station, legacy, recent))

    master = pd.concat([f for f in frames if len(f)], ignore_index=True)

    if args.weather:
        w = pd.read_csv(args.weather, parse_dates=["timestamp"])
        master = master.merge(w, on="timestamp", how="left", suffixes=("", "_city"))

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output_csv, index=False)
    print(f"\nWrote {len(master):,} rows -> {args.output_csv}")
    print("\n% of hours with a valid AQI, by station and year:")
    print(coverage_report(master).to_string())


if __name__ == "__main__":
    main()
