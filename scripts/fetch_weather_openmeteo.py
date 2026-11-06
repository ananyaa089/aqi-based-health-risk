import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 28.6139, 77.2090  # central Delhi


def fetch_year(start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
        "wind_speed_unit": "ms",
        "timezone": "Asia/Kolkata",
    }
    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(h["time"]),
            "temp_c_city": h["temperature_2m"],
            "humidity_city": h["relative_humidity_2m"],
            "windspeed_ms_city": h["wind_speed_10m"],
            "pressure_hpa_city": h["surface_pressure"],
            "rain_mm_city": h["precipitation"],
        }
    )


def main():
    out = sys.argv[1]
    start = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2017-01-01")
    end = pd.Timestamp(sys.argv[3] if len(sys.argv) > 3 else "2025-12-31")

    frames = []
    for year in range(start.year, end.year + 1):  
        s = max(start, pd.Timestamp(year=year, month=1, day=1)).strftime("%Y-%m-%d")
        e = min(end, pd.Timestamp(year=year, month=12, day=31)).strftime("%Y-%m-%d")
        print(f"fetching {s} -> {e}")
        frames.append(fetch_year(s, e))
        time.sleep(1)

    df = pd.concat(frames, ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {len(df):,} hourly rows -> {out}")


if __name__ == "__main__":
    main()
