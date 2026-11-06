import sys
import pandas as pd
from pathlib import Path

import re

SHORT_TO_NAME = {
    "PM2.5": "pm2_5",
    "PM10": "pm10",
    "NO": "no",
    "NO2": "no2",
    "NOx": "nox",
    "NH3": "nh3",
    "SO2": "so2",
    "CO": "co",
    "Ozone": "ozone",
    "AT": "temp_c",
    "RH": "humidity",
    "WS": "windspeed_ms",
    "BP": "pressure_mmhg",
}


def _short(col: str) -> str:
    return re.split(r"\s*\(", str(col).strip())[0].strip()


def _read_csv_any_encoding(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} with utf-8, cp1252 or latin-1")


def resample_file(path: str, station: str) -> pd.DataFrame:
    df = _read_csv_any_encoding(path)

    ts_cols = [c for c in df.columns if str(c).strip().lower() == "timestamp"]
    if not ts_cols:
        raise ValueError(
            f"Expected a 'Timestamp' column in {path}, got: {list(df.columns)[:8]}... "
            "Check this file's header matches the expected 2024-25 format."
        )

    df["timestamp"] = (
        pd.to_datetime(df[ts_cols[0]], utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )

    available = {c: SHORT_TO_NAME[_short(c)] for c in df.columns if _short(c) in SHORT_TO_NAME}
    missing = [n for n in SHORT_TO_NAME.values() if n not in available.values()]
    if missing:
        print(f"  Note: columns not found in this file (skipped): {missing}")

    df = df[["timestamp"] + list(available.keys())].rename(columns=available)

    for col in available.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("timestamp").resample("1h").mean(numeric_only=True).reset_index()
    df.insert(1, "station", station)
    return df


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    input_path, station_name, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    df = resample_file(input_path, station_name)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Resampled {len(df)} hourly rows for station '{station_name}' -> {output_path}")
    if len(df):
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
