import sys
import re
import csv
import pandas as pd
from pathlib import Path

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

MONTH_HEADER_RE = re.compile(r"^([A-Za-z]+)-(\d{4})$")


def parse_legacy_file(path: str, station: str) -> pd.DataFrame:
    rows_out = []
    current_year = None
    current_month = None

    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for raw_row in reader:
            if not raw_row:
                continue
            first = raw_row[0].strip() if raw_row[0] is not None else ""

            if first.lower() == "year" and len(raw_row) > 1:
                try:
                    current_year = int(raw_row[1].strip())
                except ValueError:
                    pass
                continue

            if first in ('"', "") and all(
                (c is None or c.strip() in ("", '"')) for c in raw_row
            ):
                continue

            m = MONTH_HEADER_RE.match(first)
            if m:
                month_name, year_str = m.group(1), m.group(2)
                month_name_lower = month_name.lower()
                if month_name_lower in MONTHS:
                    current_month = MONTHS[month_name_lower]
                    current_year = int(year_str)
                continue

            if current_year is None or current_month is None:
                continue

            day_str = first.replace(".0", "").strip()
            if not day_str.isdigit():
                continue
            day = int(day_str)

            values = raw_row[1:25]  
            for hour, val in enumerate(values):
                if val is None:
                    continue
                val = val.strip()
                if val == "":
                    continue
                try:
                    num = float(val)
                except ValueError:
                    continue
                try:
                    ts = pd.Timestamp(
                        year=current_year, month=current_month, day=day, hour=hour
                    )
                except ValueError:
                    continue
                rows_out.append((ts, station, num))

    df = pd.DataFrame(rows_out, columns=["timestamp", "station", "aqi"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp", "station"])
    return df


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    input_path, station_name, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    df = parse_legacy_file(input_path, station_name)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Parsed {len(df)} rows for station '{station_name}' -> {output_path}")
    if len(df):
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        print("WARNING: 0 rows parsed. Open the input file and check it matches the expected format.")
