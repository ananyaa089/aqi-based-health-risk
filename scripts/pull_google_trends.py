import sys
import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

TERMS = ["cough", "asthma", "breathlessness", "inhaler", "bronchitis"]
GEO = "IN-DL"
WINDOWS = [
    ("2017-01-01", "2021-06-30"),
    ("2020-10-01", "2025-03-31"),
    ("2024-07-01", "2025-12-31"),
]


def pull_window(pt: TrendReq, term: str, start: str, end: str, retries: int = 4) -> pd.Series:
    for attempt in range(retries):
        try:
            pt.build_payload([term], geo=GEO, timeframe=f"{start} {end}")
            df = pt.interest_over_time()
            return df[term].astype(float)
        except Exception as exc:  
            wait = 30 * (attempt + 1)
            print(f"  {term} {start}->{end}: {exc!r}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up on {term} {start}->{end}")


def stitch(windows: list[pd.Series]) -> pd.Series:
    combined = windows[0]
    for nxt in windows[1:]:
        overlap = combined.index.intersection(nxt.index)
        a, b = combined.loc[overlap].mean(), nxt.loc[overlap].mean()
        scale = a / b if b else 1.0
        nxt = nxt * scale
        combined = pd.concat([combined, nxt.loc[nxt.index.difference(combined.index)]]).sort_index()
    return combined


def main():
    out = sys.argv[1]
    pt = TrendReq(hl="en-IN", tz=330)
    series = {}
    for term in TERMS:
        print(f"term: {term}")
        parts = []
        for start, end in WINDOWS:
            parts.append(pull_window(pt, term, start, end))
            time.sleep(8)
        series[term] = stitch(parts)

    df = pd.DataFrame(series)
    df["resp_interest_index"] = df[TERMS].mean(axis=1)
    df.index.name = "week_start"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"saved {len(df)} weekly rows -> {out}")


if __name__ == "__main__":
    main()
