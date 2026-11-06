import sys
from pathlib import Path

import numpy as np
import pandas as pd

COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31")
TERMS = ["cough", "asthma", "inhaler"]


def deseason(s, week, mask):
    clim = s[mask].groupby(week[mask]).mean()
    return s - week.map(clim)


def corr(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    return (float(d.iloc[:, 0].corr(d.iloc[:, 1])), len(d)) if len(d) >= 30 else (np.nan, len(d))


def main():
    df = pd.read_csv(sys.argv[1], parse_dates=["week_start"]).sort_values("week_start").reset_index(drop=True)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    covid = (df["week_start"] >= COVID_START) & (df["week_start"] <= COVID_END)
    week = df["week_start"].dt.isocalendar().week.astype(int)
    winter = df["week_start"].dt.month.isin([10, 11, 12, 1])

    z = {}
    for t in [t for t in TERMS if t in df.columns]:
        s = df[t].where(~covid)
        s = s.clip(upper=s.quantile(0.975))
        z[t] = (s - s.mean()) / s.std()
    idx = pd.DataFrame(z)
    df["idx_new"] = idx.mean(axis=1)
    df["idx_new_smooth"] = df["idx_new"].rolling(3, min_periods=2, center=False).mean()

    for f in ("aqi_mean", "aqi_worst_station", "pct_severe_days"):
        if f in df.columns:
            df[f + "_2w"] = df[f].rolling(2, min_periods=1).mean()
    aqi_feats = [c for c in df.columns if c.startswith(("aqi_mean", "aqi_worst", "pct_severe"))]

    rows = []
    for target in ["idx_new", "idx_new_smooth"] + list(z):
        y = deseason(df[target], week, ~covid)
        for f in aqi_feats:
            x = deseason(df[f], week, ~covid)
            for lag in range(0, 4):
                for label, m in (("non-COVID", ~covid), ("Oct-Jan non-COVID", ~covid & winter)):
                    r, n = corr(x.shift(lag)[m], y[m])
                    rows.append(dict(target=target, aqi_feature=f, lag=lag, subset=label, r=round(r, 3), n=n))
    R = pd.DataFrame(rows)
    R.to_csv(out / "F_rescue.csv", index=False)

    print("Top 12 by |r| (n >= 60):")
    top = R[R.n >= 60].assign(a=lambda d: d.r.abs()).sort_values("a", ascending=False).head(12).drop(columns="a")
    print(top.to_string(index=False), "\n")

    print("Headline: new composite vs aqi_mean and aqi_worst_station, lag 0-1, non-COVID:")
    h = R[(R.target == "idx_new") & R.aqi_feature.isin(["aqi_mean", "aqi_worst_station"]) & R.lag.isin([0, 1]) & (R.subset == "non-COVID")]
    print(h.to_string(index=False), "\n")

    best = R[(R.n >= 60)].r.abs().max()
    print(f"Strongest |r| found: {best:.3f}")
    print("DECISION:", "weak signal worth keeping as a secondary feature" if best >= 0.25 else "no usable signal -> stop and pivot to AQI forecasting + literature-based health translation")
    print("Note: with ~20 target/feature/lag/subset combinations tested, a lone |r| ~0.2 can be chance.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main()
