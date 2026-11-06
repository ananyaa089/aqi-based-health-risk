import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_HOURS_PER_DAY = 12   
MIN_DAYS_PER_WEEK = 4
MAX_LAG = 8


def week_start(ts: pd.Series) -> pd.Series:
    d = ts.dt.normalize()
    return d - pd.to_timedelta((d.dt.weekday + 1) % 7, unit="D")


def build_city_weekly(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["date"] = m["timestamp"].dt.normalize()

    g = m.groupby(["station", "date"])["aqi"]
    sd = pd.DataFrame({"aqi": g.mean(), "n": g.count()}).reset_index()
    sd = sd[sd["n"] >= MIN_HOURS_PER_DAY]

    cd = sd.groupby("date").agg(
        aqi_city=("aqi", "mean"), aqi_worst=("aqi", "max"), n_stations=("station", "nunique")
    ).reset_index()
    cd["week_start"] = week_start(pd.to_datetime(cd["date"]))
    cd["severe_day"] = (cd["aqi_city"] > 300).astype(float)

    wk = cd.groupby("week_start").agg(
        aqi_mean=("aqi_city", "mean"),
        aqi_max=("aqi_city", "max"),
        aqi_worst_station=("aqi_worst", "max"),
        pct_severe_days=("severe_day", "mean"),
        n_days=("date", "nunique"),
        mean_stations=("n_stations", "mean"),
    )
    wk = wk[wk["n_days"] >= MIN_DAYS_PER_WEEK].drop(columns="n_days")

    weather_cols = [c for c in m.columns if c.endswith("_city")]
    if weather_cols:
        m["week_start"] = week_start(m["timestamp"])
        w = m.groupby("week_start")[weather_cols].mean()
        wk = wk.join(w, how="left")
    return wk.reset_index()


def deseasonalise(df: pd.DataFrame, col: str) -> pd.Series:
    woy = df["week_start"].dt.isocalendar().week.astype(int)
    return df[col] - df.groupby(woy)[col].transform("mean")


def lag_table(df: pd.DataFrame, target: str, feature: str, max_lag: int) -> pd.DataFrame:
    rows = []
    for lag in range(0, max_lag + 1):
        x = df[feature].shift(lag)
        ok = x.notna() & df[target].notna()
        raw = np.corrcoef(x[ok], df.loc[ok, target])[0, 1] if ok.sum() > 10 else np.nan
        xd, yd = df["_x_ds"].shift(lag), df["_y_ds"]
        ok2 = xd.notna() & yd.notna()
        ds = np.corrcoef(xd[ok2], yd[ok2])[0, 1] if ok2.sum() > 10 else np.nan
        rows.append({"lag_weeks": lag, "raw_corr": raw, "deseasonalised_corr": ds, "n": int(ok.sum())})
    return pd.DataFrame(rows)


def main():
    master_csv, trends_csv, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(master_csv, parse_dates=["timestamp"])
    trends = pd.read_csv(trends_csv, parse_dates=["week_start"])

    cov = master.assign(year=master["timestamp"].dt.year).groupby(["station", "year"])["aqi"].apply(
        lambda s: round(100 * s.notna().mean(), 1)
    ).unstack("year")
    cov.to_csv(out_dir / "station_coverage.csv")
    print("Valid-AQI coverage (% of hours):\n", cov.to_string(), "\n")

    wk = build_city_weekly(master)
    df = wk.merge(trends, on="week_start", how="inner").sort_values("week_start").reset_index(drop=True)
    df.to_csv(out_dir / "weekly_dataset.csv", index=False)
    print(f"Weekly dataset: {len(df)} weeks, {df['week_start'].min().date()} to {df['week_start'].max().date()}")
    print("Missing share per column:\n", df.isna().mean().round(3).to_string(), "\n")

    target = "resp_interest_index"
    df["_y_ds"] = deseasonalise(df, target)
    df["_x_ds"] = deseasonalise(df, "aqi_mean")
    tab = lag_table(df, target, "aqi_mean", MAX_LAG)
    tab.to_csv(out_dir / "lag_correlation.csv", index=False)
    print("Correlation between AQI (lagged) and respiratory search index:\n", tab.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tab))
    ax.bar(x - 0.2, tab["raw_corr"], width=0.4, label="raw", color="#1F7A8C")
    ax.bar(x + 0.2, tab["deseasonalised_corr"], width=0.4, label="deseasonalised", color="#F2A541")
    ax.set_xticks(x)
    ax.set_xlabel("AQI leads search interest by (weeks)")
    ax.set_ylabel("Pearson correlation")
    ax.axhline(0, color="#5E6E7A", lw=0.8)
    ax.legend(frameon=False)
    ax.set_title("Does AQI predict respiratory searches beyond seasonality?")
    fig.tight_layout()
    fig.savefig(out_dir / "lag_correlation.png", dpi=150)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(df["week_start"], df["aqi_mean"], color="#C8473B", lw=1, label="City AQI (weekly mean)")
    ax.set_ylabel("AQI", color="#C8473B")
    ax2 = ax.twinx()
    ax2.plot(df["week_start"], df[target], color="#1F7A8C", lw=1, label="Respiratory search index")
    ax2.set_ylabel("Search index", color="#1F7A8C")
    ax.set_title("Weekly AQI and respiratory search interest, Delhi")
    fig.tight_layout()
    fig.savefig(out_dir / "aqi_vs_searches.png", dpi=150)

    best = tab.loc[tab["deseasonalised_corr"].abs().idxmax()]
    print(f"\nStrongest deseasonalised link: lag {int(best.lag_weeks)} weeks, r = {best.deseasonalised_corr:.2f}")
    print("Rule of thumb: |r| below ~0.15 after deseasonalising means AQI adds little beyond the calendar.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main()
