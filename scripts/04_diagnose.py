import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31")
OFFICIAL_DEC = {2021: 336, 2022: 319, 2023: 348, 2024: 238}
TERMS = ["cough", "asthma", "breathlessness", "inhaler", "bronchitis"]


def corr(a: pd.Series, b: pd.Series, method="pearson"):
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 30:
        return np.nan, len(d)
    return float(d.iloc[:, 0].corr(d.iloc[:, 1], method=method)), len(d)


def deseason(df: pd.DataFrame, col: str, clim_mask: pd.Series) -> pd.Series:
    woy = df["week_start"].dt.isocalendar().week.astype(int)
    clim = df.loc[clim_mask].groupby(woy[clim_mask])[col].mean()
    return df[col] - woy.map(clim)


def monthly_city_aqi(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["date"] = m["timestamp"].dt.normalize()
    g = m.groupby(["station", "date"])["aqi"]
    sd = pd.DataFrame({"aqi": g.mean(), "n": g.count()}).reset_index()
    sd = sd[sd["n"] >= 12]
    cd = sd.groupby("date")["aqi"].mean().reset_index()
    cd["year"], cd["month"] = cd["date"].dt.year, cd["date"].dt.month
    t = cd.groupby(["year", "month"])["aqi"].mean().unstack("month").round(0)
    return t[[c for c in (10, 11, 12, 1) if c in t.columns]].rename(columns={10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan"})


def main():
    master_csv, weekly_csv, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(master_csv, parse_dates=["timestamp"])
    df = pd.read_csv(weekly_csv, parse_dates=["week_start"]).sort_values("week_start").reset_index(drop=True)
    covid = (df["week_start"] >= COVID_START) & (df["week_start"] <= COVID_END)

    t = monthly_city_aqi(master)
    t["Dec official (press)"] = [OFFICIAL_DEC.get(y, np.nan) for y in t.index]
    print("A. Mean city AQI by month (ours = average of your stations) vs official December:\n", t.to_string(), "\n")
    t.to_csv(out_dir / "A_monthly_aqi.csv")

    rows = []
    for col in ["resp_interest_index"] + [c for c in TERMS if c in df.columns]:
        y = deseason(df, col, ~covid)
        x = deseason(df, "aqi_mean", ~covid)
        for lag in (0, 1, 2):
            r_all, n_all = corr(x.shift(lag), y)
            r_nc, n_nc = corr(x.shift(lag)[~covid], y[~covid])
            rows.append({"term": col, "lag_weeks": lag, "r_all_weeks": r_all, "r_excl_covid": r_nc, "n_excl_covid": n_nc})
    B = pd.DataFrame(rows).round(3)
    print("B. Deseasonalised correlation of AQI (leading) with each search term:\n", B.to_string(index=False), "\n")
    B.to_csv(out_dir / "B_by_term.csv", index=False)

    y = deseason(df, "resp_interest_index", ~covid)
    x = deseason(df, "aqi_mean", ~covid)
    winter = df["week_start"].dt.month.isin([10, 11, 12, 1])
    variants = {
        "all weeks": pd.Series(True, index=df.index),
        "excluding Mar-2020..Dec-2021": ~covid,
        "pre-COVID (to Feb 2020)": df["week_start"] < COVID_START,
        "2022 onward": df["week_start"] > COVID_END,
        "Oct-Jan only, excl. COVID": winter & ~covid,
    }
    rows = []
    for name, mask in variants.items():
        for lag in (0, 1):
            xs = x.shift(lag)
            rp, n = corr(xs[mask], y[mask])
            rs, _ = corr(xs[mask], y[mask], "spearman")
            rows.append({"subset": name, "lag_weeks": lag, "pearson": rp, "spearman": rs, "n_weeks": n})
    C = pd.DataFrame(rows).round(3)
    print("C. Does the link change once COVID weeks are handled?\n", C.to_string(index=False), "\n")
    C.to_csv(out_dir / "C_subperiods.csv", index=False)

    dx, dy = df["aqi_mean"].diff(), df["resp_interest_index"].diff()
    rows = []
    for lag in (0, 1, 2):
        r, n = corr(dx.shift(lag)[~covid], dy[~covid])
        rows.append({"lag_weeks": lag, "corr_of_weekly_changes": r, "n_weeks": n})
    D = pd.DataFrame(rows).round(3)
    print("D. Week-to-week changes (excluding COVID):\n", D.to_string(index=False), "\n")
    D.to_csv(out_dir / "D_changes.csv", index=False)

    rows = []
    for f in ["aqi_mean", "aqi_max", "aqi_worst_station", "pct_severe_days"]:
        if f in df.columns:
            xf = deseason(df, f, ~covid)
            for lag in (0, 1):
                r, n = corr(xf.shift(lag)[~covid], y[~covid])
                rows.append({"aqi_feature": f, "lag_weeks": lag, "deseasonalised_r": r, "n_weeks": n})
    E = pd.DataFrame(rows).round(3)
    print("E. Other AQI features vs the search index (excluding COVID):\n", E.to_string(index=False), "\n")
    E.to_csv(out_dir / "E_features.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    for c in [c for c in TERMS if c in df.columns]:
        ax.plot(df["week_start"], df[c], lw=0.8, label=c)
    ax.axvspan(COVID_START, COVID_END, color="#F2A541", alpha=0.2, label="COVID window")
    ax.set_title("Weekly Google Trends interest by term (Delhi)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "terms_over_time.png", dpi=150)
    print(f"Saved tables and plot to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main()
