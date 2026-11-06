import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PM25_X = [0, 30, 60, 90, 120, 250, 380]
PM25_Y = [0, 50, 100, 200, 300, 400, 500]
SLOPE = 0.709  
REF_PM25 = 60.0
COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31")
SEED = 42


def aqi_to_pm25(aqi):
    return np.interp(np.clip(aqi, 0, 500), PM25_Y, PM25_X)


def excess_admissions(aqi, slope=SLOPE):
    return np.maximum(aqi_to_pm25(aqi) - REF_PM25, 0) * slope


def tier(x):
    return pd.cut(x, [-1, 0.001, 20, 60, 120, 1e9], labels=["none", "low", "moderate", "high", "very high"])


def part1(pred_csv, out):
    P = pd.read_csv(pred_csv, parse_dates=["date", "target_date"])
    P = P.dropna(subset=["actual", "xgb"])
    rows = []
    for (block, h), g in P.groupby(["block", "horizon"]):
        for mult, lab in ((0.5, "50% slope"), (1.0, "central"), (1.5, "150% slope")):
            a = excess_admissions(g["actual"].to_numpy(), SLOPE * mult)
            f = excess_admissions(g["xgb"].to_numpy(), SLOPE * mult)
            rows.append(dict(block=block, horizon=h, slope=lab, n=len(g), mean_actual_excess_per_week=round(a.mean(), 1),
                             mean_forecast_excess_per_week=round(f.mean(), 1), MAE_excess=round(np.abs(f - a).mean(), 1),
                             tier_exact=round((tier(a) == tier(f)).mean(), 3),
                             tier_within_one=round((np.abs(tier(a).codes - tier(f).codes) <= 1).mean(), 3)))
    R = pd.DataFrame(rows)
    R.to_csv(out / "burden_translation.csv", index=False)
    print("1. Burden translation: forecast vs actual AQI converted to excess weekly admissions (six-hospital scale)")
    print(R[R.slope == "central"].drop(columns="slope").to_string(index=False))
    print("   Range across slopes (mean actual excess per week):")
    print(R.pivot_table(index=["block", "horizon"], columns="slope", values="mean_actual_excess_per_week").to_string())

    # city-level daily example table for the dashboard
    g = P[(P.block == P.block.iloc[0]) & (P.horizon == P.horizon.min())].copy()
    g["pm25_est"] = aqi_to_pm25(g["xgb"]).round(0)
    g["excess_admissions_wk"] = excess_admissions(g["xgb"]).round(1)
    g["risk_tier"] = tier(g["excess_admissions_wk"])
    g.to_csv(out / "station_day_risk_example.csv", index=False)


def block_boot_r(x, y, block=4, n_boot=2000):
    rng = np.random.default_rng(SEED)
    n = len(x)
    nb = int(np.ceil(n / block))
    rs = []
    for _ in range(n_boot):
        s = rng.integers(0, max(n - block + 1, 1), size=nb)
        idx = np.concatenate([np.arange(i, i + block) for i in s])[:n]
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        rs.append(np.corrcoef(x[idx], y[idx])[0, 1])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def part2(weekly_csv, out):
    df = pd.read_csv(weekly_csv, parse_dates=["week_start"]).sort_values("week_start").set_index("week_start")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="7D")).rename_axis("week_start")
    covid = (df.index >= COVID_START) & (df.index <= COVID_END)
    woy = pd.Series(df.index.isocalendar().week.astype(int).to_numpy(), index=df.index)
    fit = (~covid) & (df.index < pd.Timestamp("2024-01-01"))

    def deseason(s):
        clim = s[fit].groupby(woy[fit]).mean()
        return s - woy.map(clim)

    x = deseason(df["aqi_worst_station"].rolling(2, min_periods=1).mean()).shift(2)
    y = deseason(df["asthma"])
    winter = pd.Series(df.index.month, index=df.index).isin([10, 11, 12, 1])

    print("\n2. Pre-committed hypothesis: asthma search interest vs 2-week-mean worst-station AQI lagged 2 weeks, Oct-Jan")
    res = []
    for label, m in (("discovery weeks (2017-2023, excl. COVID)", fit & winter), ("HELD-OUT winters (2024-25)", (df.index >= pd.Timestamp("2024-01-01")) & winter)):
        d = pd.concat([x[m], y[m]], axis=1).dropna()
        if len(d) < 15:
            print(f"   {label}: only {len(d)} weeks, cannot test")
            continue
        r = float(d.iloc[:, 0].corr(d.iloc[:, 1]))
        lo, hi = block_boot_r(d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy())
        res.append(dict(subset=label, n_weeks=len(d), r=round(r, 3), ci_low=round(lo, 3), ci_high=round(hi, 3)))
    R = pd.DataFrame(res)
    print(R.to_string(index=False))
    R.to_csv(out / "heldout_hypothesis.csv", index=False)
    if len(R) == 2:
        h = R.iloc[1]
        verdict = "SUPPORTED (positive, interval above 0)" if h.ci_low > 0 else ("NOT supported (interval includes 0 or negative)" if h.ci_high >= 0 else "CONTRADICTED (interval below 0)")
        print(f"   Verdict on held-out winters: {verdict}")
        print("   Report this result as-is, whichever way it falls. n is small: about 2 winters of weekly data.")

        d = pd.concat([x, y], axis=1).dropna()
        d = d[winter.reindex(d.index)]
        fig, ax = plt.subplots(figsize=(4.8, 4))
        old, new = d[d.index < pd.Timestamp("2024-01-01")], d[d.index >= pd.Timestamp("2024-01-01")]
        ax.scatter(old.iloc[:, 0], old.iloc[:, 1], s=10, alpha=0.5, label="2017-23 (discovery)", color="#8FB8C4")
        ax.scatter(new.iloc[:, 0], new.iloc[:, 1], s=16, label="2024-25 (held out)", color="#C8473B")
        ax.set_xlabel("worst-station AQI, 2-wk mean, lag 2 (deseasonalised)")
        ax.set_ylabel("asthma search interest (deseasonalised)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "heldout_scatter.png", dpi=150)


def main():
    pred_csv, weekly_csv, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    part1(pred_csv, out)
    part2(weekly_csv, out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main()
