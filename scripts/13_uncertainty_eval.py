import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED, BLOCK_DAYS, N_BOOT = 42, 14, 1000
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
RED, TEAL = "#C8473B", "#4C7C8A"


def block_ci(stat, dates):
    rng = np.random.default_rng(SEED)
    k = (pd.DatetimeIndex(dates).normalize() - pd.Timestamp("2000-01-01")).days // BLOCK_DAYS
    groups = {b: np.where(k == b)[0] for b in np.unique(k)}
    keys = np.array(list(groups))
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[b] for b in rng.choice(keys, size=len(keys))])
        v = stat(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5])


def intervals(D, out):
    rows = []
    winter = D["date"].dt.month.isin([10, 11, 12, 1]).to_numpy()
    for h in (1, 2, 3):
        a, lo, hi = D[f"actual_h{h}"].to_numpy(), D[f"lo_h{h}"].to_numpy(), D[f"hi_h{h}"].to_numpy()
        ok = ~np.isnan(a)
        inside = ((a >= lo) & (a <= hi)).astype(float)
        for label, m in (("all days", ok), ("Oct-Jan", ok & winter)):
            idx_all = np.where(m)[0]
            dates = D["date"].to_numpy()[idx_all]
            lo_c, hi_c = block_ci(lambda i: inside[idx_all][i].mean(), dates)
            rows.append(dict(horizon=h, subset=label, n=int(m.sum()), coverage=round(inside[m].mean(), 3), cov_ci=f"[{lo_c:.3f}, {hi_c:.3f}]",
                             mean_width=round(float((hi - lo)[m].mean()), 1)))
    R = pd.DataFrame(rows)
    R.to_csv(out / "interval_coverage.csv", index=False)
    print("1. 80% prediction ranges, 2024-25 (nominal coverage 0.80):\n", R.to_string(index=False))
    return R


def calibration(D, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    rows = []
    for ax, T in zip(axes, (300, 400)):
        d = D.dropna(subset=[f"actual_sev{T}"])
        y, p = d[f"actual_sev{T}"].to_numpy(), d[f"p_sev{T}"].to_numpy()
        bins = pd.qcut(p, 10, duplicates="drop")
        t = pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b", observed=True).agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
        ece = float((t["n"] * (t["mean_pred"] - t["observed"]).abs()).sum() / t["n"].sum())
        brier, brier_base = float(np.mean((p - y) ** 2)), float(np.mean((y.mean() - y) ** 2))
        rows.append(dict(severity=T, brier=round(brier, 4), brier_if_base_rate=round(brier_base, 4), brier_skill=round(1 - brier / brier_base, 3), ECE=round(ece, 3)))
        ax.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=0.8)
        ax.plot(t["mean_pred"], t["observed"], marker="o", color=RED)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"Calibration, AQI >= {T} within 3 days", fontsize=9)
        print(f"\n2. Calibration AQI>={T}:\n", t.round(3).to_string())
    fig.tight_layout()
    fig.savefig(out / "fig5_calibration.png")
    plt.close(fig)
    R = pd.DataFrame(rows)
    R.to_csv(out / "calibration_summary.csv", index=False)
    print("\n   Summary (Brier skill > 0 beats always predicting the base rate; ECE near 0 is well calibrated):\n", R.to_string(index=False))


def tiers(D, out):
    rows = []
    for T in (300, 400):
        d = D.dropna(subset=[f"actual_sev{T}"])
        y = d[f"actual_sev{T}"].to_numpy()
        for name, col in (("standard alert", f"alert{T}"), ("high-confidence alert", f"alert_hi{T}")):
            a = d[col].to_numpy() == 1
            tp = int((a & (y == 1)).sum())
            rows.append(dict(severity=T, tier=name, share_of_station_days_alerted=round(a.mean(), 3), precision=round(tp / max(a.sum(), 1), 3),
                             recall=round(tp / max((y == 1).sum(), 1), 3), false_alarms=int((a & (y == 0)).sum()), missed=int((~a & (y == 1)).sum())))
    R = pd.DataFrame(rows)
    R.to_csv(out / "alert_tiers.csv", index=False)
    print("\n3. Alert tiers, 2024-25:\n", R.to_string(index=False))


def drivers(path, out):
    if not path.exists():
        return
    S = pd.read_csv(path)
    models = list(S["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        s = S[S.model == m].sort_values("mean_abs_contribution").tail(7)
        ax.barh(s["feature"], s["mean_abs_contribution"], color=TEAL)
        ax.set_title(m, fontsize=8)
        ax.set_xlabel("mean |contribution|", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig6_global_drivers.png")
    plt.close(fig)
    print("\n4. Top drivers per model:")
    for m in models:
        print("  ", m, "->", ", ".join(S[S.model == m].sort_values("mean_abs_contribution", ascending=False)["feature"].head(5)))


def main():
    dash, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(dash / "dashboard_data.csv", parse_dates=["date"])
    intervals(D, out)
    calibration(D, out)
    tiers(D, out)
    drivers(dash / "shap_global.csv", out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main()
