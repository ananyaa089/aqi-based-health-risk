import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PM25_X, PM25_Y = [0, 30, 60, 90, 120, 250, 380], [0, 50, 100, 200, 300, 400, 500]
PM10_X, PM10_Y = [0, 50, 100, 250, 350, 430, 510], [0, 50, 100, 200, 300, 400, 500]
SEED, BLOCK_DAYS, N_BOOT = 42, 14, 1000
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def cams_aqi(d):
    return np.interp(d["cams_pm25"], PM25_X, PM25_Y)


def block_ci(stat, dates):
    rng = np.random.default_rng(SEED)
    k = (pd.DatetimeIndex(dates).normalize() - pd.Timestamp("2000-01-01")).days // BLOCK_DAYS
    groups = {b: np.where(k == b)[0] for b in np.unique(k)}
    keys = np.array(list(groups))
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[b] for b in rng.choice(keys, size=len(keys))])
        vals.append(stat(idx))
    return np.percentile(vals, [2.5, 97.5])


def main():
    P = pd.read_csv(sys.argv[1], parse_dates=["date", "target_date"])
    C = pd.read_csv(sys.argv[2], parse_dates=["date"])
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    C["cams_aqi"] = cams_aqi(C)
    print(f"CAMS daily rows: {len(C)}  ({C['date'].min().date()} to {C['date'].max().date()})")

    M = P.merge(C[["station", "date", "cams_aqi"]].rename(columns={"date": "target_date"}), on=["station", "target_date"], how="inner")
    M = M.dropna(subset=["actual", "xgb", "cams_aqi"])
    A, B = M[M.block.str.startswith("A")], M[M.block.str.startswith("B")]
    print(f"Rows with CAMS available: block A (2023) {len(A)}, block B (2024-25) {len(B)}\n")
    if len(A) < 300 or len(B) < 300:
        print("Not enough overlap between CAMS and the forecast blocks. Check 12a output (CAMS may start after 2022-07).")
        return

    rows = []
    for h in sorted(M["horizon"].unique()):
        a, b = A[A.horizon == h].copy(), B[B.horizon == h].copy()
        coef = {}
        for st, g in a.groupby("station"):
            coef[st] = np.polyfit(g["cams_aqi"], g["actual"], 1) if len(g) >= 60 else None
        pooled = np.polyfit(a["cams_aqi"], a["actual"], 1)
        b["cams_cal"] = [np.polyval(coef.get(s) if coef.get(s) is not None else pooled, v) for s, v in zip(b["station"], b["cams_aqi"])]
        Xa = np.column_stack([np.ones(len(a)), a["xgb"], a["cams_aqi"]])
        beta = np.linalg.lstsq(Xa, a["actual"].to_numpy(), rcond=None)[0]
        b["stack"] = np.column_stack([np.ones(len(b)), b["xgb"], b["cams_aqi"]]) @ beta
        for label, sub in (("2024", b[b["target_date"].dt.year == 2024]), ("2025", b[b["target_date"].dt.year == 2025]), ("2024-25", b)):
            if len(sub) < 300:
                continue
            y = sub["actual"].to_numpy()
            e = {m: np.abs(sub[m].to_numpy() - y) for m in ("xgb", "persistence", "cams_cal", "stack")}
            lo, hi = block_ci(lambda i: 1 - e["stack"][i].mean() / e["xgb"][i].mean(), sub["target_date"].to_numpy())
            r = float(np.corrcoef(sub["cams_aqi"], y)[0, 1])
            rows.append(dict(horizon=h, test_period=label, n=len(sub), MAE_persistence=round(e["persistence"].mean(), 1), MAE_our_model=round(e["xgb"].mean(), 1),
                             MAE_cams_calibrated_alone=round(e["cams_cal"].mean(), 1), MAE_model_plus_cams=round(e["stack"].mean(), 1),
                             gain_from_adding_cams=round(1 - e["stack"].mean() / e["xgb"].mean(), 3), gain_ci=f"[{lo:+.3f}, {hi:+.3f}]",
                             corr_cams_vs_actual=round(r, 3), stack_weights=f"xgb {beta[1]:.2f}, cams {beta[2]:.2f}"))
        y = b["actual"].to_numpy()
        r = float(np.corrcoef(b["cams_aqi"], y)[0, 1])
        if h == min(M["horizon"].unique()):
            fig, ax = plt.subplots(figsize=(4.4, 4))
            ax.scatter(b["cams_aqi"], y, s=3, alpha=0.25, color="#4C7C8A")
            ax.plot([0, 500], [0, 500], color="#bbb", lw=0.8, ls="--")
            ax.set_xlabel("CAMS-derived AQI (uncalibrated)")
            ax.set_ylabel("station AQI")
            ax.set_title(f"CAMS vs station AQI, 2024-25 (r = {r:.2f})", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / "fig7_cams_vs_station.png")
            plt.close(fig)
    R = pd.DataFrame(rows)
    R.to_csv(out / "cams_benchmark.csv", index=False)
    print(R.to_string(index=False))
    print("\nHow to read: 'CAMS calibrated alone' is a same-day model estimate (oracle for the target day), so beating it at 1-3 days "
          "is not expected. 'gain_from_adding_cams' > 0 with a CI above 0 means real CAMS forecast information would be worth integrating "
          "(upper bound, since this CAMS series is not a lead-time forecast).")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main()
