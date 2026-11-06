import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

SEED, N_BOOT, BLOCK_DAYS = 42, 1000, 14
COL = {"xgb": "#C8473B", "persistence": "#4C7C8A", "climatology": "#9A9A9A"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def date_blocks(dates):
    k = (pd.DatetimeIndex(dates).normalize() - pd.Timestamp("2000-01-01")).days // BLOCK_DAYS
    return {b: np.where(k == b)[0] for b in np.unique(k)}


def boot(stat, dates, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    groups = date_blocks(dates)
    keys = np.array(list(groups))
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys))])
        v = stat(idx)
        if v is not None and np.isfinite(v):
            out.append(v)
    return np.percentile(out, [2.5, 97.5])


def forecast_cis(P, out):
    rows = []
    for (block, h), g in P.groupby(["block", "horizon"]):
        g = g.dropna(subset=["actual", "xgb", "persistence", "climatology"]).reset_index(drop=True)
        a = g["actual"].to_numpy()
        err = {m: np.abs(g[m].to_numpy() - a) for m in ("xgb", "persistence", "climatology")}
        for m in err:
            lo, hi = boot(lambda i, m=m: err[m][i].mean(), g["target_date"])
            rows.append(dict(block=block, horizon=h, quantity=f"MAE {m}", value=round(err[m].mean(), 1), ci_low=round(lo, 1), ci_high=round(hi, 1)))
        for ref in ("persistence", "climatology"):
            lo, hi = boot(lambda i, ref=ref: 1 - err["xgb"][i].mean() / err[ref][i].mean(), g["target_date"])
            rows.append(dict(block=block, horizon=h, quantity=f"skill vs {ref}", value=round(1 - err["xgb"].mean() / err[ref].mean(), 3),
                             ci_low=round(lo, 3), ci_high=round(hi, 3)))
    R = pd.DataFrame(rows)
    R.to_csv(out / "forecast_error_cis.csv", index=False)
    print("1. Forecast error with 95% block-bootstrap CIs (skill > 0 means better than the reference):")
    print(R[R.quantity.str.startswith("skill")].to_string(index=False))
    return R


def fig_mae(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    for ax, (blk, g) in zip(axes, R[R.quantity.str.startswith("MAE")].groupby("block")):
        for m in ("persistence", "climatology", "xgb"):
            s = g[g.quantity == f"MAE {m}"].sort_values("horizon")
            ax.errorbar(s["horizon"], s["value"], yerr=[s["value"] - s["ci_low"], s["ci_high"] - s["value"]], marker="o", capsize=3,
                        color=COL[m], label={"xgb": "XGBoost"}.get(m, m))
        ax.set_title({"A_reported_2023": "Test 2023 (reported AQI)", "B_cross_era_2024_25": "Test 2024-25 (computed AQI)"}.get(blk, blk))
        ax.set_xlabel("forecast horizon (days)")
        ax.set_xticks(sorted(g["horizon"].unique()))
    axes[0].set_ylabel("MAE (AQI points), 95% CI")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig1_mae_by_horizon.png")
    plt.close(fig)


def error_analysis(P, out):
    B = P[P.block.str.startswith("B")].dropna(subset=["actual", "xgb"]).copy()
    B["err"] = B["xgb"] - B["actual"]
    B["abs"] = B["err"].abs()
    bins = [0, 100, 200, 300, 400, 600]
    B["level"] = pd.cut(B["actual"], bins, labels=["<=100", "101-200", "201-300", "301-400", "400+"])
    t_level = B.groupby(["horizon", "level"], observed=True).agg(n=("err", "size"), mean_bias=("err", "mean"), MAE=("abs", "mean")).round(1)
    t_station = B.groupby(["horizon", "station"]).agg(n=("err", "size"), mean_bias=("err", "mean"), MAE=("abs", "mean")).round(1)
    B["month"] = B["target_date"].dt.month
    t_month = B.groupby(["horizon", "month"]).agg(n=("err", "size"), MAE=("abs", "mean")).round(1)
    t_level.to_csv(out / "error_by_aqi_level.csv")
    t_station.to_csv(out / "error_by_station.csv")
    t_month.to_csv(out / "error_by_month.csv")
    print("\n2. Error by actual AQI level, test 2024-25 (negative bias = model under-predicts):\n", t_level.to_string())
    print("\n   Error by station, horizon 1:\n", t_station.loc[B["horizon"].min()].to_string())

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    h = sorted(B["horizon"].unique())
    w = 0.8 / len(h)
    lv = t_level.reset_index()
    cats = list(lv["level"].cat.categories) if hasattr(lv["level"], "cat") else sorted(lv["level"].unique())
    for j, hh in enumerate(h):
        s = lv[lv.horizon == hh].set_index("level").reindex(cats)
        axes[0].bar(np.arange(len(cats)) + j * w, s["mean_bias"], w, label=f"+{hh} d", color=plt.cm.Reds(0.35 + 0.25 * j))
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xticks(np.arange(len(cats)) + 0.4 - w / 2)
    axes[0].set_xticklabels(cats)
    axes[0].set_xlabel("actual AQI")
    axes[0].set_ylabel("mean forecast error (forecast - actual)")
    axes[0].set_title("Bias by AQI level (test 2024-25)", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    s1 = t_station.loc[h[0]].sort_values("MAE")
    axes[1].barh(s1.index, s1["MAE"], color=COL["xgb"])
    axes[1].set_xlabel(f"MAE (AQI points), +{h[0]} day")
    axes[1].set_title("Error by station", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig2_error_analysis.png")
    plt.close(fig)


def alert_cis(Dd, out):
    d = Dd.copy()
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, T in zip(axes, (300, 400)):
        y = d[f"actual_sev{T}"]
        ok = y.notna()
        g = d[ok].reset_index(drop=True)
        yy, pm, pb = g[f"actual_sev{T}"].to_numpy(), g[f"p_sev{T}"].to_numpy(), g["aqi_today"].to_numpy()
        ap_m, ap_b = average_precision_score(yy, pm), average_precision_score(yy, pb)
        dates = g["date"]
        def stat_m(i): return average_precision_score(yy[i], pm[i]) if yy[i].sum() else None
        def stat_diff(i): return average_precision_score(yy[i], pm[i]) - average_precision_score(yy[i], pb[i]) if yy[i].sum() else None
        lo_m, hi_m = boot(stat_m, dates, 500)
        lo_d, hi_d = boot(stat_diff, dates, 500)
        al = g[f"alert{T}"].to_numpy() == 1
        prec = (al & (yy == 1)).sum() / max(al.sum(), 1)
        rec = (al & (yy == 1)).sum() / max((yy == 1).sum(), 1)
        rows.append(dict(severity=T, base_rate=round(yy.mean(), 3), PR_AUC_model=round(ap_m, 3), PR_AUC_ci=f"[{lo_m:.3f}, {hi_m:.3f}]",
                         PR_AUC_today_rule=round(ap_b, 3), gain=round(ap_m - ap_b, 3), gain_ci=f"[{lo_d:+.3f}, {hi_d:+.3f}]",
                         alert_precision=round(prec, 2), alert_recall=round(rec, 2), alert_days=int(al.sum())))
        for name, sc, c in (("model", pm, COL["xgb"]), ("today's AQI rule", pb, COL["persistence"])):
            o = np.argsort(-sc)
            tp = np.cumsum(yy[o])
            ax.plot(tp / max(yy.sum(), 1), tp / np.arange(1, len(yy) + 1), color=c, label=name)
        ax.axhline(yy.mean(), color="#bbb", lw=0.8, ls="--", label="base rate")
        ax.plot([rec], [prec], "o", color=COL["xgb"], mfc="white", label="chosen alert threshold")
        ax.set_title(f"AQI >= {T} within 3 days (test 2024-25)", fontsize=9)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig3_alert_pr_curves.png")
    plt.close(fig)
    R = pd.DataFrame(rows)
    R.to_csv(out / "alert_cis.csv", index=False)
    print("\n3. Severe-episode alerts, test 2024-25, 95% block-bootstrap CIs (gain = PR-AUC model minus 'today >= T' rule):\n", R.T.to_string(header=False))


def episode_fig(Dd, out):
    d = Dd.copy()
    d["date"] = pd.to_datetime(d["date"])
    city = d.groupby("date").agg(actual=("actual_h1", "mean"), forecast=("pred_h1", "mean"), today=("aqi_today", "mean"))
    city.index = city.index + pd.Timedelta(days=1)
    mo = city.groupby(city.index.to_period("M"))["actual"].mean()
    worst = mo.idxmax()
    w = city[(city.index >= worst.start_time - pd.Timedelta(days=15)) & (city.index <= worst.end_time + pd.Timedelta(days=15))]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(w.index, w["actual"], color="#222", lw=1.6, label="actual city-mean AQI")
    ax.plot(w.index, w["forecast"], color=COL["xgb"], lw=1.4, label="1-day forecast")
    ax.plot(w.index, w["today"], color=COL["persistence"], lw=1, ls="--", label="persistence")
    ax.axhline(400, color="#8a1c3c", lw=0.7, ls=":")
    ax.text(w.index[0], 405, "severe (400)", fontsize=7, color="#8a1c3c")
    ax.set_ylabel("AQI")
    ax.set_title(f"Worst replay month ({worst}): 1-day-ahead forecast vs actual", fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "fig4_worst_month_episode.png")
    plt.close(fig)
    print(f"\n4. Episode figure drawn for {worst}.")


def main():
    pred_csv, dash_csv, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(pred_csv, parse_dates=["date", "target_date"])
    R = forecast_cis(P, out)
    fig_mae(R, out)
    error_analysis(P, out)
    Dd = pd.read_csv(dash_csv, parse_dates=["date"])
    alert_cis(Dd, out)
    episode_fig(Dd, out)
    print(f"\nSaved tables and figures to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main()
