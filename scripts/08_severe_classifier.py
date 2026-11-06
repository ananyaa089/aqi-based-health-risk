import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

spec = importlib.util.spec_from_file_location("f07", Path(__file__).resolve().parent / "07_forecast_aqi.py")
f07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f07)

SEED, WINDOW = 42, 3
THRESHOLDS = (300, 400)
BLOCKS = [
    ("A_reported_2023", "2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("B_cross_era_2024_25", "2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
]


def build(master):
    panel, wcols = f07.daily_panel(master)
    p, feats = f07.add_features(panel, wcols, horizons=[])
    g = p.groupby("station")["aqi"]
    fut = pd.concat([g.shift(-k) for k in range(1, WINDOW + 1)], axis=1)
    p["fut_max"] = fut.max(axis=1)
    p["fut_valid"] = fut.notna().sum(axis=1)
    p["recent_max3"] = pd.concat([g.shift(k) for k in range(0, 3)], axis=1).max(axis=1)
    fw = []
    for c in wcols:
        wc = p.groupby("station")[c]
        stack = pd.concat([wc.shift(-k) for k in range(1, WINDOW + 1)], axis=1)
        p[f"fut_{c}_mean"] = stack.mean(axis=1)
        fw.append(f"fut_{c}_mean")
        if c in ("rain_mm_city", "windspeed_ms_city"):
            p[f"fut_{c}_max"] = stack.max(axis=1)
            fw.append(f"fut_{c}_max")
    return p, feats, fw


def pick_threshold(y, prob):
    best_t, best = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        pred = prob >= t
        tp = (pred & (y == 1)).sum()
        prec = tp / max(pred.sum(), 1)
        rec = tp / max((y == 1).sum(), 1)
        f2 = 5 * prec * rec / max(4 * prec + rec, 1e-9)
        if f2 > best:
            best, best_t = f2, t
    return float(best_t)


def pr(y, pred):
    tp = int((pred & (y == 1)).sum())
    return tp / max(int(pred.sum()), 1), tp / max(int((y == 1).sum()), 1)


def paired_ap_ci(y, p1, p2, dates, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    wk = (pd.DatetimeIndex(dates).normalize() - pd.Timestamp("2000-01-01")).days // 14
    groups = {k: np.where(wk == k)[0] for k in np.unique(wk)}
    keys = np.array(list(groups))
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys))])
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], p1[idx]) - average_precision_score(y[idx], p2[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def make_clf():
    return xgb.XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.05, min_child_weight=5, subsample=0.8,
                             colsample_bytree=0.8, random_state=SEED, n_jobs=2, eval_metric="logloss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master_csv")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p, feats, fw = build(pd.read_csv(a.master_csv, parse_dates=["timestamp"]))
    print(f"{p['station'].nunique()} stations, {len(feats)} base features, {len(fw)} future-weather features\n")

    rows, curves = [], {}
    for T in THRESHOLDS:
        p[f"y{T}"] = (p["fut_max"] >= T).astype(int)
        for name, tr_end, va_s, va_e, te_s, te_e in BLOCKS:
            ok = p["fut_valid"].ge(2) & p["aqi_lag0"].notna() & p[feats].notna().sum(axis=1).ge(len(feats) // 2)
            tr = ok & (p["date"] + pd.Timedelta(days=WINDOW) <= pd.Timestamp(tr_end))
            va = ok & p["date"].between(pd.Timestamp(va_s), pd.Timestamp(va_e))
            te = ok & p["date"].between(pd.Timestamp(te_s), pd.Timestamp(te_e))
            ytr, yva, yte = p.loc[tr, f"y{T}"], p.loc[va, f"y{T}"], p.loc[te, f"y{T}"].to_numpy()
            if min(ytr.sum(), yva.sum(), yte.sum()) < 20:
                print(f"[{name} T={T}] too few positive days (train {ytr.sum()}, val {yva.sum()}, test {yte.sum()}); skipped")
                continue
            base_rate = yte.mean()
            res = {}
            for variant, cols in (("no_future_weather", feats), ("with_future_weather (upper bound)", feats + fw)):
                m = make_clf().fit(p.loc[tr, cols], ytr)
                thr = pick_threshold(yva.to_numpy(), m.predict_proba(p.loc[va, cols])[:, 1])
                m2 = make_clf().fit(p.loc[tr | va, cols], p.loc[tr | va, f"y{T}"])
                prob = m2.predict_proba(p.loc[te, cols])[:, 1]
                prec, rec = pr(yte, prob >= thr)
                res[variant] = prob
                rows.append(dict(block=name, severity=T, model=variant, test_days=len(yte), base_rate=round(base_rate, 3),
                                 PR_AUC=round(average_precision_score(yte, prob), 3), ROC_AUC=round(roc_auc_score(yte, prob), 3),
                                 threshold=round(thr, 2), precision=round(prec, 2), recall=round(rec, 2)))
                curves[(name, T, variant)] = (yte, prob)
            for bname, vals in (("alert if today >= T", p.loc[te, "aqi_lag0"]), ("alert if 3-day max >= T", p.loc[te, "recent_max3"])):
                pred = (vals >= T).to_numpy()
                prec, rec = pr(yte, pred)
                rows.append(dict(block=name, severity=T, model=bname, test_days=len(yte), base_rate=round(base_rate, 3),
                                 PR_AUC=round(average_precision_score(yte, vals.fillna(0).to_numpy()), 3),
                                 ROC_AUC=round(roc_auc_score(yte, vals.fillna(0).to_numpy()), 3),
                                 threshold=T, precision=round(prec, 2), recall=round(rec, 2)))
            lo, hi = paired_ap_ci(yte, res["with_future_weather (upper bound)"], res["no_future_weather"], p.loc[te, "date"].to_numpy())
            print(f"[{name} T={T}] PR-AUC gain from perfect future weather: "
                  f"{average_precision_score(yte, res['with_future_weather (upper bound)']) - average_precision_score(yte, res['no_future_weather']):+.3f}"
                  f"  95% CI [{lo:+.3f}, {hi:+.3f}]")

    R = pd.DataFrame(rows)
    R.to_csv(out / "results.csv", index=False)
    print("\nSevere-episode alerts (positive = severe day within next 3 days):\n", R.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, T in zip(axes, THRESHOLDS):
        for (name, TT, variant), (y, prob) in curves.items():
            if TT != T or not name.startswith("B"):
                continue
            o = np.argsort(-prob)
            tp = np.cumsum(y[o])
            ax.plot(tp / max(y.sum(), 1), tp / np.arange(1, len(y) + 1), label=variant.split(" ")[0])
        ax.set_title(f"Precision-recall, AQI>={T} within 3 days, test 2024-25", fontsize=9)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=150)
    print(f"\nSaved results and PR curves to {out}")


if __name__ == "__main__":
    main()
