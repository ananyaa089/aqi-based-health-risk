import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "resp_interest_index"
AQI_FEATS = ["aqi_mean", "aqi_max", "aqi_worst_station", "pct_severe_days"]
WEATHER_FEATS = ["temp_c_city", "humidity_city", "windspeed_ms_city", "pressure_hpa_city", "rain_mm_city"]
SEED = 42


def make_features(df: pd.DataFrame, horizon: int):
    d = df.sort_values("week_start").set_index("week_start")
    full = pd.date_range(d.index.min(), d.index.max(), freq="7D")
    d = d.reindex(full).rename_axis("week_start")  

    groups = {"ar": [], "aqi": [], "weather": []}
    X = pd.DataFrame(index=d.index)

    for lag in range(0, 4):
        X[f"y_lag{lag}"] = d[TARGET].shift(lag)
        groups["ar"].append(f"y_lag{lag}")
    X["y_roll4"] = d[TARGET].rolling(4, min_periods=3).mean()
    woy = d.index.isocalendar().week.astype(int).to_numpy()
    X["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    X["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    groups["ar"] += ["y_roll4", "woy_sin", "woy_cos"]

    for f in AQI_FEATS:
        if f in d.columns:
            for lag in range(0, 3):
                X[f"{f}_lag{lag}"] = d[f].shift(lag)
                groups["aqi"].append(f"{f}_lag{lag}")
    if "aqi_mean" in d.columns:
        X["aqi_roll4"] = d["aqi_mean"].rolling(4, min_periods=3).mean()
        groups["aqi"].append("aqi_roll4")

    for f in WEATHER_FEATS:
        if f in d.columns and d[f].notna().mean() > 0.5:
            X[f"{f}_lag0"] = d[f]
            groups["weather"].append(f"{f}_lag0")

    y = d[TARGET].shift(-horizon)
    baselines = pd.DataFrame(
        {
            "persistence": d[TARGET],
            "seasonal_naive": d[TARGET].shift(52 - horizon),  
        },
        index=d.index,
    )
    return X, y, baselines, groups


def metrics(y_true, y_pred):
    err = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def fit_xgb(Xtr, ytr, Xva, yva):
    grid = [
        dict(max_depth=d, n_estimators=n, learning_rate=0.05, min_child_weight=3)
        for d in (2, 3)
        for n in (100, 300)
    ]
    best, best_mae = None, np.inf
    for p in grid:
        m = xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=1, **p)
        m.fit(Xtr, ytr)
        mae = np.mean(np.abs(m.predict(Xva) - yva))
        if mae < best_mae:
            best, best_mae = p, mae
    return best, best_mae


def block_bootstrap_ci(diff, block=4, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff)
    n = len(diff)
    n_blocks = int(np.ceil(n / block))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
        sample = np.concatenate([diff[s : s + block] for s in starts])[:n]
        means.append(sample.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weekly_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--val-start", default="2023-01-01")
    ap.add_argument("--test-start", default="2024-01-01")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.weekly_csv, parse_dates=["week_start"])
    X, y, base, groups = make_features(df, a.horizon)

    val_start, test_start = pd.Timestamp(a.val_start), pd.Timestamp(a.test_start)
    target_week = X.index + pd.Timedelta(weeks=a.horizon)
    usable = y.notna() & X[groups["ar"]].notna().all(axis=1)
    tr = usable & (target_week < val_start)
    va = usable & (X.index >= val_start) & (X.index < test_start)
    te = usable & (X.index >= test_start)
    print(f"horizon={a.horizon}w  train={tr.sum()}  val={va.sum()}  test={te.sum()} weeks")
    if min(tr.sum(), va.sum(), te.sum()) < 20:
        print("WARNING: a split has fewer than 20 weeks; results will be unreliable. Adjust the dates.")

    sets = {
        "xgb_ar": groups["ar"],
        "xgb_ar_aqi": groups["ar"] + groups["aqi"],
    }
    if groups["weather"]:
        sets["xgb_ar_aqi_weather"] = groups["ar"] + groups["aqi"] + groups["weather"]

    preds = {"actual": y[te], "persistence": base.loc[te, "persistence"], "seasonal_naive": base.loc[te, "seasonal_naive"]}
    chosen = {}

    all_cols = sum(groups.values(), [])
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(X.loc[tr, all_cols], y[tr])
    preds["ridge_all"] = pd.Series(ridge.predict(X.loc[te, all_cols]), index=X.index[te])

    importances = None
    for name, cols in sets.items():
        p, vmae = fit_xgb(X.loc[tr, cols], y[tr], X.loc[va, cols], y[va])
        chosen[name] = {**p, "val_mae": round(vmae, 3)}
        final = xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=1, **p)
        trva = tr | va  
        final.fit(X.loc[trva, cols], y[trva])
        preds[name] = pd.Series(final.predict(X.loc[te, cols]), index=X.index[te])
        if name == "xgb_ar_aqi":
            importances = pd.Series(final.feature_importances_, index=cols).sort_values(ascending=False)

    P = pd.DataFrame(preds)
    P.index.name = "week_start"
    P.to_csv(out / "test_predictions.csv")

    winter = P.index.month.isin([10, 11, 12, 1])
    rows = []
    for name in P.columns.drop("actual"):
        ok = P[name].notna()
        for label, mask in (("all weeks", ok), ("Oct-Jan only", ok & winter)):
            if mask.sum() < 5:
                continue
            mae, rmse = metrics(P.loc[mask, "actual"], P.loc[mask, name])
            rows.append({"model": name, "subset": label, "n": int(mask.sum()), "MAE": round(mae, 3), "RMSE": round(rmse, 3)})
    R = pd.DataFrame(rows)
    ref = R[R.model == "persistence"].set_index("subset")["MAE"]
    R["skill_vs_persistence"] = [round(1 - r.MAE / ref[r.subset], 3) for r in R.itertuples()]
    R.to_csv(out / "results.csv", index=False)
    print("\nTest-set results (skill > 0 means better than persistence):\n", R.to_string(index=False))

    ok = P["xgb_ar"].notna() & P["xgb_ar_aqi"].notna()
    d_ar = (P.loc[ok, "xgb_ar"] - P.loc[ok, "actual"]).abs().to_numpy()
    d_aqi = (P.loc[ok, "xgb_ar_aqi"] - P.loc[ok, "actual"]).abs().to_numpy()
    diff = d_ar - d_aqi  
    lo, hi = block_bootstrap_ci(diff)
    verdict = "AQI features help" if lo > 0 else ("AQI features hurt" if hi < 0 else "no detectable difference")
    print(f"\nMAE(AR only) - MAE(AR + AQI) = {diff.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]  -> {verdict}")

    pd.Series({"horizon_weeks": a.horizon, "chosen_params": chosen, "aqi_effect_mean": diff.mean(), "aqi_effect_ci": [lo, hi], "verdict": verdict}).to_json(out / "summary.json", indent=2)
    if importances is not None:
        importances.to_csv(out / "feature_importance_xgb_ar_aqi.csv", header=["importance"])
        print("\nTop features (xgb_ar_aqi):\n", importances.head(8).round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(P.index, P["actual"], color="#1B2A33", lw=1.4, label="actual")
    ax.plot(P.index, P["persistence"], color="#8FB8C4", lw=1, label="persistence")
    ax.plot(P.index, P["xgb_ar_aqi"], color="#C8473B", lw=1.2, label="XGBoost + AQI")
    ax.set_title(f"Test period: {a.horizon}-week-ahead respiratory search index")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "test_forecast.png", dpi=150)


if __name__ == "__main__":
    main()
