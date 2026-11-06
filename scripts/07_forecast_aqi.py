import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
WEATHER = ["temp_c_city", "humidity_city", "windspeed_ms_city", "pressure_hpa_city", "rain_mm_city"]
SEVERE = 400.0


def daily_panel(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    m["date"] = m["timestamp"].dt.normalize()
    g = m.groupby(["station", "date"])["aqi"]
    d = pd.DataFrame({"aqi": g.mean(), "n": g.count()}).reset_index()
    d.loc[d["n"] < 12, "aqi"] = np.nan
    wcols = [c for c in WEATHER if c in m.columns]
    if wcols:
        w = m.groupby("date")[wcols].mean().reset_index()
        d = d.merge(w, on="date", how="left")
    frames = []
    for st, s in d.groupby("station"):
        s = s.set_index("date").reindex(pd.date_range(s["date"].min(), s["date"].max(), freq="D"))
        s["station"] = st
        frames.append(s.rename_axis("date").reset_index())
    return pd.concat(frames, ignore_index=True), wcols


def add_features(p: pd.DataFrame, wcols, horizons):
    p = p.sort_values(["station", "date"]).reset_index(drop=True)
    g = p.groupby("station")["aqi"]
    for k in (0, 1, 2, 3, 7):
        p[f"aqi_lag{k}"] = g.shift(k)
    p["aqi_roll3"] = g.transform(lambda s: s.rolling(3, min_periods=2).mean())
    p["aqi_roll7"] = g.transform(lambda s: s.rolling(7, min_periods=4).mean())
    p["aqi_trend"] = p["aqi_lag0"] - p["aqi_lag3"]
    city = p.groupby("date")["aqi"].transform("mean")  
    p["city_aqi_lag0"] = city
    p["station_vs_city"] = p["aqi_lag0"] - city
    doy = p["date"].dt.dayofyear
    p["doy_sin"], p["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    p["station_id"] = p["station"].astype("category").cat.codes
    for c in wcols:
        p[f"{c}_lag0"] = p[c]
        p[f"{c}_chg1"] = p[c] - p.groupby("station")[c].shift(1)
    feats = [c for c in p.columns if c.startswith(("aqi_lag", "aqi_roll", "aqi_trend", "city_aqi", "station_vs", "doy_", "station_id"))]
    feats += [f"{c}_lag0" for c in wcols] + [f"{c}_chg1" for c in wcols]
    for h in horizons:
        p[f"y_h{h}"] = g.shift(-h)
    return p, feats


def climatology(train: pd.DataFrame):
    t = train.dropna(subset=["aqi"]).copy()
    t["doy"] = t["date"].dt.dayofyear
    tab = t.groupby(["station", "doy"])["aqi"].mean().unstack("doy").reindex(columns=range(1, 367))
    ext = pd.concat([tab.iloc[:, -7:], tab, tab.iloc[:, :7]], axis=1)
    sm = ext.T.rolling(15, center=True, min_periods=3).mean().T.iloc[:, 7:-7]
    sm.columns = tab.columns
    return sm


def clim_pred(sm, stations, dates):
    doy = pd.DatetimeIndex(dates).dayofyear
    out = []
    for s, d in zip(stations, doy):
        out.append(sm.at[s, d] if s in sm.index and d in sm.columns else np.nan)
    return np.array(out)


def fit_xgb(Xtr, ytr, Xva, yva):
    best, best_mae = None, np.inf
    for depth in (3, 5):
        for n in (200, 500):
            m = xgb.XGBRegressor(max_depth=depth, n_estimators=n, learning_rate=0.04, min_child_weight=5,
                                 subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=2)
            m.fit(Xtr, ytr)
            mae = np.mean(np.abs(m.predict(Xva) - yva))
            if mae < best_mae:
                best, best_mae = dict(max_depth=depth, n_estimators=n), mae
    return best, best_mae


def scores(y, yhat):
    ok = ~(np.isnan(y) | np.isnan(yhat))
    y, yhat = y[ok], yhat[ok]
    mae = float(np.mean(np.abs(yhat - y)))
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    tp = int(((yhat >= SEVERE) & (y >= SEVERE)).sum())
    prec = tp / max(int((yhat >= SEVERE).sum()), 1)
    rec = tp / max(int((y >= SEVERE).sum()), 1)
    return mae, rmse, prec, rec, int(ok.sum())


def run_block(name, p, feats, h, tr_end, va_start, va_end, te_start, te_end, wint_only_rows):
    ycol = f"y_h{h}"
    tdate = p["date"] + pd.Timedelta(days=h)  
    base_ok = p[ycol].notna() & p[feats].notna().sum(axis=1).ge(len(feats) // 2) & p["aqi_lag0"].notna()
    tr = base_ok & (tdate <= pd.Timestamp(tr_end))
    va = base_ok & (p["date"] >= pd.Timestamp(va_start)) & (p["date"] <= pd.Timestamp(va_end))
    te = base_ok & (p["date"] >= pd.Timestamp(te_start)) & (p["date"] <= pd.Timestamp(te_end))
    if min(tr.sum(), va.sum(), te.sum()) < 100:
        print(f"  [{name} h={h}] too few rows (train {tr.sum()}, val {va.sum()}, test {te.sum()}); skipped")
        return None, None
    params, vmae = fit_xgb(p.loc[tr, feats], p.loc[tr, ycol], p.loc[va, feats], p.loc[va, ycol])
    trva = tr | va
    final = xgb.XGBRegressor(learning_rate=0.04, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                             random_state=SEED, n_jobs=2, **params)
    final.fit(p.loc[trva, feats], p.loc[trva, ycol])

    T = p.loc[te, ["station", "date", ycol, "aqi_lag0"]].copy()
    T["xgb"] = final.predict(p.loc[te, feats])
    T["persistence"] = T["aqi_lag0"]
    sm = climatology(p.loc[trva & (p["date"] <= pd.Timestamp(va_end))])
    T["climatology"] = clim_pred(sm, T["station"].to_numpy(), T["date"] + pd.Timedelta(days=h))
    T["target_date"] = T["date"] + pd.Timedelta(days=h)
    T = T.rename(columns={ycol: "actual"})
    T["block"], T["horizon"] = name, h

    rows = []
    winter = T["target_date"].dt.month.isin([10, 11, 12, 1]).to_numpy()
    for model in ("persistence", "climatology", "xgb"):
        for label, mask in (("all days", np.ones(len(T), bool)), ("Oct-Jan", winter)):
            mae, rmse, prec, rec, n = scores(T.loc[mask, "actual"].to_numpy(), T.loc[mask, model].to_numpy())
            rows.append(dict(block=name, horizon=h, model=model, subset=label, n=n, MAE=round(mae, 1), RMSE=round(rmse, 1),
                             severe_precision=round(prec, 2), severe_recall=round(rec, 2)))
    R = pd.DataFrame(rows)
    pers = R[R.model == "persistence"].set_index("subset")["MAE"]
    R["skill_vs_persistence"] = [round(1 - r.MAE / pers[r.subset], 3) for r in R.itertuples()]
    imp = pd.Series(final.feature_importances_, index=feats).sort_values(ascending=False)
    return R, (T, imp, params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 7])
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(a.master_csv, parse_dates=["timestamp"])
    panel, wcols = daily_panel(master)
    p, feats = add_features(panel, wcols, a.horizons)
    print(f"{p['station'].nunique()} stations, {p['date'].min().date()} to {p['date'].max().date()}, "
          f"{len(feats)} features, weather columns used: {wcols or 'none'}")

    blocks = [
        ("A_reported_2023", "2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("B_cross_era_2024_25", "2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
    ]
    allR, preds, imps = [], [], {}
    for h in a.horizons:
        for b in blocks:
            R, extra = run_block(b[0], p, feats, h, *b[1:], None)
            if R is None:
                continue
            allR.append(R)
            T, imp, params = extra
            preds.append(T)
            imps[(b[0], h)] = imp
            print(f"  done {b[0]} h={h} params={params}")

    if not allR:
        print("Nothing evaluated; check dates in the data.")
        return
    R = pd.concat(allR, ignore_index=True)
    R.to_csv(out / "results.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(out / "test_predictions.csv", index=False)
    key = R[R.subset == "all days"].pivot_table(index=["block", "horizon"], columns="model", values="MAE")
    key["skill_xgb_vs_pers"] = (1 - key["xgb"] / key["persistence"]).round(3)
    key["skill_xgb_vs_clim"] = (1 - key["xgb"] / key["climatology"]).round(3)
    print("\nMAE (AQI points), all days:\n", key.round(1).to_string())
    print("\nWinter (Oct-Jan) rows + severe-day detection (AQI>=400):\n",
          R[R.subset == "Oct-Jan"][["block", "horizon", "model", "n", "MAE", "skill_vs_persistence", "severe_precision", "severe_recall"]].to_string(index=False))
    h0 = a.horizons[0]
    print(f"\nTop features, block B, horizon {h0}:\n", imps.get(("B_cross_era_2024_25", h0), pd.Series(dtype=float)).head(8).round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (bn, g) in zip(axes, R[R.subset == "all days"].groupby("block")):
        for model, c in (("persistence", "#8FB8C4"), ("climatology", "#B8B8B8"), ("xgb", "#C8473B")):
            s = g[g.model == model].sort_values("horizon")
            ax.plot(s["horizon"], s["MAE"], marker="o", color=c, label=model)
        ax.set_title(bn, fontsize=9)
        ax.set_xlabel("forecast horizon (days)")
    axes[0].set_ylabel("MAE (AQI points)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "mae_by_horizon.png", dpi=150)
    print(f"\nSaved results, predictions and plot to {out}")


if __name__ == "__main__":
    main()
