import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f07, f08 = load("f07", "07_forecast_aqi.py"), load("f08", "08_severe_classifier.py")
SEED = 42
HORIZONS = (1, 2, 3)
TRAIN_END, VAL_START, VAL_END, REPLAY_START = "2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01"
COVERAGE = 0.80
PRECISION_TARGET = {300: 0.80, 400: 0.70}  


def nice(f):
    base = {"aqi_lag0": "today's AQI", "aqi_roll3": "3-day mean AQI", "aqi_roll7": "7-day mean AQI", "aqi_trend": "3-day AQI trend",
            "city_aqi_lag0": "city-wide AQI today", "station_vs_city": "station-vs-city gap", "doy_sin": "time of year", "doy_cos": "time of year",
            "station_id": "station"}
    if f in base:
        return base[f]
    if f.startswith("aqi_lag"):
        return f"AQI {f.replace('aqi_lag', '')} days ago"
    w = {"windspeed_ms_city": "wind speed", "rain_mm_city": "rainfall", "temp_c_city": "temperature", "humidity_city": "humidity",
         "pressure_hpa_city": "pressure"}
    for k, v in w.items():
        if f.startswith(k):
            return f"change in {v}" if f.endswith("_chg1") else v
    return f


def why_text(contrib, names, k=3):
    """contrib: (n, F+1) with bias last; returns top-k drivers as text per row (log-odds contributions)."""
    c = contrib[:, :-1]
    uniq = list(dict.fromkeys(names))
    merged = np.stack([c[:, [i for i, n in enumerate(names) if n == u]].sum(axis=1) for u in uniq], axis=1)
    out = []
    for row in merged:
        top = np.argsort(-np.abs(row))[:k]
        out.append(" | ".join(f"{uniq[i]} ({'raises' if row[i] > 0 else 'lowers'} risk, {row[i]:+.2f})" for i in top))
    return out, uniq, np.abs(merged).mean(axis=0)


def pick_threshold_precision(y, prob, target):
    for t in np.linspace(0.05, 0.99, 95):
        pred = prob >= t
        if pred.sum() < 20:
            break
        if ((pred) & (y == 1)).sum() / pred.sum() >= target:
            return float(t), True
    return 0.5, False  


def main():
    master = pd.read_csv(sys.argv[1], parse_dates=["timestamp"])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    p, feats, _ = f08.build(master)
    g = p.groupby("station")["aqi"]
    for h in HORIZONS:
        p[f"y_h{h}"] = g.shift(-h)

    ok_base = p["aqi_lag0"].notna() & p[feats].notna().sum(axis=1).ge(len(feats) // 2)
    replay = ok_base & (p["date"] >= pd.Timestamp(REPLAY_START))
    D = p.loc[replay, ["station", "date", "aqi_lag0"]].rename(columns={"aqi_lag0": "aqi_today"}).copy()
    meta = {"trained_through": VAL_END, "replay_start": REPLAY_START, "thresholds": {}, "thresholds_hi": {}, "interval_nominal": COVERAGE}
    names = [nice(f) for f in feats]
    shap_rows = []

    for h in HORIZONS:
        ycol = f"y_h{h}"
        lab_ok = ok_base & p[ycol].notna()
        tr_all = lab_ok & (p["date"] + pd.Timedelta(days=h) <= pd.Timestamp(VAL_END))
        m = xgb.XGBRegressor(max_depth=5, n_estimators=300, learning_rate=0.04, min_child_weight=5, subsample=0.8,
                             colsample_bytree=0.8, random_state=SEED, n_jobs=2)
        m.fit(p.loc[tr_all, feats], p.loc[tr_all, ycol])
        pred = m.predict(p.loc[replay, feats])
        D[f"pred_h{h}"] = pred
        D[f"actual_h{h}"] = p.loc[replay, ycol].to_numpy()

        a = (1 - COVERAGE) / 2
        tr = lab_ok & (p["date"] + pd.Timedelta(days=h) <= pd.Timestamp(TRAIN_END))
        cal = lab_ok & p["date"].between(pd.Timestamp(VAL_START), pd.Timestamp(VAL_END))
        qm = {}
        for q in (a, 1 - a):
            qm[q] = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, max_depth=4, n_estimators=300, learning_rate=0.04,
                                     min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=2)
            qm[q].fit(p.loc[tr, feats], p.loc[tr, ycol])
        lo_c, hi_c = qm[a].predict(p.loc[cal, feats]), qm[1 - a].predict(p.loc[cal, feats])
        yc = p.loc[cal, ycol].to_numpy()
        E = np.maximum(lo_c - yc, yc - hi_c)
        n = len(E)
        qhat = float(np.quantile(E, min(1.0, np.ceil((n + 1) * COVERAGE) / n), method="higher"))
        lo, hi = qm[a].predict(p.loc[replay, feats]) - qhat, qm[1 - a].predict(p.loc[replay, feats]) + qhat
        D[f"lo_h{h}"] = np.clip(np.minimum(lo, pred), 0, 500)
        D[f"hi_h{h}"] = np.clip(np.maximum(hi, pred), 0, 500)
        meta[f"interval_margin_h{h}"] = round(qhat, 1)
        print(f"regression h={h}: trained on {int(tr_all.sum())} station-days; 80% range margin {qhat:+.1f}")

        if h == 1:
            c = m.get_booster().predict(xgb.DMatrix(p.loc[replay, feats], feature_names=feats), pred_contribs=True)
            _, uniq, imp = why_text(c, names)
            shap_rows += [dict(model="AQI forecast +1 day", feature=u, mean_abs_contribution=round(float(v), 3)) for u, v in zip(uniq, imp)]

    for T in f08.THRESHOLDS:
        y = (p["fut_max"] >= T).astype(int)
        base = ok_base & p["fut_valid"].ge(2)
        tr = base & (p["date"] + pd.Timedelta(days=f08.WINDOW) <= pd.Timestamp(TRAIN_END))
        va = base & p["date"].between(pd.Timestamp(VAL_START), pd.Timestamp(VAL_END))
        m1 = f08.make_clf().fit(p.loc[tr, feats], y[tr])
        pv = m1.predict_proba(p.loc[va, feats])[:, 1]
        thr = f08.pick_threshold(y[va].to_numpy(), pv)
        thr_hi, reached = pick_threshold_precision(y[va].to_numpy(), pv, PRECISION_TARGET[T])
        tr2 = base & (p["date"] + pd.Timedelta(days=f08.WINDOW) <= pd.Timestamp(VAL_END))
        m2 = f08.make_clf().fit(p.loc[tr2, feats], y[tr2])
        prob = m2.predict_proba(p.loc[replay, feats])[:, 1]
        D[f"p_sev{T}"] = prob
        D[f"alert{T}"] = (prob >= thr).astype(int)
        D[f"alert_hi{T}"] = (prob >= max(thr_hi, thr)).astype(int)
        D[f"actual_sev{T}"] = np.where(p.loc[replay, "fut_valid"].ge(2), (p.loc[replay, "fut_max"] >= T).astype(int), np.nan)
        c = m2.get_booster().predict(xgb.DMatrix(p.loc[replay, feats], feature_names=feats), pred_contribs=True)
        txt, uniq, imp = why_text(c, names)
        D[f"why_sev{T}"] = txt
        shap_rows += [dict(model=f"Severe alert AQI>={T}", feature=u, mean_abs_contribution=round(float(v), 3)) for u, v in zip(uniq, imp)]
        meta["thresholds"][str(T)] = round(thr, 3)
        meta["thresholds_hi"][str(T)] = round(max(thr_hi, thr), 3)
        print(f"classifier AQI>={T}: alert threshold {thr:.2f}; high-confidence tier threshold {max(thr_hi, thr):.2f}"
              + ("" if reached else f"  (validation never reached {PRECISION_TARGET[T]:.0%} precision; used the p>=0.5 fallback)"))

    D["date"] = D["date"].dt.strftime("%Y-%m-%d")
    D.round(3).to_csv(out / "dashboard_data.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(out / "shap_global.csv", index=False)
    (out / "dashboard_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nExported {len(D)} station-day forecasts ({D['date'].min()} to {D['date'].max()}) to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main()
