import numpy as np
import pandas as pd

from conftest import load_script, make_master

f07 = load_script("07_forecast_aqi.py")
f08 = load_script("08_severe_classifier.py")


def daily_means(master):
    m = master.copy()
    m["date"] = m["timestamp"].dt.normalize()
    return m.groupby(["station", "date"])["aqi"].mean()


def test_lags_and_targets_are_aligned(master):
    p, feats, _ = f08.build(master)
    dm = daily_means(master)
    st = "Anand Vihar"
    s = p[p.station == st].set_index("date")
    d = pd.Timestamp("2020-06-15")
    assert np.isclose(s.loc[d, "aqi_lag0"], dm[(st, d)])
    assert np.isclose(s.loc[d, "aqi_lag1"], dm[(st, d - pd.Timedelta(days=1))])
    assert np.isclose(s.loc[d, "aqi_lag7"], dm[(st, d - pd.Timedelta(days=7))])
    nxt = [dm[(st, d + pd.Timedelta(days=k))] for k in (1, 2, 3)]
    assert np.isclose(s.loc[d, "fut_max"], max(nxt))
    assert s.loc[d, "fut_valid"] == 3


def test_regression_target_is_h_days_ahead(master):
    panel, wcols = f07.daily_panel(master)
    p, feats = f07.add_features(panel, wcols, [1, 3])
    dm = daily_means(master)
    st, d = "R K Puram", pd.Timestamp("2020-07-01")
    row = p[(p.station == st) & (p.date == d)].iloc[0]
    assert np.isclose(row["y_h1"], dm[(st, d + pd.Timedelta(days=1))])
    assert np.isclose(row["y_h3"], dm[(st, d + pd.Timedelta(days=3))])


def test_changing_the_future_never_changes_past_features(master):
    cut = pd.Timestamp("2020-09-01")
    m2 = master.copy()
    m2.loc[m2["timestamp"].dt.normalize() == cut, "aqi"] += 150
    a, feats, _ = f08.build(master)
    b, _, _ = f08.build(m2)
    before = a["date"] < cut
    for col in feats:
        assert np.allclose(a.loc[before, col].to_numpy(float), b.loc[before, col].to_numpy(float), equal_nan=True), f"feature {col} leaks the future"


def test_days_with_fewer_than_12_valid_hours_are_missing(master):
    m = master.copy()
    day = pd.Timestamp("2020-03-10")
    mask = (m["timestamp"].dt.normalize() == day) & (m["timestamp"].dt.hour >= 8)  
    m.loc[mask, "aqi"] = np.nan
    panel, _ = f07.daily_panel(m)
    assert panel[(panel.date == day)]["aqi"].isna().all()


def test_time_ordered_block_has_no_overlap_and_correct_horizon():
    big = make_master(start="2017-01-01", days=5 * 365, seed=3)
    panel, wcols = f07.daily_panel(big)
    p, feats = f07.add_features(panel, wcols, [3])
    R, extra = f07.run_block("t", p, feats, 3, "2019-12-31", "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31", None)
    T, imp, params = extra
    assert T["date"].min() >= pd.Timestamp("2021-01-01") and T["date"].max() <= pd.Timestamp("2021-12-31")
    assert ((T["target_date"] - T["date"]).dt.days == 3).all()
    assert np.isfinite(T["xgb"]).all()
    mae = R[(R.subset == "all days")].set_index("model")["MAE"]
    assert mae["xgb"] < mae["climatology"]
