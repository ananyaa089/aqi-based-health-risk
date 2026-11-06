import numpy as np
import pandas as pd

from conftest import load_script

bm = load_script("build_master.py")


def frame(pm25, pm10, hours=72):
    idx = pd.date_range("2024-01-01", periods=hours, freq="1h")
    return pd.DataFrame({"pm2_5": pm25, "pm10": pm10}, index=idx)


def test_breakpoints_pm25():
    for conc, expected in [(30, 50), (60, 100), (90, 200), (120, 300), (250, 400), (380, 500)]:
        assert bm.compute_aqi(frame(conc, 0)).iloc[-1] == expected


def test_breakpoints_pm10():
    for conc, expected in [(50, 50), (100, 100), (250, 200), (350, 300), (430, 400), (510, 500)]:
        assert bm.compute_aqi(frame(0, conc)).iloc[-1] == expected


def test_aqi_is_max_of_sub_indices():
    assert bm.compute_aqi(frame(30, 250)).iloc[-1] == 200
    assert bm.compute_aqi(frame(250, 50)).iloc[-1] == 400  


def test_values_above_top_breakpoint_are_clamped_to_500():
    assert bm.compute_aqi(frame(900, 900)).iloc[-1] == 500


def test_needs_16_of_24_hours():
    out = bm.compute_aqi(frame(100, 100))
    assert out.iloc[:15].isna().all() and out.iloc[15:].notna().all()


def test_interpolates_between_breakpoints():
    assert bm.compute_aqi(frame(45, 0)).iloc[-1] == 75  


def test_uses_24h_rolling_mean():
    pm25 = np.r_[np.full(24, 30.0), np.full(24, 90.0)]
    out = bm.compute_aqi(frame(pm25, np.zeros(48), hours=48))
    assert out.iloc[23] == 50 and out.iloc[-1] == 200  
    assert 50 < out.iloc[30] < 200  