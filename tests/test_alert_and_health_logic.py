import numpy as np

from conftest import load_script

f08 = load_script("08_severe_classifier.py")
f09 = load_script("09_health_translation.py")
f10 = load_script("10_export_dashboard_data.py")


def test_f2_threshold_is_recall_oriented():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.2).astype(int)
    prob = np.clip(0.5 * y + rng.normal(0.25, 0.15, 4000), 0, 1)
    thr = f08.pick_threshold(y, prob)
    prec, rec = f08.pr(y, prob >= thr)
    assert 0 < thr < 1 and rec > prec  


def test_precision_tier_reaches_target_or_falls_back():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.2).astype(int)
    good = np.clip(0.6 * y + rng.normal(0.2, 0.1, 5000), 0, 1)
    thr, reached = f10.pick_threshold_precision(y, good, 0.8)
    assert reached and ((good >= thr) & (y == 1)).sum() / (good >= thr).sum() >= 0.8
    useless = rng.random(5000)  
    thr2, reached2 = f10.pick_threshold_precision(y, useless, 0.8)
    assert not reached2 and thr2 == 0.5


def test_friendly_names_and_why_text():
    assert f10.nice("aqi_lag0") == "today's AQI" and f10.nice("windspeed_ms_city_chg1") == "change in wind speed"
    names = [f10.nice(f) for f in ["aqi_lag0", "doy_sin", "doy_cos", "rain_mm_city_lag0"]]
    contrib = np.array([[2.0, 0.3, -0.9, 0.1, 0.0], [-0.5, 0.0, 0.0, 1.2, 0.0]])  
    text, uniq, imp = f10.why_text(contrib, names, k=2)
    assert len(text) == 2 and uniq.count("time of year") == 1  
    assert text[0].startswith("today's AQI (raises risk") and text[1].startswith("rainfall (raises risk")


def test_aqi_to_pm25_inverts_the_breakpoints():
    for aqi, pm in [(0, 0), (50, 30), (100, 60), (200, 90), (300, 120), (400, 250), (500, 380)]:
        assert np.isclose(f09.aqi_to_pm25(aqi), pm)
    assert f09.aqi_to_pm25(900) == 380  


def test_excess_admissions_use_the_literature_slope():
    assert f09.excess_admissions(100) == 0                     
    assert f09.excess_admissions(30) == 0                      
    assert np.isclose(f09.excess_admissions(200), (90 - 60) * 0.709)
    assert np.isclose(f09.excess_admissions(200, 0.5 * 0.709), 0.5 * f09.excess_admissions(200))
    a = [f09.excess_admissions(v) for v in range(0, 501, 25)]
    assert all(x <= y for x, y in zip(a, a[1:]))               


def test_burden_tiers():
    assert str(f09.tier(np.array([0.0]))[0]) == "none"
    assert str(f09.tier(np.array([30.0]))[0]) == "moderate"
    assert str(f09.tier(np.array([200.0]))[0]) == "very high"
