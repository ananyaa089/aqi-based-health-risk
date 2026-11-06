# AQI-Linked Respiratory Health Early-Warning System for Delhi 

Station-level 1-3 day AQI forecasts for 10 Delhi monitoring stations, a two-tier severe-pollution alert, an 80% prediction
range, plain-language advisories, and a literature-based respiratory-burden index. A Google-Trends health signal and a CAMS
benchmark were tested and are reported as weak / negative results. SDG 3 (primary), 11, 13.

## Method notes 
* **Time-ordered evaluation, no leakage.** Block A: train <= 2021, tune 2022, test 2023. Block B: train <= 2022, tune 2023, test 2024-25.
  Training rows whose label date falls in the tuning period are purged. `tests/test_features_leakage.py` proves no feature uses the future.
* **Baselines:** persistence (today's AQI) and climatology (station x day-of-year mean). Skill = 1 - MAE_model / MAE_baseline.
* **Intervals:** every confidence interval is a 14-day block bootstrap over dates (stations on a date move together).
* **Ranges:** conformalised quantile regression, calibrated on 2023, 80% nominal.
* **Alerts:** standard tier = threshold maximising F2 on validation (recall-first); high-confidence tier = lowest threshold reaching 80% (AQI>=300) / 70% (AQI>=400) validation precision.
* **Burden index:** one Delhi hospital study (six hospitals, news-reported): +0.709 respiratory admissions/week per ug/m3 PM2.5, above a 60 ug/m3 reference.
  A relative index, uncertain by about a factor of two; not a patient count.

## Honest limitations
* Skill is useful at 1-3 days; at 7 days it is little better than climatology. Peaks (AQI 400+) are under-predicted, and AQI is capped at 500.
* 2017-23 AQI is reported, 2024-25 AQI is computed by us; levels are not fully reconciled (models score similarly in both eras).
* Search interest: weak, winter-only, exploratory association, selected on data that overlapped the test period. Not used for forecasting.
* CAMS (Open-Meteo) is a same-day analysis, not a lead-time forecast, and its bias drifts by season/year; no robust gain.
* Dashboard is a replay of out-of-sample forecasts, not a live feed, and not medical advice.
* `test_synth/` (if present) is SYNTHETIC data used only to test the code; never use it for results.
