# AQI-Linked Respiratory Health Early-Warning System for Delhi (capstone)

Station-level 1-3 day AQI forecasts for 10 Delhi monitoring stations, a two-tier severe-pollution alert, an 80% prediction
range, plain-language advisories, and a literature-based respiratory-burden index. A Google-Trends health signal and a CAMS
benchmark were tested and are reported as weak / negative results. SDG 3 (primary), 11, 13.

## Folder layout
    data/                 raw station CSVs (OpenCity / CPCB / DPCC / IMD)       [not committed]
    data/clean/           everything generated (master file, forecasts, report tables and figures)
    scripts/              pipeline, numbered in the order they were built; run_all.py runs them all
    app/dashboard.py      Streamlit dashboard (replay mode)
    tests/                pytest unit tests (AQI formula, parsers, leakage, thresholds, burden, dashboard smoke test)
    DEPLOY.md             how to publish the dashboard

## Install (once)
    pip install -r requirements.txt

## Run everything
From the folder that contains `data/`, `scripts/` and `app/`:

    python scripts/run_all.py                    # all steps (needs internet for weather, Trends and CAMS; Trends takes ~10 min)
    python scripts/run_all.py --skip-downloads   # reuse weather / Trends / CAMS files that are already in data/clean
    python scripts/run_all.py --list             # show every step and its output
    python scripts/run_all.py --from 10          # resume from a step after a failure
    streamlit run app/dashboard.py               # launch the dashboard

If you only want to rebuild the dashboard after changing a model:  `python scripts/run_all.py --only 13 14 15`.

## What each step does
| step | script | output (in data/clean) |
|---|---|---|
| 01, 03 | build_master.py | master_aqi_hourly.csv: all stations, hourly. 2017-23 AQI is reported; 2024-25 AQI is computed from PM2.5/PM10 (CPCB breakpoints, 24 h mean); `aqi_source` marks which |
| 02 | fetch_weather_openmeteo.py | weather_hourly.csv (city-wide) |
| 04 | pull_google_trends.py | trends_weekly.csv (stitched windows, relative 0-100) |
| 05 | 02_eda_correlation.py | eda/: weekly dataset, lag correlations (raw and deseasonalised) |
| 06 | 03_forecast.py | forecast/: weekly search-index forecast + ablation (AQI adds nothing detectable) |
| 07, 09 | 04_diagnose.py, 06_rescue_trends.py | diagnostics/: COVID confound, per-term, winter-only association |
| 08 | 05_check_aqi_shift.py | prints station-composition and PM2.5/PM10 checks on the 2023 -> 2024 AQI level |
| 10 | 07_forecast_aqi.py | aqi_forecast/: daily AQI forecast, 1/3/7 days, baselines, two test blocks |
| 11 | 08_severe_classifier.py | severe_alerts/: P(AQI >= 300 / 400 within 3 days), future-weather ablation |
| 12 | 09_health_translation.py | health/: burden translation + pre-committed held-out hypothesis test |
| 13 | 10_export_dashboard_data.py | dashboard/: final models, ranges, alert tiers, "why" text, replay forecasts for 2024-25 |
| 14, 15 | 11_ci_and_figures.py, 13_uncertainty_eval.py | report/: confidence intervals, error analysis, calibration, tiers, figures 1-6 |
| 16-18 | 12a, 12b, 14 | cams_daily.csv, report/cams_benchmark.csv, CAMS diagnostics |
| 19 | tests/ | pytest |

## Method notes (for the report and the viva)
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
