# Proposed division of responsibility (3 members)

IMPORTANT: this is a proposed split of areas to own and explain. The project log and the report must state who ACTUALLY did what,
because the guide marks each student individually and every member is questioned in the viva. Edit this to match reality.

| | Member 1: Data and pipeline | Member 2: Modelling and evaluation | Member 3: Health analysis, product and documentation |
|---|---|---|---|
| Scripts to own | parse_legacy_2017_2023, resample_2024_25, build_master, fetch_weather_openmeteo, 05_check_aqi_shift, 12a_fetch_cams, 14_cams_diagnose, run_all | 07_forecast_aqi, 08_severe_classifier, 10_export_dashboard_data, 11_ci_and_figures, 13_uncertainty_eval, 12b_cams_benchmark | pull_google_trends, 02_eda_correlation, 03_forecast, 04_diagnose, 06_rescue_trends, 09_health_translation, app/dashboard.py |
| Tests to own | AQI formula, parsers | Leakage, split, thresholds, bootstrap | Burden conversion, dashboard smoke tests |
| Report chapters | Data sources and cleaning, data quality (two AQI eras), requirements for data | Methodology, results, validation, limitations of the models | Introduction, SDG impact, literature, stakeholder analysis, dashboard design, risk analysis, Trends negative result |
| Must be able to explain in the viva | How each raw format is parsed; UTC to IST; the CPCB AQI formula; why 2017-23 and 2024-25 AQI differ in type | Why the split is time-ordered; leakage tests; baselines and skill; bootstrap CIs; conformal ranges; calibration; two alert tiers | Why search interest failed (COVID, null correlations, held-out test and its flaw); the burden slope and its limits; dashboard design choices |
| Shared | Weekly team check-in; everyone can run `python scripts/run_all.py` and explain the architecture diagram | | |

Responsible-AI note: your university framework requires responsible AI use. State in the report which parts were developed with AI assistance
(code generation and review, documentation drafts) and confirm that every member has read, run and understood their parts.
