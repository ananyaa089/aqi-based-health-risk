import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def steps(raw, c):
    py = sys.executable
    s = lambda name, *a: [py, str(SCRIPTS / name), *map(str, a)]  # noqa: E731
    return [
        ("01", "Merge stations (no weather yet)", s("build_master.py", raw, f"{c}/master_aqi_hourly.csv"), f"{c}/master_aqi_hourly.csv", False),
        ("02", "Download city weather (Open-Meteo)", s("fetch_weather_openmeteo.py", f"{c}/weather_hourly.csv", "2017-01-01", "2025-12-31"), f"{c}/weather_hourly.csv", True),
        ("03", "Merge stations again, with weather", s("build_master.py", raw, f"{c}/master_aqi_hourly.csv", "--weather", f"{c}/weather_hourly.csv"), f"{c}/master_aqi_hourly.csv", False),
        ("04", "Download Google Trends (slow)", s("pull_google_trends.py", f"{c}/trends_weekly.csv"), f"{c}/trends_weekly.csv", True),
        ("05", "Weekly dataset + AQI vs searches correlations", s("02_eda_correlation.py", f"{c}/master_aqi_hourly.csv", f"{c}/trends_weekly.csv", f"{c}/eda"), f"{c}/eda/weekly_dataset.csv", False),
        ("06", "Weekly search-index forecast (ablation)", s("03_forecast.py", f"{c}/eda/weekly_dataset.csv", f"{c}/forecast"), f"{c}/forecast/results.csv", False),
        ("07", "Diagnostics: COVID, per-term, sub-periods", s("04_diagnose.py", f"{c}/master_aqi_hourly.csv", f"{c}/eda/weekly_dataset.csv", f"{c}/diagnostics"), f"{c}/diagnostics/C_subperiods.csv", False),
        ("08", "AQI level-shift check (2023 -> 2024)", s("05_check_aqi_shift.py", f"{c}/master_aqi_hourly.csv"), None, False),
        ("09", "Rescue attempt for the search index", s("06_rescue_trends.py", f"{c}/eda/weekly_dataset.csv", f"{c}/diagnostics"), f"{c}/diagnostics/F_rescue.csv", False),
        ("10", "Daily AQI forecast, 1/3/7 days", s("07_forecast_aqi.py", f"{c}/master_aqi_hourly.csv", f"{c}/aqi_forecast"), f"{c}/aqi_forecast/test_predictions.csv", False),
        ("11", "Severe-episode classifier", s("08_severe_classifier.py", f"{c}/master_aqi_hourly.csv", f"{c}/severe_alerts"), f"{c}/severe_alerts/results.csv", False),
        ("12", "Health translation + held-out hypothesis", s("09_health_translation.py", f"{c}/aqi_forecast/test_predictions.csv", f"{c}/eda/weekly_dataset.csv", f"{c}/health"), f"{c}/health/heldout_hypothesis.csv", False),
        ("13", "Export dashboard data (final models)", s("10_export_dashboard_data.py", f"{c}/master_aqi_hourly.csv", f"{c}/dashboard"), f"{c}/dashboard/dashboard_data.csv", False),
        ("14", "Confidence intervals + report figures 1-4", s("11_ci_and_figures.py", f"{c}/aqi_forecast/test_predictions.csv", f"{c}/dashboard/dashboard_data.csv", f"{c}/report"), f"{c}/report/forecast_error_cis.csv", False),
        ("15", "Uncertainty, calibration, alert tiers (figs 5-6)", s("13_uncertainty_eval.py", f"{c}/dashboard", f"{c}/report"), f"{c}/report/alert_tiers.csv", False),
        ("16", "Download CAMS PM2.5/PM10 (Open-Meteo)", s("12a_fetch_cams.py", f"{c}/cams_daily.csv"), f"{c}/cams_daily.csv", True),
        ("17", "CAMS benchmark (upper bound)", s("12b_cams_benchmark.py", f"{c}/aqi_forecast/test_predictions.csv", f"{c}/cams_daily.csv", f"{c}/report"), f"{c}/report/cams_benchmark.csv", False),
        ("18", "CAMS diagnostics", s("14_cams_diagnose.py", f"{c}/master_aqi_hourly.csv", f"{c}/cams_daily.csv"), None, False),
        ("19", "Unit tests", [py, "-m", "pytest", str(SCRIPTS.parent / "tests"), "-q"], None, False),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data", help="folder with the raw station CSVs")
    ap.add_argument("--clean", default="data/clean", help="folder for everything generated")
    ap.add_argument("--skip-downloads", action="store_true", help="skip weather/Trends/CAMS steps whose output file already exists")
    ap.add_argument("--from", dest="start", help="first step id to run, e.g. 07")
    ap.add_argument("--only", nargs="+", help="run only these step ids")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    plan = steps(a.raw, a.clean)
    if a.list:
        for sid, title, _, out, dl in plan:
            print(f"{sid}  {title}{'  [internet]' if dl else ''}" + (f"  -> {out}" if out else ""))
        return
    if a.only:
        plan = [p for p in plan if p[0] in set(a.only)]
    elif a.start:
        ids = [p[0] for p in plan]
        if a.start not in ids:
            sys.exit(f"Unknown step {a.start}; use --list")
        plan = plan[ids.index(a.start):]

    Path(a.clean).mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for sid, title, cmd, out, dl in plan:
        if a.skip_downloads and dl and out and Path(out).exists():
            print(f"[{sid}] {title}: skipped (already have {out})")
            continue
        print(f"\n[{sid}] {title}\n    $ {' '.join(cmd)}")
        if a.dry_run:
            continue
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"\nStep {sid} failed (exit code {r.returncode}). Fix it, then resume with:  python scripts/run_all.py --from {sid}")
        print(f"    done in {time.time() - t0:.0f}s")
    print(f"\nAll requested steps finished in {(time.time() - t_all) / 60:.1f} min.")
    if not a.dry_run:
        print("Launch the dashboard with:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
