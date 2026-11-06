import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Delhi AQI Early Warning", layout="wide")

COORDS = {
    "Anand Vihar": (28.6469, 77.3158), "Ashok Vihar": (28.6954, 77.1818), "Aya Nagar": (28.4706, 77.1099),
    "Bawana": (28.7762, 77.0511), "Burari Crossing": (28.7256, 77.2011), "CRRI Mathura Road": (28.5512, 77.2736),
    "Chandni Chowk": (28.6578, 77.2273), "DTU": (28.7500, 77.1112), "Dwarka Sector 8": (28.5710, 77.0719),
    "R K Puram": (28.5633, 77.1869),
}
CATS = [(50, "Good", [0, 158, 96]), (100, "Satisfactory", [141, 198, 63]), (200, "Moderate", [247, 211, 40]),
        (300, "Poor", [243, 146, 0]), (400, "Very poor", [220, 53, 69]), (1e9, "Severe", [130, 20, 60])]
PM25_X, PM25_Y = [0, 30, 60, 90, 120, 250, 380], [0, 50, 100, 200, 300, 400, 500]
SLOPE, REF_PM25 = 0.709, 60.0  
TIERS = [(0.001, "None"), (20, "Low"), (60, "Moderate"), (120, "High"), (1e9, "Very high")]


def category(aqi):
    for hi, name, col in CATS:
        if aqi <= hi:
            return name, col
    return CATS[-1][1], CATS[-1][2]


def burden(aqi):
    pm = np.interp(np.clip(aqi, 0, 500), PM25_Y, PM25_X)
    x = max(pm - REF_PM25, 0) * SLOPE
    return x, next(n for hi, n in TIERS if x <= hi)


def advisory(station, peak, p400, alert400, alert300, tier, hi400=False):
    cat, _ = category(peak)
    severe_act = ("Limit outdoor exertion, keep windows closed, use a well-fitted N95/FFP2 mask outdoors, and keep "
                  "quick-relief inhalers accessible. People with asthma, COPD, heart disease, older adults, children and "
                  "pregnant women should avoid going out where possible.")
    if hi400:
        head = f"**High-confidence severe-pollution alert for {station}.** A severe episode (AQI 400+) is very likely within 3 days."
        act = severe_act
    elif alert400 or p400 >= 0.5:
        head = f"**Severe-pollution watch for {station}.** A severe episode (AQI 400+) is possible within 3 days (standard alert tier: catches most episodes, with more false alarms)."
        act = severe_act
    elif alert300:
        head = f"**Elevated-pollution watch for {station}.** AQI above 300 ('very poor') is likely within 3 days."
        act = "Reduce prolonged outdoor activity, especially near traffic. Sensitive groups should take extra precautions."
    elif peak > 200:
        head = f"**Poor air expected at {station}** (peak forecast {peak:.0f}, {cat.lower()})."
        act = "Sensitive groups should cut back on strenuous outdoor activity."
    else:
        head = f"**No major pollution episode expected at {station}** (peak forecast {peak:.0f}, {cat.lower()})."
        act = "Normal activity is fine for most people."
    return f"{head} Estimated respiratory-burden tier: **{tier}**.\n\n{act}"


@st.cache_data
def load_data(path):
    d = pd.read_csv(path, parse_dates=["date"])
    return d


def read_csv_if(path):
    try:
        return pd.read_csv(path) if Path(path).exists() else None
    except Exception:
        return None


st.sidebar.title("Delhi AQI early warning")
root = Path(st.sidebar.text_input("Data folder", "data/clean"))
dash_csv = root / "dashboard" / "dashboard_data.csv"
if not dash_csv.exists():
    st.error(f"Could not find {dash_csv}. Run scripts/10_export_dashboard_data.py first.")
    st.stop()
D = load_data(str(dash_csv))
if "lo_h1" not in D.columns or "alert_hi400" not in D.columns:
    st.error("dashboard_data.csv is from an older export. Re-run scripts/10_export_dashboard_data.py (adds ranges, alert tiers and explanations).")
    st.stop()
meta = json.loads((root / "dashboard" / "dashboard_meta.json").read_text()) if (root / "dashboard" / "dashboard_meta.json").exists() else {}

dmin, dmax = D["date"].min().date(), D["date"].max().date()
default = pd.Timestamp("2024-11-01").date() if dmin <= pd.Timestamp("2024-11-01").date() <= dmax else dmin
day = st.sidebar.date_input("Issue date (forecast made at end of this day)", default, min_value=dmin, max_value=dmax)
reveal = st.sidebar.checkbox("Reveal what actually happened", value=False)
st.sidebar.info(
    "**Replay mode.** Forecasts shown are out-of-sample: models were trained only on data through "
    f"{meta.get('trained_through', '2023-12-31')}. This is not a live forecast and not medical advice."
)

day_df = D[D["date"] == pd.Timestamp(day)].copy()
st.title("Delhi AQI early-warning system")
st.caption("Station-level 1-3 day AQI forecasts, severe-episode alerts and a literature-based respiratory-burden index.")

tab1, tab2, tab3 = st.tabs(["Forecast map", "Station history", "Evidence and limits"])

with tab1:
    if day_df.empty:
        st.warning("No forecasts for this date (data gap).")
    else:
        day_df["peak_pred"] = day_df[["pred_h1", "pred_h2", "pred_h3"]].max(axis=1)
        day_df["cat"] = day_df["peak_pred"].apply(lambda v: category(v)[0])
        day_df["color"] = day_df["peak_pred"].apply(lambda v: category(v)[1])
        day_df["burden"] = day_df["peak_pred"].apply(lambda v: burden(v)[0])
        day_df["tier"] = day_df["peak_pred"].apply(lambda v: burden(v)[1])
        day_df["lat"] = day_df["station"].map(lambda s: COORDS.get(s, (np.nan, np.nan))[0])
        day_df["lon"] = day_df["station"].map(lambda s: COORDS.get(s, (np.nan, np.nan))[1])
        day_df["label"] = day_df.apply(lambda r: f"{r['station']}: peak {r['peak_pred']:.0f} ({r['cat']})", axis=1)
        m = day_df.dropna(subset=["lat", "lon"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("City mean AQI today", f"{day_df['aqi_today'].mean():.0f}")
        c2.metric("Mean AQI, next day (forecast)", f"{day_df['pred_h1'].mean():.0f}")
        c3.metric("400+ alerts: standard / high-confidence", f"{int(day_df['alert400'].sum())} / {int(day_df['alert_hi400'].sum())}")
        c4.metric("300+ alerts: standard / high-confidence", f"{int(day_df['alert300'].sum())} / {int(day_df['alert_hi300'].sum())}")

        left, right = st.columns([3, 2])
        with left:
            layer = pdk.Layer("ScatterplotLayer", data=m, get_position="[lon, lat]", get_fill_color="color",
                              get_radius=1800, pickable=True, opacity=0.85, stroked=True, get_line_color=[40, 40, 40],
                              line_width_min_pixels=1)
            view = pdk.ViewState(latitude=28.64, longitude=77.15, zoom=9.3)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{label}"}))
            st.caption("Colour = worst forecast daily AQI over the next 3 days (CPCB categories). Locations are approximate.")
        with right:
            t = day_df.sort_values("pred_h1", ascending=False)
            show = pd.DataFrame({"Station": t["station"], "AQI today": t["aqi_today"].round(0).astype(int)})
            for h in (1, 2, 3):
                show[f"+{h} day (80% range)"] = [f"{p:.0f} ({lo:.0f}-{hi:.0f})" for p, lo, hi in zip(t[f"pred_h{h}"], t[f"lo_h{h}"], t[f"hi_h{h}"])]
            show["P(300+)"], show["P(400+)"], show["Burden tier"] = t["p_sev300"].round(2), t["p_sev400"].round(2), t["tier"].astype(str)
            st.dataframe(show, hide_index=True, width="stretch")

        st.subheader("Plain-language advisories")
        for r in day_df.sort_values("peak_pred", ascending=False).itertuples():
            with st.expander(f"{r.station}: peak forecast {r.peak_pred:.0f} ({r.cat}); burden tier {r.tier}"):
                st.markdown(advisory(r.station, r.peak_pred, r.p_sev400, bool(r.alert400), bool(r.alert300), r.tier, bool(r.alert_hi400)))
                st.markdown(f"**Why (severe-episode model, main drivers):** {r.why_sev400}")
                if reveal:
                    act = [r.actual_h1, r.actual_h2, r.actual_h3]
                    st.markdown("**What actually happened** (daily mean AQI, +1/+2/+3 days): "
                                + ", ".join("n/a" if pd.isna(a) else f"{a:.0f}" for a in act))
        st.caption("Burden tier = excess weekly respiratory admissions (six-hospital sample) implied by the forecast PM2.5 above 60 ug/m3. "
                   "It is a relative index, not a count of patients.")

with tab2:
    st.subheader("Forecast vs actual at one station")
    st_name = st.selectbox("Station", sorted(D["station"].unique()))
    horizon = st.radio("Forecast horizon", [1, 2, 3], horizontal=True, format_func=lambda h: f"+{h} day")
    S = D[D["station"] == st_name].sort_values("date")
    win = S[(S["date"] >= pd.Timestamp(day) - pd.Timedelta(days=60)) & (S["date"] <= pd.Timestamp(day) + pd.Timedelta(days=30))]
    chart = pd.DataFrame({"actual": win[f"actual_h{horizon}"].to_numpy(), "forecast": win[f"pred_h{horizon}"].to_numpy(),
                          "persistence (today)": win["aqi_today"].to_numpy(), "range low (10%)": win[f"lo_h{horizon}"].to_numpy(),
                          "range high (90%)": win[f"hi_h{horizon}"].to_numpy()}, index=win["date"] + pd.Timedelta(days=horizon))
    st.line_chart(chart)
    ok = S.dropna(subset=[f"actual_h{horizon}"])
    mae_m = (ok[f"pred_h{horizon}"] - ok[f"actual_h{horizon}"]).abs().mean()
    mae_p = (ok["aqi_today"] - ok[f"actual_h{horizon}"]).abs().mean()
    a, b = st.columns(2)
    a.metric(f"Model MAE, +{horizon} day (whole replay period)", f"{mae_m:.1f}")
    b.metric("Persistence MAE", f"{mae_p:.1f}")

with tab3:
    st.subheader("How well does it work?")
    r07 = read_csv_if(root / "aqi_forecast" / "results.csv")
    r08 = read_csv_if(root / "severe_alerts" / "results.csv")
    r09 = read_csv_if(root / "health" / "heldout_hypothesis.csv")
    if r07 is not None:
        st.markdown("**AQI forecast error (MAE, AQI points; lower is better).** Test block B = 2024-25.")
        t = r07[(r07.subset == "all days")].pivot_table(index=["block", "horizon"], columns="model", values="MAE").round(1)
        st.dataframe(t, width="stretch")
    if r08 is not None:
        st.markdown("**Severe-episode alerts** (severe day within next 3 days). PR-AUC: higher is better.")
        st.dataframe(r08, hide_index=True, width="stretch")
    if r09 is not None:
        st.markdown("**Search-interest hypothesis (asthma searches vs lagged worst-station AQI, winters).**")
        st.dataframe(r09, hide_index=True, width="stretch")
    rep = root / "report"
    for fname, title in (("forecast_error_cis.csv", "**Forecast skill with 95% confidence intervals.**"),
                         ("interval_coverage.csv", "**Do the 80% ranges contain the truth about 80% of the time?**"),
                         ("calibration_summary.csv", "**Are the alert probabilities calibrated?**"),
                         ("alert_tiers.csv", "**Standard vs high-confidence alerts (2024-25).**"),
                         ("cams_benchmark.csv", "**External benchmark: CAMS-derived PM (upper bound, not a lead-time forecast).**")):
        tbl = read_csv_if(rep / fname)
        if tbl is not None:
            st.markdown(title)
            st.dataframe(tbl, hide_index=True, width="stretch")
    for fname in ("fig1_mae_by_horizon.png", "fig3_alert_pr_curves.png", "fig5_calibration.png", "fig6_global_drivers.png"):
        if (rep / fname).exists():
            st.image(str(rep / fname))
    st.subheader("Limits you should know")
    st.markdown(
        "- **Skill fades with horizon.** Useful at 1-3 days; at 7 days it is little better than the seasonal average.\n"
        "- **Alerts trade false alarms for recall.** The standard tier favours catching severe days; the high-confidence tier has fewer false alarms but misses more.\n"
        "- **Peaks are under-predicted.** Very high AQI (400+) is forecast too low, more so at longer horizons; AQI is also capped at 500.\n"
        "- **Two AQI eras.** 2017-23 AQI is officially reported; 2024-25 AQI is computed by us from PM2.5/PM10. "
        "The levels have not been fully reconciled, though models score similarly in both eras.\n"
        "- **Health link is literature-based.** The burden index uses one Delhi hospital study (six hospitals, news-reported). "
        "It is a relative index; its slope is uncertain by roughly a factor of two.\n"
        "- **Search interest is only exploratory evidence.** It was a weak winter-only association, selected on data that "
        "overlapped the test period, so it is not used for forecasting here.\n"
        "- **Replay, not live.** No live AQI feed is connected. This is not medical advice."
    )
