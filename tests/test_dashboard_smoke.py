import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from conftest import ROOT  

STATIONS = ["Anand Vihar", "Bawana", "R K Puram"]


def fake_dashboard(folder):
    rows = []
    for day in pd.date_range("2024-10-30", "2024-11-03"):
        for i, st in enumerate(STATIONS):
            hi = int(day == pd.Timestamp("2024-11-01") and i == 0)  
            base = 450 if hi else 150 + 120 * i
            r = dict(station=st, date=day.strftime("%Y-%m-%d"), aqi_today=base)
            for h in (1, 2, 3):
                r.update({f"pred_h{h}": base + h, f"lo_h{h}": base - 40, f"hi_h{h}": base + 60, f"actual_h{h}": base + 5})
            for T in (300, 400):
                r.update({f"p_sev{T}": 0.9 if hi else 0.1, f"alert{T}": hi, f"alert_hi{T}": hi, f"actual_sev{T}": hi,
                          f"why_sev{T}": "today's AQI (raises risk, +1.00)"})
            rows.append(r)
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder / "dashboard_data.csv", index=False)
    (folder / "dashboard_meta.json").write_text(json.dumps({"trained_through": "2023-12-31"}))


def test_app_renders_all_alert_branches(tmp_path):
    fake_dashboard(tmp_path / "dashboard")
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=90)
    at.run()
    at.text_input[0].set_value(str(tmp_path)).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    labels = [m.label for m in at.metric]
    assert any("400+ alerts" in l for l in labels)
    at.date_input[0].set_value(pd.Timestamp("2024-11-01").date()).run()  
    assert not at.exception, [e.value for e in at.exception]
    assert any("High-confidence severe-pollution alert" in md.value for md in at.markdown)


def test_app_asks_for_reexport_when_columns_are_missing(tmp_path):
    folder = tmp_path / "dashboard"
    fake_dashboard(folder)
    d = pd.read_csv(folder / "dashboard_data.csv").drop(columns=["lo_h1"])
    d.to_csv(folder / "dashboard_data.csv", index=False)
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=90)
    at.run()
    at.text_input[0].set_value(str(tmp_path)).run()
    assert any("older export" in e.value for e in at.error)
