import csv

import pandas as pd

from conftest import load_script

legacy = load_script("parse_legacy_2017_2023.py")
recent = load_script("resample_2024_25.py")


def write_legacy(path):
    hours = [f"{h:02d}:00:00" for h in range(24)]
    rows = [["Year", "2021"],
            ["January-2021"] + hours,
            ["1"] + [str(100 + h) for h in range(24)],
            ["2.0"] + [str(200 + h) if h != 5 else "" for h in range(24)],  
            ['"'], ['"'],
            ["February-2021"] + hours,
            ["31"] + ["50"] * 24,  
            ["1"] + ["300"] * 24]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def test_legacy_wide_format(tmp_path):
    p = tmp_path / "anand vihar legacy.csv"
    write_legacy(p)
    df = legacy.parse_legacy_file(str(p), "Anand Vihar").set_index("timestamp")["aqi"]
    assert df[pd.Timestamp("2021-01-01 00:00")] == 100
    assert df[pd.Timestamp("2021-01-01 23:00")] == 123
    assert df[pd.Timestamp("2021-01-02 00:00")] == 200
    assert pd.Timestamp("2021-01-02 05:00") not in df.index  
    assert df[pd.Timestamp("2021-02-01 12:00")] == 300
    assert not any(t.month == 2 and t.day == 31 for t in df.index)
    assert df.index.is_unique


def test_recent_format_converts_utc_to_ist_and_resamples(tmp_path):
    p = tmp_path / "anand vihar 15 minute.csv"
    ts = ["2024-01-01 00:00:00+0000", "2024-01-01 00:15:00+0000", "2024-01-01 00:30:00+0000", "2024-01-01 00:45:00+0000"]
    df = pd.DataFrame({"Station ID": "x", "Timestamp": ts, "PM2.5 (µg/m³)": [10, 20, 30, 40], "PM10 (µg/m³)": [100, 100, 100, 100]})
    df.to_csv(p, index=False, encoding="cp1252")  
    out = recent.resample_file(str(p), "Anand Vihar").set_index("timestamp")
    assert out.loc[pd.Timestamp("2024-01-01 05:00"), "pm2_5"] == 15
    assert out.loc[pd.Timestamp("2024-01-01 06:00"), "pm2_5"] == 35
    assert out["pm10"].iloc[0] == 100
