import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(file_name: str):
    spec = importlib.util.spec_from_file_location("mod_" + file_name.replace(".py", "").replace("-", "_"), SCRIPTS / file_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_master(start="2020-01-01", days=400, stations=("Anand Vihar", "R K Puram"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="1h")
    frames = []
    for k, st in enumerate(stations):
        day = np.arange(days)
        noise = np.zeros(days)
        for i in range(1, days):
            noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 20)
        daily = 220 + 90 * np.sin(2 * np.pi * (day + 280) / 365.25) + 15 * k + noise
        aqi = np.repeat(np.clip(daily, 20, 500), 24)
        df = pd.DataFrame({"timestamp": idx, "station": st, "aqi": aqi, "aqi_source": "reported",
                           "temp_c_city": 25.0, "humidity_city": 55.0, "windspeed_ms_city": 2.0,
                           "pressure_hpa_city": 1000.0, "rain_mm_city": 0.0})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def master():
    return make_master()
