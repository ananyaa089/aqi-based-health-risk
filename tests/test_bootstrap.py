import numpy as np
import pandas as pd

from conftest import load_script

f11 = load_script("11_ci_and_figures.py")
f13 = load_script("13_uncertainty_eval.py")


def test_boot_ci_brackets_the_mean_and_is_reproducible():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    x = rng.normal(10, 3, 400)
    lo, hi = f11.boot(lambda i: x[i].mean(), dates, n_boot=300)
    assert lo < x.mean() < hi and hi - lo < 2
    lo2, hi2 = f11.boot(lambda i: x[i].mean(), dates, n_boot=300)
    assert (lo, hi) == (lo2, hi2)  # fixed seed


def test_ci_is_wider_for_noisier_data():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    a, b = rng.normal(0, 1, 400), rng.normal(0, 5, 400)
    wa = np.diff(f13.block_ci(lambda i: a[i].mean(), dates))[0]
    wb = np.diff(f13.block_ci(lambda i: b[i].mean(), dates))[0]
    assert wb > wa
