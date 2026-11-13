"""Entirely artificial fixture for software correctness tests, not clinical data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def artificial_dataset(n=240, seed=17):
    rng = np.random.default_rng(seed)
    age = rng.integers(35, 78, n)
    bp = np.clip(90 + 0.9 * (age - 35) + rng.normal(0, 16, n), 70, 220).round()
    dia = np.clip(bp - rng.integers(30, 60, n), 35, 150).round()
    cholesterol = rng.integers(1, 4, n)
    glucose = rng.integers(1, 4, n)
    other = rng.normal(0, 0.9, n)
    logit = -0.6 + .035 * (age - 55) + .027 * (bp - 125) + .30 * (cholesterol - 2) + other
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob)
    return pd.DataFrame({
        "age": age, "height": rng.integers(148, 191, n),
        "weight": rng.integers(50, 118, n), "sex": rng.integers(1, 3, n),
        "ap_hi": bp, "ap_lo": dia, "cholesterol": cholesterol,
        "glucose": glucose, "smoking": rng.integers(0, 2, n),
        "alcohol": rng.integers(0, 2, n), "physical_activity": rng.integers(0, 2, n),
        "cardio": y,
    })


@pytest.fixture
def training_frame():
    return artificial_dataset()
