from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_scaling import fit_power, select_optima


def test_power_law_recovery():
    x = np.logspace(2, 8, 20)
    y = 3.5 * x**0.42
    coefficient, exponent = fit_power(x, y)
    assert np.isclose(coefficient, 3.5)
    assert np.isclose(exponent, 0.42)


def test_select_optima_uses_discrete_minimum():
    rows = [
        {"compute_budget": 10.0, "parameters": 2.0, "final_loss": 3.0},
        {"compute_budget": 10.0, "parameters": 4.0, "final_loss": 2.0},
    ]
    result = select_optima(rows)
    assert result[0]["parameters"] == 4.0
    assert np.isclose(result[0]["tokens"], 10 / 24)
