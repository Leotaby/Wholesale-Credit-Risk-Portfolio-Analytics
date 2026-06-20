"""
Tests for concentration_risk.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.concentration_risk import (
    concentration_by_dimension,
    herfindahl_hirschman_index,
    lorenz_curve,
    top_n_obligors,
)


def _make_panel(sectors, eads):
    """Helper: one-row-per-obligor panel (latest snapshot style)."""
    return pd.DataFrame(
        {
            "obligor_id": [f"O{i}" for i in range(len(sectors))],
            "sector": sectors,
            "geography": ["UK"] * len(sectors),
            "rating": ["BBB"] * len(sectors),
            "ead": [float(e) for e in eads],
        }
    )


@pytest.fixture
def uniform_portfolio():
    return _make_panel(
        sectors=[f"Sector{i % 10}" for i in range(100)],
        eads=[1_000_000.0] * 100,
    )


@pytest.fixture
def skewed_portfolio():
    """One sector has 80% of EAD."""
    sectors = ["Energy"] * 80 + ["Other"] * 20
    eads = [1_000_000.0] * 80 + [250_000.0] * 20
    return _make_panel(sectors, eads)


def test_hhi_uniform_low(uniform_portfolio):
    shares = uniform_portfolio.groupby("sector")["ead"].sum()
    assert herfindahl_hirschman_index(shares) < 0.15


def test_hhi_monopoly():
    shares = pd.Series([1_000_000.0], index=["Energy"])
    assert herfindahl_hirschman_index(shares) == pytest.approx(1.0)


@pytest.mark.parametrize("shares", [pd.Series(dtype=float), pd.Series([0.0, 0.0])])
def test_hhi_rejects_missing_or_zero_exposure(shares):
    with pytest.raises(ValueError, match="positive exposure"):
        herfindahl_hirschman_index(shares)


def test_hhi_skewed_high(skewed_portfolio):
    shares = skewed_portfolio.groupby("sector")["ead"].sum()
    assert herfindahl_hirschman_index(shares) > 0.40


def test_concentration_share_sums_to_100(uniform_portfolio):
    conc = concentration_by_dimension(uniform_portfolio, "sector")
    assert abs(conc["share_pct"].sum() - 100.0) < 1e-6


def test_lorenz_gini_perfect_equality(uniform_portfolio):
    _, _, gini = lorenz_curve(uniform_portfolio)
    assert gini < 0.05, f"Expected Gini ≈ 0 but got {gini:.4f}"


def test_lorenz_gini_bounded():
    df = _make_panel(["S"] * 200, np.random.exponential(1e6, 200))
    _, _, gini = lorenz_curve(df)
    assert 0.0 <= gini <= 1.0


def test_lorenz_x_monotone(uniform_portfolio):
    x, _, _ = lorenz_curve(uniform_portfolio)
    assert all(x[i] <= x[i + 1] for i in range(len(x) - 1))


def test_top_n_count(uniform_portfolio):
    top = top_n_obligors(uniform_portfolio, n=10)
    assert len(top) == 10


def test_top_n_all_cum_share_100(uniform_portfolio):
    top = top_n_obligors(uniform_portfolio, n=len(uniform_portfolio))
    assert abs(top["cum_share_pct"].iloc[-1] - 100.0) < 1e-4
