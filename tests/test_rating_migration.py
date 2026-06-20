"""
Tests for rating_migration.py
Uses a longitudinal panel (multiple quarters per obligor) — not a static snapshot.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rating_migration import (
    RATING_ORDER,
    annual_transition_matrix,
    bootstrap_transition_matrix,
    build_quarterly_transition_matrix,
    cumulative_default_rates,
)


@pytest.fixture
def panel():
    """
    Minimal longitudinal panel: 4 obligors each observed for 3 quarters
    with controlled rating paths.
    Obligor A: BBB → BB → B
    Obligor B: BB  → BB → D  (defaults in Q3)
    Obligor C: A   → A  → A
    Obligor D: B   → CCC → D (defaults in Q3)
    """
    rows = []
    paths = {
        "OBL-A": ["BBB", "BB", "B"],
        "OBL-B": ["BB", "BB", "D"],
        "OBL-C": ["A", "A", "A"],
        "OBL-D": ["B", "CCC", "D"],
    }
    quarters = pd.date_range("2020-01-01", periods=3, freq="QS")
    for obl, ratings in paths.items():
        for i, (q, r) in enumerate(zip(quarters, ratings, strict=True)):
            rows.append(
                {
                    "obligor_id": obl,
                    "snapshot_date": q,
                    "rating": r,
                    "rating_at_origination": ratings[0],
                    "ead": 1_000_000.0,
                    "lgd": 0.45,
                    "pd": 0.01,
                    "expected_loss": 4500.0,
                    "is_new_default": int(r == "D" and (i == 0 or ratings[i - 1] != "D")),
                    "default_flag": int(r == "D"),
                    "sector": "Energy",
                    "geography": "UK",
                }
            )
    return pd.DataFrame(rows)


def test_quarterly_matrix_shape(panel):
    tm = build_quarterly_transition_matrix(panel)
    assert tm.shape == (len(RATING_ORDER), len(RATING_ORDER))


def test_quarterly_matrix_rows_sum_to_one(panel):
    tm = build_quarterly_transition_matrix(panel)
    for rating in RATING_ORDER:
        if tm.loc[rating].sum() > 0:
            assert abs(tm.loc[rating].sum() - 1.0) < 1e-9, f"Row {rating} sums to {tm.loc[rating].sum()}"


def test_d_is_absorbing(panel):
    tm = build_quarterly_transition_matrix(panel)
    assert tm.loc["D", "D"] == pytest.approx(1.0, abs=1e-9)


def test_downgrade_captured(panel):
    """BBB → BB should appear in the transition matrix."""
    tm = build_quarterly_transition_matrix(panel)
    assert tm.loc["BBB", "BB"] > 0, "BBB→BB downgrade should be positive"


def test_annual_matrix_is_p4(panel):
    tm_q = build_quarterly_transition_matrix(panel)
    tm_ann = annual_transition_matrix(tm_q)
    assert tm_ann.shape == (len(RATING_ORDER), len(RATING_ORDER))
    for r in RATING_ORDER:
        assert abs(tm_ann.loc[r].sum() - 1.0) < 1e-6, f"Annual row {r} not sum=1"


def test_probabilities_non_negative(panel):
    tm = build_quarterly_transition_matrix(panel)
    assert (tm.values >= -1e-12).all()


def test_bootstrap_keys(panel):
    ci = bootstrap_transition_matrix(panel, n_iter=20, seed=0)
    assert set(ci.keys()) == {"mean", "lower_95", "upper_95"}


def test_bootstrap_ci_ordering(panel):
    ci = bootstrap_transition_matrix(panel, n_iter=50, seed=1)
    assert (ci["lower_95"].values <= ci["mean"].values + 1e-9).all()
    assert (ci["mean"].values <= ci["upper_95"].values + 1e-9).all()


def test_cumulative_dr_monotone(panel):
    cum = cumulative_default_rates(panel, horizons=[1, 3, 5])
    for rating, grp in cum.groupby("rating"):
        rates = grp.sort_values("horizon_yrs")["cum_default_rate"].dropna().values
        assert all(rates[i] <= rates[i + 1] + 1e-9 for i in range(len(rates) - 1)), (
            f"Non-monotone cumulative DR for {rating}"
        )


def test_cumulative_dr_bounded(panel):
    cum = cumulative_default_rates(panel, horizons=[1, 3])
    observed = cum["cum_default_rate"].dropna()
    assert (observed >= 0).all()
    assert (observed <= 1).all()
