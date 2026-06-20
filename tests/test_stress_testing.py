"""
Tests for stress_testing.py
Uses a panel-compatible fixture (each row = one obligor-quarter).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.stress_testing import (
    _stressed_pd,
    aggregate_scenario_results,
    run_scenario,
)


@pytest.fixture
def mini_panel():
    """Small panel-format portfolio (5 obligors, 2 quarters each)."""
    rows = []
    obligors = [
        ("OBL-A", "Energy", "BBB", 0.0025, 0.45, 5_000_000),
        ("OBL-B", "Manufacturing", "BB", 0.0100, 0.45, 2_500_000),
        ("OBL-C", "Healthcare", "A", 0.0010, 0.25, 8_000_000),
        ("OBL-D", "Energy", "B", 0.0400, 0.70, 1_500_000),
        ("OBL-E", "Technology", "BBB", 0.0025, 0.45, 3_000_000),
    ]
    for qtr in pd.date_range("2022-01-01", periods=2, freq="QS"):
        for obl_id, sector, rating, pd_val, lgd, ead in obligors:
            rows.append(
                {
                    "obligor_id": obl_id,
                    "snapshot_date": qtr,
                    "sector": sector,
                    "rating": rating,
                    "ead": float(ead),
                    "lgd": lgd,
                    "pd": pd_val,
                    "expected_loss": pd_val * lgd * ead,
                    "default_flag": 0,
                    "is_new_default": 0,
                }
            )
    return pd.DataFrame(rows)


def test_stressed_pd_rises_under_gdp_shock():
    """A negative GDP shock must INCREASE PD (not decrease it)."""
    pd_base = 0.02
    pd_stress = _stressed_pd(pd_base, "BB", gdp_shock_pct=-3.0, rate_shock_bps=0, unemployment_shock_ppts=0)
    assert pd_stress > pd_base, (
        f"Stressed PD {pd_stress:.4f} should exceed base {pd_base:.4f} under GDP contraction"
    )


def test_stressed_pd_bounded(mini_panel):
    for _, row in mini_panel.drop_duplicates("obligor_id").iterrows():
        s = _stressed_pd(
            row["pd"], row["rating"], gdp_shock_pct=-8.0, rate_shock_bps=400, unemployment_shock_ppts=6.0
        )
        assert 0 < s < 1.0, f"Stressed PD out of (0,1) for {row['obligor_id']}: {s}"


def test_stressed_pd_not_below_base():
    """Even with positive macro shock, PD floor at base (no artificial improvement)."""
    pd_base = 0.01
    s = _stressed_pd(pd_base, "A", gdp_shock_pct=5.0, rate_shock_bps=-100, unemployment_shock_ppts=-1.0)
    assert s >= pd_base - 1e-9


def test_run_scenario_adds_columns(mini_panel):
    result = run_scenario(mini_panel, "test", -2.0, 1.5, 0)
    for col in ["pd_stressed", "el_stressed", "el_delta", "scenario"]:
        assert col in result.columns, f"Missing column: {col}"


def test_adverse_el_delta_non_negative(mini_panel):
    result = run_scenario(mini_panel, "adverse", -4.0, 3.0, 0)
    assert (result["el_delta"] >= -1e-6).all(), "All EL deltas should be ≥ 0 under adverse shock"


def test_aggregate_returns_required_cols(mini_panel):
    results = {
        "baseline": run_scenario(mini_panel, "baseline", 0, 0, 0),
        "recession": run_scenario(mini_panel, "recession", -3.0, 2.0, 0),
    }
    summary = aggregate_scenario_results(results)
    for col in ["scenario", "el_base_pct_ead", "el_stressed_pct_ead", "el_delta_pct_ead"]:
        assert col in summary.columns, f"Missing column: {col}"


def test_baseline_el_close_to_original(mini_panel):
    """Zero-shock scenario: stressed EL ≈ original EL (within 2%)."""
    result = run_scenario(mini_panel, "baseline", 0.0, 0.0, 0)
    orig = mini_panel["expected_loss"].sum()
    stressed = result["el_stressed"].sum()
    rel_diff = abs(stressed - orig) / max(orig, 1)
    assert rel_diff < 0.02, f"Baseline EL drift too large: {rel_diff:.3f}"
