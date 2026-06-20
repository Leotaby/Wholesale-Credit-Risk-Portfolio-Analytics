"""Controls for portfolio surveillance and model-validation statistics."""

import numpy as np
import pandas as pd
import pytest

from src.data_ingestion import generate_panel
from src.eda_portfolio import (
    latest_snapshot,
    performance_trends,
    quarterly_default_rate,
    sector_breakdown,
    summarise_portfolio,
    watchlist,
)
from src.model_backtesting import (
    calibration_intercept_slope,
    discriminatory_power,
    hosmer_lemeshow_test,
    traffic_light_test,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return generate_panel(300, start_date="2019-01-01", end_date="2023-12-31", seed=17)


def test_current_snapshot_and_summaries_reconcile(panel: pd.DataFrame) -> None:
    current = latest_snapshot(panel)
    rating = summarise_portfolio(panel)
    sector = sector_breakdown(panel)
    assert current["snapshot_date"].nunique() == 1
    assert rating["count"].sum() == len(current)
    assert rating["total_ead"].sum() == pytest.approx(current["ead"].sum())
    assert sector["total_ead"].sum() == pytest.approx(current["ead"].sum())
    assert sector["ead_share_pct"].sum() == pytest.approx(100.0)


def test_latest_snapshot_rejects_duplicate_obligor_rows(panel: pd.DataFrame) -> None:
    current = latest_snapshot(panel)
    duplicate = pd.concat([panel, current.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="More than one facility row"):
        latest_snapshot(duplicate)


def test_default_rates_use_beginning_performing_population(panel: pd.DataFrame) -> None:
    rates = quarterly_default_rate(panel)
    latest_date = rates["snapshot_date"].iloc[-1]
    rows = panel[panel["snapshot_date"].eq(latest_date) & panel["rating_prev"].ne("D")]
    assert rates["n_at_risk"].iloc[-1] == rows["obligor_id"].nunique()
    assert rates["n_defaults"].sum() == panel["is_new_default"].sum()


def test_trends_and_watchlist_are_current_and_valid(panel: pd.DataFrame) -> None:
    current = latest_snapshot(panel)
    trends = performance_trends(panel)
    flagged = watchlist(panel)
    assert len(trends) == panel["snapshot_date"].nunique()
    assert trends["total_ead"].iloc[-1] == pytest.approx(current["ead"].sum())
    assert set(flagged["obligor_id"]).issubset(set(current["obligor_id"]))
    assert flagged["rating"].ne("D").all()
    assert flagged["watch_reason"].ne("").all()


def test_discrimination_and_calibration_statistics() -> None:
    y_true = np.array([0] * 90 + [1] * 10)
    probabilities = np.linspace(0.001, 0.8, 100)
    discrimination = discriminatory_power(y_true, probabilities)
    calibration = calibration_intercept_slope(y_true, probabilities)
    hosmer = hosmer_lemeshow_test(y_true, probabilities, n_groups=5)
    assert discrimination["auc"] == pytest.approx(1.0)
    assert discrimination["gini"] == pytest.approx(1.0)
    assert discrimination["brier_score"] >= 0
    assert np.isfinite(calibration["intercept"])
    assert calibration["slope"] > 0
    assert {"chi2", "df", "p_value", "calibrated"} <= hosmer.keys()


def test_traffic_light_uses_prior_quarter_pd(panel: pd.DataFrame) -> None:
    result = traffic_light_test(panel)
    assert not result.empty
    assert result["rating"].is_unique
    assert result["n_obligor_quarters"].gt(0).all()
    assert result["p_value_underprediction"].between(0, 1).all()
    assert set(result["traffic_light"]) <= {"Green", "Yellow", "Red"}
