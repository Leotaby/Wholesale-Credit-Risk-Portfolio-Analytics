"""Economic and structural controls for the synthetic panel."""

import numpy as np

from src.data_ingestion import (
    ANNUAL_TRANSITION_MATRIX,
    BASE_QUARTERLY_MATRIX,
    _generate_macro_fallback,
    generate_panel,
)


def test_quarterly_matrix_reconciles_to_annual():
    reconstructed = np.linalg.matrix_power(BASE_QUARTERLY_MATRIX, 4)
    assert np.allclose(reconstructed, ANNUAL_TRANSITION_MATRIX, atol=1e-8)


def test_panel_has_common_reporting_date_and_unique_keys():
    panel = generate_panel(250, seed=7, macro_df=_generate_macro_fallback())
    latest = panel[panel["snapshot_date"] == panel["snapshot_date"].max()]
    assert latest["obligor_id"].nunique() == 250
    assert not panel.duplicated(["obligor_id", "snapshot_date"]).any()


def test_ead_is_wholesale_scale():
    panel = generate_panel(250, seed=8, macro_df=_generate_macro_fallback())
    latest = panel[panel["snapshot_date"] == panel["snapshot_date"].max()]
    assert 1_000_000 < latest["ead"].median() < 15_000_000
    assert latest["ead"].sum() > 500_000_000


def test_default_state_is_absorbing_and_pd_is_one():
    panel = generate_panel(500, seed=9, macro_df=_generate_macro_fallback())
    default_rows = panel[panel["rating"] == "D"]
    assert not default_rows.empty
    assert (default_rows["pd"] == 1.0).all()
    for _, group in panel.groupby("obligor_id"):
        states = group["rating"].tolist()
        if "D" in states:
            first = states.index("D")
            assert set(states[first:]) == {"D"}


def test_macro_downturn_raises_transition_risk():
    panel = generate_panel(1_000, seed=10, macro_df=_generate_macro_fallback())
    quarterly = panel.dropna(subset=["rating_prev"])
    event_rate = quarterly.groupby("snapshot_date")["is_new_default"].mean()
    assert event_rate.loc["2020-04-01":"2020-10-01"].mean() > event_rate.loc["2022-01-01":"2023-10-01"].mean()
