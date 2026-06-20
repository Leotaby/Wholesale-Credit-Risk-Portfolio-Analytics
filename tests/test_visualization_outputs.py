"""Smoke tests for the charts shipped with the management report."""

from pathlib import Path

from src.data_ingestion import _generate_macro_fallback, generate_panel
from src.visualization import run


def test_shared_visualizations_are_written(tmp_path: Path) -> None:
    macro = _generate_macro_fallback()
    panel = generate_panel(
        80,
        start_date="2019-01-01",
        end_date="2021-12-31",
        seed=31,
        macro_df=macro,
    )
    run(panel, macro, {"viz": {"output_dir": str(tmp_path)}})

    expected = {
        "macro_trends.png",
        "portfolio_el_trend.png",
        "rating_composition_trend.png",
    }
    assert {path.name for path in tmp_path.glob("*.png")} == expected
    assert all((tmp_path / filename).stat().st_size > 10_000 for filename in expected)
