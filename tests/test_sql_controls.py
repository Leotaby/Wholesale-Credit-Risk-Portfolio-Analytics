"""Relational-schema and named-query integration controls."""

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from main import run_sql_queries
from src.data_ingestion import (
    _create_schema,
    _generate_macro_fallback,
    generate_panel,
    write_to_db,
)


def test_schema_queries_and_foreign_keys_reconcile(tmp_path: Path) -> None:
    macro = _generate_macro_fallback()
    panel = generate_panel(300, seed=29, macro_df=macro)
    database = tmp_path / "credit_risk.db"
    engine = create_engine(f"sqlite:///{database}")
    _create_schema(engine, Path("sql/schema.sql"))
    write_to_db(panel, macro, engine)

    outputs = run_sql_queries({"database": {"sqlite_path": str(database)}})
    assert set(outputs) == {
        "current_portfolio_by_rating",
        "current_sector_concentration",
        "quarterly_default_trend",
        "current_watchlist",
        "quarterly_transition_counts",
        "stress_results_latest",
    }
    assert outputs["quarterly_default_trend"]["n_defaults"].sum() == panel["is_new_default"].sum()
    assert outputs["current_sector_concentration"]["ead_share_pct"].sum() == pytest.approx(100.0)
    assert outputs["quarterly_transition_counts"]["n_transitions"].sum() == (
        panel["rating_prev"].notna().sum()
    )

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        indexes = connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'").fetchone()[0]
        assert indexes >= 4
