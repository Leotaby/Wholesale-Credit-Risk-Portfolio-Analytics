"""Build the synthetic quarterly portfolio and persist it to SQLite.

The transition matrix is a published-style synthetic assumption rather than a
vendor or bank calibration. Defaulted obligors remain in the panel while their
exposure runs off, which keeps the reporting-date portfolio internally consistent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from scipy.linalg import fractional_matrix_power
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
RATING_IDX = {rating: idx for idx, rating in enumerate(RATINGS)}

ANNUAL_TRANSITION_MATRIX = np.array(
    [
        [0.9281, 0.0612, 0.0079, 0.0006, 0.0006, 0.0010, 0.0002, 0.0004],
        [0.0059, 0.9100, 0.0768, 0.0052, 0.0010, 0.0006, 0.0002, 0.0003],
        [0.0006, 0.0213, 0.9107, 0.0593, 0.0054, 0.0016, 0.0005, 0.0006],
        [0.0003, 0.0023, 0.0413, 0.8954, 0.0466, 0.0102, 0.0024, 0.0015],
        [0.0002, 0.0007, 0.0042, 0.0598, 0.8601, 0.0622, 0.0065, 0.0063],
        [0.0001, 0.0006, 0.0022, 0.0064, 0.0741, 0.8456, 0.0460, 0.0250],
        [0.0001, 0.0003, 0.0013, 0.0053, 0.0162, 0.0986, 0.7000, 0.1782],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
    ],
    dtype=float,
)


def annual_to_quarterly(matrix: np.ndarray) -> np.ndarray:
    """Return a row-stochastic quarterly matrix whose fourth power is annual."""
    # 4th root of annual matrix gives quarterly probabilities
    quarterly = np.real_if_close(fractional_matrix_power(matrix, 0.25)).real
    quarterly[np.abs(quarterly) < 1e-12] = 0.0
    if quarterly.min() < -1e-8:
        raise ValueError("Annual matrix has no stable non-negative quarterly root")
    quarterly = np.clip(quarterly, 0.0, None)
    quarterly /= quarterly.sum(axis=1, keepdims=True)
    quarterly[-1, :] = 0.0
    quarterly[-1, -1] = 1.0
    return quarterly


BASE_QUARTERLY_MATRIX = annual_to_quarterly(ANNUAL_TRANSITION_MATRIX)

SECTORS = [
    "Energy",
    "Manufacturing",
    "Real Estate",
    "Financial Services",
    "Healthcare",
    "Consumer Discretionary",
    "Technology",
    "Utilities",
    "Telecommunications",
    "Transportation",
]
SECTOR_RISK = {
    "Energy": 1.25,
    "Manufacturing": 1.05,
    "Real Estate": 1.30,
    "Financial Services": 1.00,
    "Healthcare": 0.80,
    "Consumer Discretionary": 1.20,
    "Technology": 0.95,
    "Utilities": 0.70,
    "Telecommunications": 1.00,
    "Transportation": 1.15,
}
GEOGRAPHIES = ["UK", "Germany", "France", "Netherlands", "US", "Other"]
SENIORITIES = ["senior_secured", "senior_unsecured", "subordinated"]
LGD_BY_SENIORITY = {
    "senior_secured": 0.25,
    "senior_unsecured": 0.45,
    "subordinated": 0.70,
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _generate_macro_fallback() -> pd.DataFrame:
    """Deterministic quarterly macro history with a visible 2020 downturn."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2015-01-01", "2023-10-01", freq="QS")
    n_periods = len(dates)

    gdp = rng.normal(2.2, 0.55, n_periods)
    gdp[20:24] = [-5.0, -8.0, 5.5, 4.0]
    unemployment = np.linspace(5.5, 3.6, n_periods)
    unemployment[20:24] = [5.0, 12.0, 8.0, 6.5]
    unemployment += rng.normal(0, 0.12, n_periods)
    fed_funds = np.concatenate(
        [
            np.linspace(0.1, 2.4, 16),
            np.linspace(2.4, 0.1, 5),
            np.full(7, 0.1),
            np.linspace(0.1, 5.3, n_periods - 28),
        ]
    )[:n_periods]
    yield_10y = np.clip(fed_funds + rng.normal(1.2, 0.25, n_periods), 0.4, 5.5)
    inflation = np.concatenate(
        [np.linspace(1.5, 2.1, 20), np.linspace(1.0, 8.5, 9), np.linspace(8.5, 3.2, n_periods - 29)]
    )[:n_periods]
    recession = np.zeros(n_periods, dtype=int)
    recession[20:22] = 1

    return pd.DataFrame(
        {
            "date": dates,
            "gdp_growth_yoy": np.round(gdp, 2),
            "UNRATE": np.round(np.clip(unemployment, 2.0, 15.0), 2),
            "FEDFUNDS": np.round(fed_funds, 2),
            "DGS10": np.round(yield_10y, 2),
            "cpi_yoy": np.round(inflation, 2),
            "USREC": recession,
        }
    )


def fetch_macro_data(config: dict) -> pd.DataFrame:
    """Load transformed FRED series or fall back to the reproducible history."""
    load_dotenv()
    fred_key = os.getenv("FRED_API_KEY") or config.get("data", {}).get("fred_api_key")
    if not fred_key:
        logger.info("FRED_API_KEY not set; using deterministic macro fallback")
        return _generate_macro_fallback()

    try:
        from fredapi import Fred

        fred = Fred(api_key=fred_key)
        start, end = "2014-01-01", "2024-01-01"
        real_gdp = fred.get_series("GDPC1", observation_start=start, observation_end=end)
        cpi = fred.get_series("CPIAUCSL", observation_start=start, observation_end=end)
        data = pd.DataFrame(
            {
                "gdp_growth_yoy": real_gdp.resample("QS").mean().pct_change(4) * 100,
                "UNRATE": fred.get_series("UNRATE", observation_start=start, observation_end=end)
                .resample("QS")
                .mean(),
                "FEDFUNDS": fred.get_series("FEDFUNDS", observation_start=start, observation_end=end)
                .resample("QS")
                .mean(),
                "DGS10": fred.get_series("DGS10", observation_start=start, observation_end=end)
                .resample("QS")
                .mean(),
                "cpi_yoy": cpi.resample("QS").mean().pct_change(4) * 100,
                "USREC": fred.get_series("USREC", observation_start=start, observation_end=end)
                .resample("QS")
                .max(),
            }
        ).dropna()
        data.index.name = "date"
        logger.info("Loaded and transformed %d quarterly FRED observations", len(data))
        return data.reset_index()
    except Exception as exc:
        logger.warning("FRED load failed (%s); using deterministic fallback", exc)
        return _generate_macro_fallback()


def macro_pressure(macro: pd.Series) -> float:
    """Positive values represent an adverse macro environment."""
    pressure = (
        0.55 * (2.0 - float(macro["gdp_growth_yoy"])) / 3.0
        + 0.30 * (float(macro["UNRATE"]) - 4.5) / 3.0
        + 0.15 * (float(macro["FEDFUNDS"]) - 2.0) / 3.0
    )
    return float(np.clip(pressure, -1.5, 2.5))


def _initial_financials(rating: str, rng: np.random.Generator) -> dict[str, float]:
    idx = RATING_IDX[rating]
    return {
        "leverage_ratio": float(np.clip(rng.normal(1.4 + 0.55 * idx, 0.45), 0.2, 12.0)),
        "interest_coverage": float(np.clip(rng.normal(12.0 - 1.45 * idx, 1.4), 0.3, 30.0)),
        "current_ratio": float(np.clip(rng.normal(1.8 - 0.10 * idx, 0.25), 0.3, 5.0)),
        "return_on_assets": float(rng.normal(0.065 - 0.009 * idx, 0.018)),
        "revenue_growth_yoy": float(rng.normal(0.045 - 0.006 * idx, 0.055)),
    }


def _drift_financials(
    financials: dict[str, float], rating: str, pressure: float, rng: np.random.Generator
) -> dict[str, float]:
    idx = RATING_IDX[rating]
    return {
        "leverage_ratio": float(
            np.clip(
                0.88 * financials["leverage_ratio"]
                + 0.12 * (1.4 + 0.55 * idx)
                + 0.08 * pressure
                + rng.normal(0, 0.10),
                0.2,
                15.0,
            )
        ),
        "interest_coverage": float(
            np.clip(
                0.88 * financials["interest_coverage"]
                + 0.12 * (12.0 - 1.45 * idx)
                - 0.25 * pressure
                + rng.normal(0, 0.35),
                0.2,
                30.0,
            )
        ),
        "current_ratio": float(
            np.clip(financials["current_ratio"] - 0.025 * pressure + rng.normal(0, 0.035), 0.25, 5.0)
        ),
        "return_on_assets": float(
            0.85 * financials["return_on_assets"]
            + 0.15 * (0.065 - 0.009 * idx)
            - 0.004 * pressure
            + rng.normal(0, 0.003)
        ),
        "revenue_growth_yoy": float(rng.normal(0.045 - 0.006 * idx - 0.018 * pressure, 0.05)),
    }


def transition_probabilities(
    rating: str,
    sector: str,
    financials: dict[str, float],
    macro: pd.Series,
) -> np.ndarray:
    """Quarterly probabilities conditioned on rating, industry and risk drivers."""
    idx = RATING_IDX[rating]
    if rating == "D":
        result = np.zeros(len(RATINGS))
        result[-1] = 1.0
        return result

    pressure = macro_pressure(macro)
    firm_risk = 0.10 * (financials["leverage_ratio"] - 3.0) - 0.025 * (financials["interest_coverage"] - 5.0)
    combined = float(np.clip(SECTOR_RISK[sector] * pressure + firm_risk, -2.0, 3.0))
    severity_change = np.arange(len(RATINGS)) - idx
    tilt = np.exp(0.28 * combined * np.clip(severity_change, -2, 3))
    tilt[-1] *= np.exp(0.45 * combined)
    probabilities = BASE_QUARTERLY_MATRIX[idx] * tilt
    return probabilities / probabilities.sum()


def generate_panel(
    n_obligors: int = 2500,
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate a quarterly obligor panel with a common reporting date."""
    rng = np.random.default_rng(seed)
    quarters = pd.date_range(start_date, end_date, freq="QS")
    macro = macro_df.copy() if macro_df is not None else _generate_macro_fallback()
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").reindex(quarters).ffill().bfill()

    initial_ratings = rng.choice(RATINGS[:-1], n_obligors, p=[0.03, 0.08, 0.15, 0.28, 0.22, 0.15, 0.09])
    sectors = rng.choice(SECTORS, n_obligors, p=[0.14, 0.13, 0.12, 0.11, 0.09, 0.10, 0.10, 0.08, 0.07, 0.06])
    geographies = rng.choice(GEOGRAPHIES, n_obligors, p=[0.30, 0.18, 0.15, 0.12, 0.15, 0.10])
    seniorities = rng.choice(SENIORITIES, n_obligors, p=[0.55, 0.35, 0.10])
    currencies = rng.choice(["GBP", "EUR", "USD"], n_obligors, p=[0.48, 0.32, 0.20])
    facility_types = rng.choice(["term_loan", "revolver"], n_obligors, p=[0.68, 0.32])
    origination_idx = rng.integers(0, max(1, len(quarters) // 3), n_obligors)
    # Centre the facility distribution on a wholesale-scale £5m median exposure.
    base_ead = rng.lognormal(mean=np.log(5_000_000), sigma=0.75, size=n_obligors)

    rows: list[dict] = []
    for idx in range(n_obligors):
        rating = str(initial_ratings[idx])
        previous_rating: str | None = None
        financials = _initial_financials(rating, rng)
        ead = float(base_ead[idx])
        limit = float(ead * rng.uniform(1.05, 1.45))
        lgd = float(np.clip(LGD_BY_SENIORITY[str(seniorities[idx])] + rng.normal(0, 0.035), 0.05, 0.90))
        origination_date = quarters[origination_idx[idx]]
        # Current facilities include refinancings/extensions; maturities span the
        # seven years following the reporting date rather than expiring mid-panel.
        maturity_date = pd.Timestamp(end_date) + pd.DateOffset(months=int(rng.integers(6, 85)))
        default_age = 0

        for snapshot_date in quarters[origination_idx[idx] :]:
            macro_row = macro.loc[snapshot_date]
            pressure = macro_pressure(macro_row)
            is_new_default = rating == "D" and previous_rating not in (None, "D")

            if rating == "D":
                if default_age > 0:
                    ead *= 0.88
                default_age += 1
                probabilities = np.eye(len(RATINGS))[-1]
                pd_quarterly = pd_1y = 1.0
            else:
                financials = _drift_financials(financials, rating, pressure, rng)
                ead_growth = -0.008 + 0.006 * max(pressure, 0) + rng.normal(0, 0.012)
                ead = float(np.clip(ead * (1 + ead_growth), 0.35 * base_ead[idx], limit))
                probabilities = transition_probabilities(rating, str(sectors[idx]), financials, macro_row)
                pd_quarterly = float(probabilities[-1])
                pd_1y = float(1 - (1 - pd_quarterly) ** 4)

            expected_loss = float(pd_1y * lgd * ead)
            rows.append(
                {
                    "obligor_id": f"OBL-{idx + 1:05d}",
                    "facility_id": f"FAC-{idx + 1:05d}-01",
                    "snapshot_date": snapshot_date.date(),
                    "origination_date": origination_date.date(),
                    "maturity_date": maturity_date.date(),
                    "sector": str(sectors[idx]),
                    "geography": str(geographies[idx]),
                    "currency": str(currencies[idx]),
                    "facility_type": str(facility_types[idx]),
                    "seniority": str(seniorities[idx]),
                    "rating": rating,
                    "rating_prev": previous_rating,
                    "rating_at_origination": str(initial_ratings[idx]),
                    "ead": ead,
                    "ead_limit": limit,
                    "undrawn_amount": max(limit - ead, 0.0),
                    "lgd": lgd,
                    "pd": pd_1y,
                    "pd_quarterly": pd_quarterly,
                    "expected_loss": expected_loss,
                    "default_flag": int(rating == "D"),
                    "is_new_default": int(is_new_default),
                    "macro_pressure": pressure,
                    "sector_risk_score": SECTOR_RISK[str(sectors[idx])],
                    **{name: round(value, 6) for name, value in financials.items()},
                }
            )

            next_rating = str(rng.choice(RATINGS, p=probabilities))
            previous_rating, rating = rating, next_rating

    panel = pd.DataFrame(rows)
    panel["snapshot_date"] = pd.to_datetime(panel["snapshot_date"])
    panel["origination_date"] = pd.to_datetime(panel["origination_date"])
    panel["maturity_date"] = pd.to_datetime(panel["maturity_date"])
    latest = panel[panel["snapshot_date"] == panel["snapshot_date"].max()]
    logger.info(
        "Panel: %s rows | %s obligors | %s quarters | latest EAD £%.2fbn | new defaults %s",
        f"{len(panel):,}",
        f"{panel['obligor_id'].nunique():,}",
        panel["snapshot_date"].nunique(),
        latest["ead"].sum() / 1e9,
        int(panel["is_new_default"].sum()),
    )
    return panel


def build_transition_log(panel: pd.DataFrame) -> pd.DataFrame:
    transitions = panel.dropna(subset=["rating_prev"]).copy()
    transitions["period_start"] = transitions["snapshot_date"] - pd.DateOffset(months=3)
    return transitions[
        [
            "obligor_id",
            "period_start",
            "snapshot_date",
            "rating_prev",
            "rating",
            "is_new_default",
            "sector",
            "ead",
        ]
    ].rename(
        columns={
            "snapshot_date": "period_end",
            "rating_prev": "rating_start",
            "rating": "rating_end",
            "is_new_default": "is_default",
        }
    )


def _create_schema(engine, schema_path: Path) -> None:
    raw = engine.raw_connection()
    try:
        raw.executescript(schema_path.read_text(encoding="utf-8"))
        raw.commit()
    finally:
        raw.close()


def write_to_db(panel: pd.DataFrame, macro: pd.DataFrame, engine) -> None:
    """Write through the declared schema so constraints and indexes remain active."""
    panel_out = panel.copy()
    for column in ["snapshot_date", "origination_date", "maturity_date"]:
        panel_out[column] = pd.to_datetime(panel_out[column]).dt.strftime("%Y-%m-%d")
    transition_out = build_transition_log(panel)
    for column in ["period_start", "period_end"]:
        transition_out[column] = pd.to_datetime(transition_out[column]).dt.strftime("%Y-%m-%d")
    macro_out = macro.copy()
    macro_out["date"] = pd.to_datetime(macro_out["date"]).dt.strftime("%Y-%m-%d")

    panel_out.to_sql("portfolio", engine, if_exists="append", index=False, chunksize=500, method="multi")
    transition_out.to_sql(
        "rating_transitions", engine, if_exists="append", index=False, chunksize=1_000, method="multi"
    )
    macro_out.to_sql("macro_series", engine, if_exists="append", index=False, method="multi")
    logger.info("Persisted panel and %s transition records", f"{len(transition_out):,}")


def run(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build panel and macro data, validate them, and persist to SQLite."""
    Path(config["data"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["data"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    if config["database"].get("engine") != "sqlite":
        raise NotImplementedError("The public demo supports SQLite; use an adapter for PostgreSQL")

    macro = fetch_macro_data(config)
    panel = generate_panel(
        n_obligors=int(config["data"]["synthetic_n_obligors"]),
        start_date=config["data"]["synthetic_start_date"],
        end_date=config["data"]["synthetic_end_date"],
        seed=int(config["project"]["seed"]),
        macro_df=macro,
    )
    if panel.duplicated(["obligor_id", "snapshot_date"]).any():
        raise ValueError("Duplicate obligor-snapshot keys detected")
    if not np.allclose(np.linalg.matrix_power(BASE_QUARTERLY_MATRIX, 4), ANNUAL_TRANSITION_MATRIX, atol=1e-8):
        raise ValueError("Quarterly transition calibration failed")

    engine = create_engine(f"sqlite:///{config['database']['sqlite_path']}")
    _create_schema(engine, Path("sql/schema.sql"))
    write_to_db(panel, macro, engine)
    return panel, macro
