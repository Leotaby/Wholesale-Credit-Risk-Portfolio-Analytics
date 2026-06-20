"""Stress testing: PD, LGD and EAD channels across macro scenarios."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit

logger = logging.getLogger(__name__)

PD_GDP_BETA = {
    "AAA": -0.05,
    "AA": -0.08,
    "A": -0.10,
    "BBB": -0.16,
    "BB": -0.24,
    "B": -0.34,
    "CCC": -0.42,
    "D": 0.0,
}
PD_RATE_BETA = {
    "AAA": 0.01,
    "AA": 0.015,
    "A": 0.02,
    "BBB": 0.035,
    "BB": 0.065,
    "B": 0.10,
    "CCC": 0.14,
    "D": 0.0,
}
SECTOR_SENSITIVITY = {
    "Energy": 1.20,
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


def _stressed_pd(
    pd_base: float,
    rating: str,
    gdp_shock_pct: float,
    rate_shock_bps: float,
    unemployment_shock_ppts: float,
    sector_multiplier: float = 1.0,
) -> float:
    if rating == "D" or pd_base >= 1.0:
        return 1.0
    if gdp_shock_pct == 0 and rate_shock_bps == 0 and unemployment_shock_ppts == 0:
        return float(pd_base)
    base = float(np.clip(pd_base, 1e-8, 1 - 1e-8))
    beta_gdp = PD_GDP_BETA.get(rating, -0.18)
    beta_rate = PD_RATE_BETA.get(rating, 0.04)
    beta_unemployment = abs(beta_gdp) * 0.35
    logit_shift = sector_multiplier * (
        beta_gdp * gdp_shock_pct
        + beta_rate * rate_shock_bps / 100
        + beta_unemployment * unemployment_shock_ppts
    )
    return float(np.clip(expit(logit(base) + logit_shift), base, 0.999999))


def _scenario_severity(gdp: float, unemployment: float, rates_bps: float) -> float:
    return float(np.clip(max(-gdp, 0) / 5 + max(unemployment, 0) / 4 + max(rates_bps, 0) / 300, 0, 3))


def run_scenario(
    portfolio: pd.DataFrame,
    scenario_name: str,
    gdp_shock_pct: float,
    unemployment_shock_ppts: float,
    rate_shock_bps: float,
) -> pd.DataFrame:
    result = portfolio.copy()
    baseline = gdp_shock_pct == unemployment_shock_ppts == rate_shock_bps == 0
    severity = _scenario_severity(gdp_shock_pct, unemployment_shock_ppts, rate_shock_bps)
    multipliers = result["sector"].map(SECTOR_SENSITIVITY).fillna(1.0)
    result["pd_stressed"] = [
        _stressed_pd(pd_value, rating, gdp_shock_pct, rate_shock_bps, unemployment_shock_ppts, multiplier)
        for pd_value, rating, multiplier in zip(result["pd"], result["rating"], multipliers, strict=True)
    ]

    if baseline:
        result["lgd_stressed"] = result["lgd"]
        result["ead_stressed"] = result["ead"]
    else:
        collateral_haircut = 0.035 * severity * multipliers
        result["lgd_stressed"] = np.clip(result["lgd"] + collateral_haircut, result["lgd"], 0.95)
        revolver_draw = np.where(
            result.get("facility_type", "term_loan") == "revolver",
            result.get("undrawn_amount", 0.0) * min(0.10 * severity, 0.35),
            0.0,
        )
        result["ead_stressed"] = result["ead"] + revolver_draw

    result["el_stressed"] = result["pd_stressed"] * result["lgd_stressed"] * result["ead_stressed"]
    result["el_delta"] = result["el_stressed"] - result["expected_loss"]
    if baseline:
        result["el_stressed"] = result["expected_loss"]
        result["el_delta"] = 0.0
    result["scenario"] = scenario_name
    return result


def aggregate_scenario_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for scenario, detail in results.items():
        total_ead = detail["ead"].sum()
        base_el = detail["expected_loss"].sum()
        stressed_el = detail["el_stressed"].sum()
        rows.append(
            {
                "scenario": scenario,
                "total_ead": total_ead,
                "el_base_abs": base_el,
                "el_stressed_abs": stressed_el,
                "el_delta_abs": stressed_el - base_el,
                "el_base_pct_ead": base_el / total_ead * 100,
                "el_stressed_pct_ead": stressed_el / total_ead * 100,
                "el_delta_pct_ead": (stressed_el - base_el) / total_ead * 100,
            }
        )
    return pd.DataFrame(rows)


def sector_sensitivity(results: dict[str, pd.DataFrame], scenario: str = "severe_recession") -> pd.DataFrame:
    detail = results[scenario]
    output = (
        detail.groupby("sector")
        .agg(base_el=("expected_loss", "sum"), stressed_el=("el_stressed", "sum"), ead=("ead", "sum"))
        .reset_index()
    )
    output["el_delta"] = output["stressed_el"] - output["base_el"]
    output["el_delta_pct_ead"] = output["el_delta"] / output["ead"] * 100
    return output.sort_values("el_delta", ascending=False)


def plot_scenario_comparison(summary: pd.DataFrame, save_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#2678B2", "#E5A823", "#B33A3A", "#7652A8"][: len(summary)]
    values = summary["el_stressed_abs"] / 1e6
    bars = axes[0].bar(summary["scenario"], values, color=colors)
    axes[0].set(title="Expected Loss by Scenario", ylabel="Expected loss (£m)")
    offset = max(values.max() * 0.015, 0.05)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + offset, f"£{value:,.0f}m", ha="center")
    axes[1].bar(summary["scenario"], summary["el_stressed_pct_ead"], color=colors)
    axes[1].set(title="Expected Loss Rate", ylabel="EL / EAD (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Wholesale Credit Stress Test", fontsize=14)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "stress_test_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sector_sensitivity(sectors: pd.DataFrame, save_dir: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    ordered = sectors.sort_values("el_delta")
    axis.barh(ordered["sector"], ordered["el_delta"] / 1e6, color="#B33A3A")
    axis.set(title="Severe Recession: Incremental EL by Sector", xlabel="Incremental expected loss (£m)")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "sector_el_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(portfolio: pd.DataFrame, config: dict) -> dict:
    scenarios = config.get("stress_testing", {}).get("scenarios", {})
    details = {
        name: run_scenario(
            portfolio,
            name,
            float(parameters.get("gdp_shock_pct", 0)),
            float(parameters.get("unemployment_shock_ppts", 0)),
            float(parameters.get("rate_shock_bps", 0)),
        )
        for name, parameters in scenarios.items()
    }
    summary = aggregate_scenario_results(details)
    sectors = sector_sensitivity(details)
    save_dir = config.get("viz", {}).get("output_dir", "reports/figures")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plot_scenario_comparison(summary, save_dir)
    plot_sector_sensitivity(sectors, save_dir)
    report_dir = Path(config.get("reports", {}).get("output_dir", "reports"))
    summary.to_csv(report_dir / "stress_summary.csv", index=False)
    sectors.to_csv(report_dir / "stress_sector_sensitivity.csv", index=False)
    logger.info(
        "Stress summary:\n%s",
        summary[["scenario", "el_stressed_pct_ead", "el_delta_pct_ead"]].to_string(index=False),
    )
    return {"summary": summary, "scenario_details": details, "sector_sensitivity": sectors}
