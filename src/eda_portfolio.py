"""EDA: portfolio snapshots, default rates, sector breakdown, watchlist."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]


def latest_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the common as-of-date snapshot; never mix reporting dates."""
    data = panel.copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])
    latest_date = data["snapshot_date"].max()
    snapshot = data[data["snapshot_date"] == latest_date].copy()
    if snapshot["obligor_id"].duplicated().any():
        raise ValueError("More than one facility row per obligor in current demo snapshot")
    return snapshot


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values, weights=weights)) if weights.sum() else np.nan


def summarise_portfolio(panel: pd.DataFrame) -> pd.DataFrame:
    snapshot = latest_snapshot(panel)
    rows = []
    for rating in RATING_ORDER:
        group = snapshot[snapshot["rating"] == rating]
        if group.empty:
            continue
        rows.append(
            {
                "rating": rating,
                "count": len(group),
                "total_ead": group["ead"].sum(),
                "wa_pd": _weighted_average(group["pd"], group["ead"]),
                "wa_lgd": _weighted_average(group["lgd"], group["ead"]),
                "total_el": group["expected_loss"].sum(),
                "el_pct_ead": group["expected_loss"].sum() / group["ead"].sum(),
                "ead_share_pct": group["ead"].sum() / snapshot["ead"].sum() * 100,
            }
        )
    return pd.DataFrame(rows)


def sector_breakdown(panel: pd.DataFrame) -> pd.DataFrame:
    snapshot = latest_snapshot(panel)
    result = (
        snapshot.groupby("sector")
        .agg(
            n_obligors=("obligor_id", "nunique"),
            total_ead=("ead", "sum"),
            total_el=("expected_loss", "sum"),
            defaulted_ead=(
                "ead",
                lambda series: series[snapshot.loc[series.index, "default_flag"] == 1].sum(),
            ),
        )
        .reset_index()
    )
    result["el_rate"] = result["total_el"] / result["total_ead"]
    result["ead_share_pct"] = result["total_ead"] / result["total_ead"].sum() * 100
    result["defaulted_ead_pct"] = result["defaulted_ead"] / result["total_ead"] * 100
    return result.sort_values("total_ead", ascending=False).reset_index(drop=True)


def quarterly_default_rate(panel: pd.DataFrame) -> pd.DataFrame:
    """Default events divided by obligors performing at quarter start."""
    data = panel.dropna(subset=["rating_prev"]).copy()
    at_risk = data[data["rating_prev"] != "D"]
    result = (
        at_risk.groupby("snapshot_date")
        .agg(n_at_risk=("obligor_id", "nunique"), n_defaults=("is_new_default", "sum"))
        .reset_index()
    )
    result["default_rate_pct"] = result["n_defaults"] / result["n_at_risk"] * 100
    return result.sort_values("snapshot_date")


def performance_trends(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in panel.groupby("snapshot_date"):
        performing = group[group["rating"] != "D"]
        transitions = group[group["rating_prev"].notna()]
        downgrades = transitions.apply(
            lambda row: RATING_ORDER.index(row["rating"]) > RATING_ORDER.index(row["rating_prev"]),
            axis=1,
        )
        rows.append(
            {
                "snapshot_date": pd.Timestamp(date),
                "n_obligors": group["obligor_id"].nunique(),
                "total_ead": group["ead"].sum(),
                "wa_pd_performing": _weighted_average(performing["pd"], performing["ead"]),
                "el_pct_ead": group["expected_loss"].sum() / group["ead"].sum(),
                "defaulted_ead_pct": group.loc[group["default_flag"] == 1, "ead"].sum() / group["ead"].sum(),
                "downgrade_rate": float(downgrades.mean()) if len(downgrades) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("snapshot_date")


def watchlist(panel: pd.DataFrame) -> pd.DataFrame:
    snapshot = latest_snapshot(panel)
    performing = snapshot[snapshot["rating"] != "D"].copy()
    performing["watch_reason"] = np.select(
        [
            performing["rating"].eq("CCC"),
            performing["rating"].eq("B")
            & ((performing["leverage_ratio"] > 4.0) | (performing["interest_coverage"] < 3.0)),
            performing["interest_coverage"] < 2.0,
            performing["leverage_ratio"] > 5.0,
            performing["rating"].isin(["BB", "B", "CCC"])
            & (performing["maturity_date"] <= performing["snapshot_date"] + pd.DateOffset(months=18)),
        ],
        ["ccc_rating", "weak_b_metrics", "low_interest_coverage", "high_leverage", "near_term_refinancing"],
        default="",
    )
    return performing[performing["watch_reason"] != ""].sort_values("ead", ascending=False)


def plot_ead_by_sector(panel: pd.DataFrame, save_dir: str) -> None:
    sectors = sector_breakdown(panel)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(
        data=sectors, x="total_ead", y="sector", hue="sector", legend=False, palette="Blues_r", ax=axes[0]
    )
    axes[0].set(title="EAD by Sector", xlabel="EAD (£bn)", ylabel="")
    axes[0].set_xticks(axes[0].get_xticks(), [f"{value / 1e9:.1f}" for value in axes[0].get_xticks()])
    axes[1].barh(sectors["sector"], sectors["el_rate"] * 100, color="#B33A3A")
    axes[1].set(title="Expected Loss Rate by Sector", xlabel="EL / EAD (%)", ylabel="")
    fig.suptitle("Current Portfolio Sector Profile", fontsize=14)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "ead_by_sector.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rating_distribution(panel: pd.DataFrame, save_dir: str) -> None:
    snapshot = latest_snapshot(panel)
    counts = snapshot["rating"].value_counts().reindex(RATING_ORDER, fill_value=0)
    ead = snapshot.groupby("rating")["ead"].sum().reindex(RATING_ORDER, fill_value=0)
    colors = [
        "#2678B2" if rating in RATING_ORDER[:4] else "#E5A823" if rating in ["BB", "B"] else "#B33A3A"
        for rating in RATING_ORDER
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(RATING_ORDER, counts, color=colors)
    axes[0].set(title="Obligors by Rating", ylabel="Count")
    axes[1].bar(RATING_ORDER, ead / 1e9, color=colors)
    axes[1].set(title="EAD by Rating", ylabel="EAD (£bn)")
    fig.suptitle("Current Rating Distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "rating_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quarterly_default_rate(panel: pd.DataFrame, save_dir: str) -> None:
    result = quarterly_default_rate(panel)
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(result["snapshot_date"], result["default_rate_pct"], width=60, color="#2678B2")
    axis.axhline(result["default_rate_pct"].mean(), color="#B33A3A", linestyle="--", label="Period average")
    axis.set(
        title="Quarterly New-Default Rate",
        xlabel="Quarter",
        ylabel="Defaults / beginning performing obligors (%)",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "quarterly_default_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_performance_trends(panel: pd.DataFrame, save_dir: str) -> None:
    trend = performance_trends(panel)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    panels = [
        ("total_ead", 1e9, "Total EAD (£bn)"),
        ("wa_pd_performing", 0.01, "Performing WA PD (%)"),
        ("defaulted_ead_pct", 0.01, "Defaulted EAD (%)"),
        ("downgrade_rate", 0.01, "Quarterly downgrade rate (%)"),
    ]
    for axis, (column, scale, title) in zip(axes.flat, panels, strict=True):
        axis.plot(trend["snapshot_date"], trend[column] / scale, marker="o", linewidth=1.8)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.suptitle("Portfolio Performance Trends", fontsize=14)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "portfolio_performance_trends.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_risk_distributions(panel: pd.DataFrame, save_dir: str) -> None:
    snapshot = latest_snapshot(panel)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, values, title, color in [
        (axes[0], snapshot["pd"], "One-Year PD", "#2678B2"),
        (axes[1], snapshot["lgd"], "LGD", "#E5A823"),
        (axes[2], snapshot["ead"] / 1e6, "EAD (£m)", "#3A8E5B"),
    ]:
        axis.hist(values, bins=35, color=color, edgecolor="white")
        axis.axvline(values.median(), color="#B33A3A", linestyle="--", label=f"Median {values.median():.2f}")
        axis.set_title(title)
        axis.legend(fontsize=8)
    fig.suptitle("Current Risk Parameter Distributions")
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "risk_parameter_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(panel: pd.DataFrame, config: dict) -> dict:
    save_dir = config.get("viz", {}).get("output_dir", "reports/figures")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    outputs = {
        "portfolio_summary": summarise_portfolio(panel),
        "sector_summary": sector_breakdown(panel),
        "quarterly_default_rate": quarterly_default_rate(panel),
        "performance_trends": performance_trends(panel),
        "watchlist": watchlist(panel),
    }
    plot_ead_by_sector(panel, save_dir)
    plot_rating_distribution(panel, save_dir)
    plot_quarterly_default_rate(panel, save_dir)
    plot_performance_trends(panel, save_dir)
    plot_risk_distributions(panel, save_dir)
    report_dir = Path(config.get("reports", {}).get("output_dir", "reports"))
    outputs["watchlist"].to_csv(report_dir / "watchlist.csv", index=False)
    outputs["performance_trends"].to_csv(report_dir / "performance_trends.csv", index=False)
    logger.info(
        "Current snapshot %s obligors | watchlist %s", len(latest_snapshot(panel)), len(outputs["watchlist"])
    )
    return outputs
