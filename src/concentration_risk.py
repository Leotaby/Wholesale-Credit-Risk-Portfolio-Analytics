"""Concentration metrics: HHI, Lorenz/Gini, sector/name breakdowns."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Resolve lazily so NumPy builds that remove ``trapz`` still import cleanly.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def herfindahl_hirschman_index(shares: pd.Series) -> float:
    """
    HHI = sum of squared market-share fractions.
    0 = perfectly diversified; 1 = single obligor.
    Industry rule-of-thumb: HHI > 0.25 = highly concentrated.
    """
    if shares.empty or shares.sum() <= 0:
        raise ValueError("HHI requires positive exposure shares")
    s = shares / shares.sum()  # normalise first just in case
    return float((s**2).sum())


def concentration_by_dimension(
    df: pd.DataFrame,
    dimension: str,
    ead_col: str = "ead",
) -> pd.DataFrame:
    """
    Return a sorted table with EAD share and HHI contribution per category.
    """
    grp = df.groupby(dimension)[ead_col].sum().reset_index()
    grp.columns = [dimension, "total_ead"]
    total = grp["total_ead"].sum()
    grp["share_pct"] = grp["total_ead"] / total * 100
    grp["hhi_contribution"] = (grp["total_ead"] / total) ** 2
    grp = grp.sort_values("total_ead", ascending=False).reset_index(drop=True)
    hhi = grp["hhi_contribution"].sum()
    logger.info(
        "HHI [%s]: %.4f (%s)", dimension, hhi, "HIGH" if hhi > 0.25 else "MODERATE" if hhi > 0.15 else "LOW"
    )
    return grp


def top_n_obligors(
    df: pd.DataFrame,
    n: int = 20,
    ead_col: str = "ead",
) -> pd.DataFrame:
    """Return the top-N obligors by EAD and their cumulative share."""
    top = df.groupby("obligor_id")[ead_col].sum().sort_values(ascending=False).head(n).reset_index()
    total = df[ead_col].sum()
    top.columns = ["obligor_id", "ead"]
    top["share_pct"] = top["ead"] / total * 100
    top["cum_share_pct"] = top["share_pct"].cumsum()
    return top


def lorenz_curve(df: pd.DataFrame, ead_col: str = "ead") -> tuple:
    """
    Compute Lorenz curve (population CDF vs EAD CDF).
    Returns (x, y) arrays and the Gini coefficient.
    """
    sorted_ead = np.sort(df[ead_col].values)
    n = len(sorted_ead)
    cum_ead = np.cumsum(sorted_ead)
    x = np.arange(1, n + 1) / n
    y = cum_ead / cum_ead[-1]
    # Gini = 1 - 2 * area under Lorenz curve
    gini = 1 - 2 * _trapezoid(y, x)
    return x, y, gini


def plot_concentration_heatmap(
    df: pd.DataFrame,
    save_dir: str = "reports/figures",
) -> None:
    """Sector × Rating EAD heatmap."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    pivot = df.pivot_table(values="ead", index="sector", columns="rating", aggfunc="sum", fill_value=0)

    rating_order = [r for r in ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"] if r in pivot.columns]
    pivot = pivot[rating_order]
    pivot = pivot.div(1e9)  # scale to £bn

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        linewidths=0.4,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "EAD (£bn)"},
    )
    ax.set_title("Sector × Rating Concentration Heatmap (EAD, £bn)", fontsize=13)
    ax.set_xlabel("Credit Rating")
    ax.set_ylabel("Sector")
    plt.tight_layout()
    out = Path(save_dir) / "concentration_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def plot_lorenz_curve(
    df: pd.DataFrame,
    save_dir: str = "reports/figures",
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    x, y, gini = lorenz_curve(df)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(x * 100, y * 100, color="steelblue", linewidth=2, label="Lorenz curve")
    ax.plot([0, 100], [0, 100], "k--", linewidth=1, label="Perfect equality")
    ax.fill_between(x * 100, y * 100, x * 100, alpha=0.15, color="steelblue")
    ax.set_xlabel("Cumulative % of Obligors")
    ax.set_ylabel("Cumulative % of EAD")
    ax.set_title(f"Lorenz Curve: Portfolio EAD Concentration\nGini = {gini:.3f}", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    out = Path(save_dir) / "lorenz_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s  (Gini=%.3f)", out, gini)


def plot_top_n_obligors(
    top_n: pd.DataFrame,
    save_dir: str = "reports/figures",
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.bar(range(len(top_n)), top_n["ead"] / 1e6, color="steelblue", alpha=0.8, label="EAD (£M)")
    ax1.set_ylabel("EAD (£M)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xticks(range(len(top_n)))
    ax1.set_xticklabels(top_n["obligor_id"], rotation=45, ha="right", fontsize=7)

    ax2 = ax1.twinx()
    ax2.plot(
        range(len(top_n)),
        top_n["cum_share_pct"],
        color="darkorange",
        marker="o",
        linewidth=2,
        label="Cumulative share",
    )
    ax2.set_ylabel("Cumulative EAD Share (%)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, 100)

    ax1.set_title(f"Top-{len(top_n)} Obligors by EAD", fontsize=13)
    plt.tight_layout()
    out = Path(save_dir) / "top_n_obligors.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def run(df: pd.DataFrame, config: dict) -> dict:
    save_dir = config.get("viz", {}).get("output_dir", "reports/figures")

    sector_conc = concentration_by_dimension(df, "sector")
    rating_conc = concentration_by_dimension(df, "rating")
    geo_conc = concentration_by_dimension(df, "geography")

    top_20 = top_n_obligors(df, n=20)

    plot_concentration_heatmap(df, save_dir)
    plot_lorenz_curve(df, save_dir)
    plot_top_n_obligors(top_20, save_dir)

    hhi_sector = herfindahl_hirschman_index(sector_conc["total_ead"])
    _, _, gini = lorenz_curve(df)

    logger.info("Summary: HHI (sector)=%.4f  Gini=%.3f", hhi_sector, gini)

    return {
        "sector_concentration": sector_conc,
        "rating_concentration": rating_conc,
        "geo_concentration": geo_conc,
        "top_20_obligors": top_20,
        "hhi_sector": hhi_sector,
        "gini": gini,
    }
