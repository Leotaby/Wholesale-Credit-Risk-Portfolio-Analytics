"""Shared plotting utilities for the portfolio analytics pipeline."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

logger = logging.getLogger(__name__)


def apply_style(config: dict) -> None:
    """Apply global matplotlib style from config."""
    style = config.get("viz", {}).get("style", "seaborn-v0_8-whitegrid")
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("ggplot")
    logger.debug("Plot style set to: %s", style)


def plot_macro_trends(
    macro_df: pd.DataFrame,
    save_dir: str = "reports/figures",
) -> None:
    """Four-panel macro dashboard: GDP, unemployment, Fed Funds, CPI."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Macro-Economic Environment: Historical Trends", fontsize=14, y=1.01)

    macro_df["date"] = pd.to_datetime(macro_df["date"])

    panel_config = [
        ("gdp_growth_yoy", "GDP Growth YoY (%)", "steelblue", "GDP Growth"),
        ("UNRATE", "Unemployment Rate (%)", "darkorange", "Unemployment"),
        ("FEDFUNDS", "Fed Funds Rate (%)", "seagreen", "Policy Rate"),
        ("cpi_yoy", "CPI Inflation YoY (%)", "crimson", "CPI Inflation"),
    ]

    for ax, (col, ylabel, color, title) in zip(axes.flat, panel_config, strict=True):
        if col not in macro_df.columns:
            ax.set_visible(False)
            continue
        ax.plot(macro_df["date"], macro_df[col], color=color, linewidth=2)
        ax.fill_between(macro_df["date"], macro_df[col], alpha=0.15, color=color)

        # Shade recession periods if available
        if "USREC" in macro_df.columns:
            rec = macro_df[macro_df["USREC"] == 1]
            if not rec.empty:
                for _, row in rec.iterrows():
                    ax.axvspan(row["date"], row["date"] + pd.DateOffset(months=3), alpha=0.12, color="grey")

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    out = Path(save_dir) / "macro_trends.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def plot_portfolio_el_trend(
    df: pd.DataFrame,
    save_dir: str = "reports/figures",
) -> None:
    """
    Portfolio expected loss trend over snapshot dates.
    Aggregated quarterly and split by IG vs HY rating.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["snapshot_q"] = df["snapshot_date"].dt.to_period("Q").dt.to_timestamp()
    df["grade"] = df["rating"].apply(
        lambda r: "Investment Grade" if r in ("AAA", "AA", "A", "BBB") else "High Yield / NR"
    )

    grp = (
        df.groupby(["snapshot_q", "grade"])
        .agg(total_el=("expected_loss", "sum"), total_ead=("ead", "sum"))
        .reset_index()
    )
    grp["el_rate"] = grp["total_el"] / grp["total_ead"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Portfolio Expected Loss Trend (Quarterly)", fontsize=13)

    for grade, grp_g in grp.groupby("grade"):
        color = "steelblue" if grade == "Investment Grade" else "crimson"
        axes[0].plot(
            grp_g["snapshot_q"],
            grp_g["total_el"] / 1e6,
            label=grade,
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
        )
        axes[1].plot(
            grp_g["snapshot_q"],
            grp_g["el_rate"],
            label=grade,
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
        )

    axes[0].set_ylabel("Expected Loss (£M)")
    axes[0].set_title("Absolute EL")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].set_ylabel("EL / EAD (%)")
    axes[1].set_title("EL Rate")
    axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = Path(save_dir) / "portfolio_el_trend.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def plot_rating_migration_time_series(
    df: pd.DataFrame,
    save_dir: str = "reports/figures",
) -> None:
    """Stacked area chart: rating composition of portfolio over time."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["snapshot_q"] = df["snapshot_date"].dt.to_period("Q").dt.to_timestamp()

    rating_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
    palette = {
        "AAA": "#0D47A1",
        "AA": "#1565C0",
        "A": "#1976D2",
        "BBB": "#42A5F5",
        "BB": "#FFC107",
        "B": "#FF7043",
        "CCC": "#D32F2F",
        "D": "#4A148C",
    }

    # EAD share by rating per quarter
    grp = (
        df.groupby(["snapshot_q", "rating"])["ead"]
        .sum()
        .reset_index()
        .pivot(index="snapshot_q", columns="rating", values="ead")
        .fillna(0)
    )
    grp = grp[[r for r in rating_order if r in grp.columns]]
    grp_pct = grp.div(grp.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = [palette.get(r, "#999999") for r in grp_pct.columns]
    grp_pct.plot(kind="area", stacked=True, ax=ax, color=colors, alpha=0.85)
    ax.set_title("Portfolio Rating Composition Over Time (EAD %)", fontsize=13)
    ax.set_ylabel("Share of EAD (%)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    out = Path(save_dir) / "rating_composition_trend.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def run(df: pd.DataFrame, macro_df: pd.DataFrame, config: dict) -> None:
    """Generate all shared visualisations."""
    save_dir = config.get("viz", {}).get("output_dir", "reports/figures")
    apply_style(config)
    plot_macro_trends(macro_df, save_dir)
    plot_portfolio_el_trend(df, save_dir)
    plot_rating_migration_time_series(df, save_dir)
