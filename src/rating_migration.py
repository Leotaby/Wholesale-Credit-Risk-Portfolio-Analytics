"""Rating migration: empirical quarterly matrix, P^4 annualisation, cohort DRs."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]


def _normalise_counts(counts: pd.DataFrame) -> pd.DataFrame:
    counts = counts.reindex(index=RATING_ORDER, columns=RATING_ORDER, fill_value=0.0)
    for rating in RATING_ORDER:
        if counts.loc[rating].sum() == 0:
            counts.loc[rating, rating] = 1.0
    matrix = counts.div(counts.sum(axis=1), axis=0)
    matrix.loc["D", :] = 0.0
    matrix.loc["D", "D"] = 1.0
    return matrix


def transition_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one row per observed t->t+1 transition, including D->D."""
    if "rating_prev" in panel.columns:
        pairs = panel.dropna(subset=["rating_prev"])[["obligor_id", "rating_prev", "rating"]].rename(
            columns={"rating_prev": "rating_start", "rating": "rating_end"}
        )
        return pairs
    ordered = panel.sort_values(["obligor_id", "snapshot_date"]).copy()
    ordered["rating_start"] = ordered.groupby("obligor_id")["rating"].shift(1)
    return ordered.dropna(subset=["rating_start"])[["obligor_id", "rating_start", "rating"]].rename(
        columns={"rating": "rating_end"}
    )


def build_quarterly_transition_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    pairs = transition_pairs(panel)
    counts = pd.crosstab(pairs["rating_start"], pairs["rating_end"]).astype(float)
    return _normalise_counts(counts)


def annual_transition_matrix(quarterly: pd.DataFrame) -> pd.DataFrame:
    annual = np.linalg.matrix_power(quarterly.loc[RATING_ORDER, RATING_ORDER].values, 4)
    annual = np.clip(annual, 0.0, 1.0)
    annual /= annual.sum(axis=1, keepdims=True)
    return pd.DataFrame(annual, index=RATING_ORDER, columns=RATING_ORDER)


def bootstrap_transition_matrix(
    panel: pd.DataFrame, n_iter: int = 300, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Cluster bootstrap that preserves duplicated obligor draws and histories."""
    pairs = transition_pairs(panel)
    cluster_counts = (
        pairs.groupby(["obligor_id", "rating_start", "rating_end"]).size().rename("n").reset_index()
    )
    obligors = panel["obligor_id"].unique()
    rng = np.random.default_rng(seed)
    matrices = np.empty((n_iter, len(RATING_ORDER), len(RATING_ORDER)))

    for iteration in range(n_iter):
        weights = pd.Series(rng.choice(obligors, len(obligors), replace=True)).value_counts()
        weighted = cluster_counts.join(weights.rename("weight"), on="obligor_id", how="inner")
        weighted["weighted_n"] = weighted["n"] * weighted["weight"]
        counts = weighted.pivot_table(
            index="rating_start",
            columns="rating_end",
            values="weighted_n",
            aggfunc="sum",
            fill_value=0.0,
        )
        matrices[iteration] = annual_transition_matrix(_normalise_counts(counts)).values

    return {
        "mean": pd.DataFrame(matrices.mean(axis=0), index=RATING_ORDER, columns=RATING_ORDER),
        "lower_95": pd.DataFrame(
            np.percentile(matrices, 2.5, axis=0), index=RATING_ORDER, columns=RATING_ORDER
        ),
        "upper_95": pd.DataFrame(
            np.percentile(matrices, 97.5, axis=0), index=RATING_ORDER, columns=RATING_ORDER
        ),
    }


def cumulative_default_rates(panel: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """Cohort default rates using only borrowers observable for each horizon."""
    horizons = horizons or [1, 3, 5]
    data = panel.copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])
    cohort = (
        data.sort_values("snapshot_date")
        .groupby("obligor_id")
        .agg(
            first_date=("snapshot_date", "min"),
            initial_rating=("rating_at_origination", "first"),
        )
        .reset_index()
    )
    defaults = (
        data[data["is_new_default"] == 1].groupby("obligor_id")["snapshot_date"].min().rename("default_date")
    )
    cohort = cohort.join(defaults, on="obligor_id")
    observation_end = data["snapshot_date"].max()

    rows = []
    for rating in RATING_ORDER[:-1]:
        rating_cohort = cohort[cohort["initial_rating"] == rating]
        for horizon in horizons:
            eligible = rating_cohort[
                rating_cohort["first_date"] <= observation_end - pd.DateOffset(years=horizon)
            ]
            horizon_end = eligible["first_date"] + pd.DateOffset(years=horizon)
            defaulted = eligible["default_date"].notna() & (
                eligible["default_date"].values <= horizon_end.values
            )
            rows.append(
                {
                    "rating": rating,
                    "horizon_yrs": horizon,
                    "n_eligible": len(eligible),
                    "n_defaulted": int(defaulted.sum()),
                    "cum_default_rate": float(defaulted.mean()) if len(eligible) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def migration_summary(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rating in RATING_ORDER[:-1]:
        idx = RATING_ORDER.index(rating)
        rows.append(
            {
                "rating": rating,
                "upgrade_rate": annual.loc[rating, RATING_ORDER[:idx]].sum(),
                "stable_rate": annual.loc[rating, rating],
                "downgrade_rate": annual.loc[rating, RATING_ORDER[idx + 1 :]].sum(),
                "default_rate": annual.loc[rating, "D"],
            }
        )
    return pd.DataFrame(rows)


def plot_transition_heatmap(matrix: pd.DataFrame, save_dir: str = "reports/figures") -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2%",
        cmap="YlOrRd",
        linewidths=0.5,
        mask=matrix < 0.0001,
        cbar_kws={"label": "Transition probability"},
        ax=axis,
    )
    axis.set(title="Annual Rating Transition Matrix", xlabel="Rating at End", ylabel="Rating at Start")
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "annual_rating_transition_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_default_rates(cumulative: pd.DataFrame, save_dir: str = "reports/figures") -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    plot_data = cumulative.dropna(subset=["cum_default_rate"])
    for rating, group in plot_data.groupby("rating"):
        axis.plot(group["horizon_yrs"], group["cum_default_rate"] * 100, marker="o", label=rating)
    axis.set(
        title="Cohort Default Rates (Censoring-Adjusted)",
        xlabel="Horizon (years)",
        ylabel="Cumulative default rate (%)",
    )
    axis.legend(title="Origination rating", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "cumulative_default_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(panel: pd.DataFrame, config: dict) -> dict:
    save_dir = config.get("viz", {}).get("output_dir", "reports/figures")
    quarterly = build_quarterly_transition_matrix(panel)
    annual = annual_transition_matrix(quarterly)
    confidence = bootstrap_transition_matrix(
        panel,
        n_iter=int(config.get("rating_migration", {}).get("bootstrap_iterations", 300)),
        seed=int(config.get("project", {}).get("seed", 42)),
    )
    cumulative = cumulative_default_rates(panel)
    summary = migration_summary(annual)
    plot_transition_heatmap(annual, save_dir)
    plot_cumulative_default_rates(cumulative, save_dir)

    report_dir = Path(config.get("reports", {}).get("output_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    annual.to_csv(report_dir / "annual_transition_matrix.csv")
    summary.to_csv(report_dir / "migration_summary.csv", index=False)
    cumulative.to_csv(report_dir / "cumulative_default_rates.csv", index=False)
    logger.info("Annual B->D %.2f%% | CCC->D %.2f%%", annual.loc["B", "D"] * 100, annual.loc["CCC", "D"] * 100)
    return {
        "quarterly_transition": quarterly,
        "annual_transition": annual,
        "bootstrap_ci": confidence,
        "cumulative_default_rates": cumulative,
        "migration_summary": summary,
    }
