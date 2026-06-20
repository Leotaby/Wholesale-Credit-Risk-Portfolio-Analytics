"""Generate dark-theme README visualisations from the pipeline database.

Run after main.py completes:

    python generate_readme_charts.py

Outputs five PNGs to reports/figures/readme_*.png. These are referenced
by README.md and verified by CI.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ── Dark palette ──────────────────────────────────────────────────────────────

BG = "#0d1117"
CARD = "#161b22"
GRID = "#21262d"
TEXT = "#c9d1d9"
TEXT_DIM = "#8b949e"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
AMBER = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": CARD,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "text.color": TEXT,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "legend.facecolor": CARD,
        "legend.edgecolor": GRID,
        "legend.labelcolor": TEXT,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    }
)

OUT = Path("reports/figures")
DB = Path("data/credit_risk.db")
RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(str(DB))


# ── 1. Portfolio performance trends ──────────────────────────────────────────


def chart_performance_trends() -> None:
    conn = _connect()
    try:
        panel = pd.read_sql("SELECT * FROM portfolio", conn)
    finally:
        conn.close()

    panel["snapshot_date"] = pd.to_datetime(panel["snapshot_date"])
    quarters = sorted(panel["snapshot_date"].unique())

    ead_vals, pd_vals, def_share, dr_vals = [], [], [], []
    for q in quarters:
        snap = panel[panel["snapshot_date"] == q]
        perf = snap[snap["rating"] != "D"]
        total_ead = snap["ead"].sum()
        ead_vals.append(total_ead / 1e9)
        wa = np.average(perf["pd"], weights=perf["ead"]) if len(perf) else 0.0
        pd_vals.append(wa * 100)
        def_share.append(snap[snap["rating"] == "D"]["ead"].sum() / total_ead * 100)
        # At-risk denominator: performing at start + new defaults this quarter
        n_new_d = int(snap["is_new_default"].sum())
        n_at_risk = len(perf) + n_new_d
        dr_vals.append(n_new_d / n_at_risk * 100 if n_at_risk else 0.0)

    dates = [pd.Timestamp(q) for q in quarters]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "Portfolio Performance Trends",
        fontsize=18, fontweight="bold", color=TEXT, y=0.98,
    )

    # Total EAD
    ax = axes[0, 0]
    ax.fill_between(dates, ead_vals, alpha=0.15, color=ACCENT)
    ax.plot(dates, ead_vals, color=ACCENT, lw=2.2, marker="o", ms=4)
    ax.set_title("Total EAD")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.0f}bn"))
    ax.set_ylim(bottom=0)

    # WA PD
    ax = axes[0, 1]
    ax.fill_between(dates, pd_vals, alpha=0.15, color=AMBER)
    ax.plot(dates, pd_vals, color=AMBER, lw=2.2, marker="o", ms=4)
    ax.set_title("Performing WA PD")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Defaulted EAD share
    ax = axes[1, 0]
    ax.fill_between(dates, def_share, alpha=0.15, color=RED)
    ax.plot(dates, def_share, color=RED, lw=2.2, marker="o", ms=4)
    ax.set_title("Defaulted EAD Share")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_ylim(bottom=0)

    # Quarterly default rate (at-risk denominator)
    ax = axes[1, 1]
    ax.bar(dates, dr_vals, width=60, color=PURPLE, alpha=0.7, edgecolor=PURPLE)
    ax.set_title("Quarterly Default Rate")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.set_ylim(bottom=0)

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "readme_performance_trends.png")
    plt.close(fig)
    print("  ✓ Performance trends")


# ── 2. Stress test ───────────────────────────────────────────────────────────


def chart_stress_test() -> None:
    conn = _connect()
    try:
        stress = pd.read_sql(
            "SELECT scenario, total_ead, el_base_pct_ead, "
            "el_stressed_pct_ead, el_delta_pct_ead "
            "FROM stress_results ORDER BY rowid DESC LIMIT 4",
            conn,
        )
    finally:
        conn.close()

    order = ["baseline", "mild_recession", "severe_recession", "stagflation"]
    labels = ["Baseline", "Mild\nRecession", "Severe\nRecession", "Stagflation"]
    colors = [ACCENT, AMBER, RED, PURPLE]
    stress = stress.set_index("scenario").reindex(order)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]},
    )
    fig.suptitle(
        "Stress Test: Expected Loss Under Macro Scenarios",
        fontsize=16, fontweight="bold", color=TEXT, y=0.98,
    )

    # Absolute EL
    el_abs = stress["el_stressed_pct_ead"].values / 100 * stress["total_ead"].values / 1e6
    bars = ax1.bar(labels, el_abs, color=colors, width=0.6, alpha=0.85)
    for bar, val in zip(bars, el_abs, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f"£{val:.0f}m", ha="center", va="bottom",
            color=TEXT, fontsize=11, fontweight="bold",
        )
    ax1.set_title("Expected Loss by Scenario", color=TEXT)
    ax1.set_ylabel("Expected Loss (£m)", color=TEXT_DIM)
    ax1.set_ylim(0, max(el_abs) * 1.18)
    ax1.grid(axis="x", visible=False)

    # Incremental EL/EAD: stacked baseline + delta
    base_rate = stress["el_base_pct_ead"].iloc[0]
    deltas = stress["el_delta_pct_ead"].values
    ax2.bar(labels, [base_rate] * 4, color=ACCENT, width=0.6, alpha=0.45, label="Baseline EL")
    ax2.bar(labels, deltas, bottom=base_rate, color=colors, width=0.6, alpha=0.85, label="Incremental")
    for i, (d, _lbl) in enumerate(zip(deltas, labels, strict=True)):
        total = base_rate + d
        ax2.text(
            i, total + 0.08, f"{total:.2f}%",
            ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold",
        )
    ax2.axhline(base_rate, color=TEXT_DIM, ls="--", lw=0.8, alpha=0.5)
    ax2.set_title("EL / EAD Breakdown", color=TEXT)
    ax2.set_ylabel("EL / EAD (%)", color=TEXT_DIM)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.set_ylim(0, stress["el_stressed_pct_ead"].max() * 1.2)
    ax2.grid(axis="x", visible=False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "readme_stress_test.png")
    plt.close(fig)
    print("  ✓ Stress test")


# ── 3. Rating transition matrix ─────────────────────────────────────────────


def chart_transition_matrix() -> None:
    conn = _connect()
    try:
        panel = pd.read_sql("SELECT rating_prev, rating FROM portfolio", conn)
    finally:
        conn.close()

    panel = panel.dropna(subset=["rating_prev"])
    counts = (
        pd.crosstab(panel["rating_prev"], panel["rating"])
        .reindex(index=RATING_ORDER, columns=RATING_ORDER, fill_value=0)
        .astype(float)
    )
    qtr = counts.div(counts.sum(axis=1), axis=0)
    qtr.loc["D", :] = 0.0
    qtr.loc["D", "D"] = 1.0

    annual = np.linalg.matrix_power(qtr.values, 4)
    annual = np.clip(annual, 0.0, 1.0)
    annual /= annual.sum(axis=1, keepdims=True)
    pct = pd.DataFrame(annual * 100, index=RATING_ORDER, columns=RATING_ORDER)

    cmap = LinearSegmentedColormap.from_list(
        "dark_heat", [CARD, "#1a3a5c", "#1f6feb", ACCENT, "#79c0ff"], N=256,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    # Show all values; dim near-zero cells via font color
    annot = pct.map(lambda v: f"{v:.2f}" if v >= 0.005 else "")
    sns.heatmap(
        pct, annot=annot, fmt="", cmap=cmap, ax=ax,
        linewidths=1.5, linecolor=BG,
        cbar_kws={"label": "Probability (%)"},
        annot_kws={"size": 11, "weight": "bold", "color": TEXT},
    )
    ax.set_title("Annual Rating Transition Matrix (%)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Rating at End", fontsize=12, color=TEXT)
    ax.set_ylabel("Rating at Start", fontsize=12, color=TEXT)
    ax.tick_params(colors=TEXT)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT)
    cbar.ax.tick_params(colors=TEXT_DIM)

    fig.tight_layout()
    fig.savefig(OUT / "readme_transition_matrix.png")
    plt.close(fig)
    print("  ✓ Transition matrix")


# ── 4. ROC curves (OOT) ─────────────────────────────────────────────────────


def chart_roc_curves() -> None:
    import sys

    from sklearn.metrics import roc_auc_score, roc_curve

    sys.path.insert(0, ".")
    from src.macro_industry_models import build_models, prepare_features

    conn = _connect()
    try:
        panel = pd.read_sql("SELECT * FROM portfolio", conn)
        macro = pd.read_sql("SELECT * FROM macro_series", conn)
    finally:
        conn.close()

    features, target, train_mask, test_mask, _meta = prepare_features(panel, macro)
    pipelines = build_models(seed=42)

    fig, ax = plt.subplots(figsize=(8, 7))
    palette = [ACCENT, GREEN, AMBER]
    for (name, pipe), color in zip(pipelines.items(), palette, strict=True):
        pipe.fit(features[train_mask], target[train_mask])
        y_prob = pipe.predict_proba(features[test_mask])[:, 1]
        fpr, tpr, _ = roc_curve(target[test_mask], y_prob)
        auc = roc_auc_score(target[test_mask], y_prob)
        label = name.replace("_", " ").title()
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{label} (AUC {auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.06, color=color)

    ax.plot([0, 1], [0, 1], "--", color=TEXT_DIM, lw=1, alpha=0.5)
    ax.set_title("ROC Curves — Out-of-Time Validation", fontsize=16, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(OUT / "readme_roc_curves.png")
    plt.close(fig)
    print("  ✓ ROC curves")


# ── 5. Sector risk profile ──────────────────────────────────────────────────


def chart_sector_profile() -> None:
    conn = _connect()
    try:
        panel = pd.read_sql("SELECT * FROM portfolio", conn)
    finally:
        conn.close()

    panel["snapshot_date"] = pd.to_datetime(panel["snapshot_date"])
    latest = panel["snapshot_date"].max()
    snap = panel[panel["snapshot_date"] == latest].copy()

    agg = (
        snap.groupby("sector")
        .agg(ead=("ead", "sum"), el=("expected_loss", "sum"))
        .sort_values("ead", ascending=True)
    )
    agg["ead_bn"] = agg["ead"] / 1e9
    agg["el_rate"] = agg["el"] / agg["ead"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        "Sector Risk Profile — Current Portfolio",
        fontsize=16, fontweight="bold", color=TEXT, y=0.98,
    )

    y_pos = range(len(agg))

    # EAD bars
    bars1 = ax1.barh(y_pos, agg["ead_bn"], color=ACCENT, alpha=0.8, height=0.65)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(agg.index, fontsize=10)
    ax1.set_title("Exposure at Default", color=TEXT)
    ax1.set_xlabel("EAD (£bn)", color=TEXT_DIM)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.1f}bn"))
    for bar, val in zip(bars1, agg["ead_bn"], strict=True):
        ax1.text(
            val + 0.03, bar.get_y() + bar.get_height() / 2,
            f"£{val:.2f}bn", va="center", color=TEXT_DIM, fontsize=9,
        )
    ax1.grid(axis="y", visible=False)

    # EL rate — continuous color scale from green to red
    el_vals = agg["el_rate"].values
    norm = Normalize(vmin=el_vals.min(), vmax=el_vals.max())
    el_cmap = LinearSegmentedColormap.from_list("el_risk", [GREEN, AMBER, RED], N=256)
    bar_colors = [el_cmap(norm(v)) for v in el_vals]

    bars2 = ax2.barh(y_pos, el_vals, color=bar_colors, alpha=0.85, height=0.65)
    ax2.set_title("Expected Loss Rate", color=TEXT)
    ax2.set_xlabel("EL / EAD (%)", color=TEXT_DIM)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    for bar, val in zip(bars2, el_vals, strict=True):
        ax2.text(
            val + 0.03, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%", va="center", color=TEXT_DIM, fontsize=9,
        )
    ax2.grid(axis="y", visible=False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "readme_sector_profile.png")
    plt.close(fig)
    print("  ✓ Sector profile")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("Generating dark-theme README charts...")
    chart_performance_trends()
    chart_stress_test()
    chart_transition_matrix()
    chart_roc_curves()
    chart_sector_profile()
    print(f"Done — 5 charts saved to {OUT}/")
