"""Backtesting: discrimination, calibration, HL test, traffic lights."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)
RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]


def traffic_light_test(panel: pd.DataFrame) -> pd.DataFrame:
    """Compare heterogeneous quarter-ahead PDs with observed events by grade."""
    data = panel.sort_values(["obligor_id", "snapshot_date"]).copy()
    data["pd_start"] = data.groupby("obligor_id")["pd_quarterly"].shift(1)
    observations = data[data["rating_prev"].notna() & (data["rating_prev"] != "D") & data["pd_start"].notna()]
    rows = []
    for rating in RATING_ORDER:
        group = observations[observations["rating_prev"] == rating]
        if group.empty:
            continue
        observed = int(group["is_new_default"].sum())
        expected = float(group["pd_start"].sum())
        variance = float((group["pd_start"] * (1 - group["pd_start"])).sum())
        z_score = (observed - expected) / np.sqrt(max(variance, 1e-12))
        p_value = float(stats.norm.sf(z_score))
        flag = "Green" if p_value > 0.05 else "Yellow" if p_value > 0.001 else "Red"
        rows.append(
            {
                "rating": rating,
                "n_obligor_quarters": len(group),
                "observed_defaults": observed,
                "expected_defaults": expected,
                "observed_rate": observed / len(group),
                "predicted_rate": expected / len(group),
                "observed_expected_ratio": observed / expected if expected else np.nan,
                "z_score": z_score,
                "p_value_underprediction": p_value,
                "traffic_light": flag,
            }
        )
    return pd.DataFrame(rows)


def hosmer_lemeshow_test(y_true: np.ndarray, y_probability: np.ndarray, n_groups: int = 10) -> dict:
    frame = pd.DataFrame({"y": y_true, "p": y_probability})
    frame["bucket"] = pd.qcut(frame["p"], q=n_groups, duplicates="drop")
    statistic, groups_used = 0.0, 0
    for _, group in frame.groupby("bucket", observed=True):
        expected_positive = group["p"].sum()
        expected_negative = len(group) - expected_positive
        if expected_positive < 1 or expected_negative < 1:
            continue
        observed_positive = group["y"].sum()
        statistic += (observed_positive - expected_positive) ** 2 / expected_positive
        statistic += (len(group) - observed_positive - expected_negative) ** 2 / expected_negative
        groups_used += 1
    degrees_freedom = max(groups_used - 2, 1)
    p_value = float(stats.chi2.sf(statistic, degrees_freedom))
    return {"chi2": statistic, "df": degrees_freedom, "p_value": p_value, "calibrated": p_value > 0.05}


def calibration_intercept_slope(y_true: np.ndarray, y_probability: np.ndarray) -> dict:
    predictor = logit(np.clip(y_probability, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)
    model.fit(predictor, y_true)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}


def discriminatory_power(y_true: np.ndarray, y_probability: np.ndarray) -> dict:
    positive = y_probability[y_true == 1]
    negative = y_probability[y_true == 0]
    ks_statistic, ks_p_value = stats.ks_2samp(positive, negative)
    auc = float(roc_auc_score(y_true, y_probability))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "ks_stat": float(ks_statistic),
        "ks_p_value": float(ks_p_value),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
    }


def plot_roc(y_true: np.ndarray, probabilities: dict[str, np.ndarray], save_dir: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    for name, values in probabilities.items():
        false_positive, true_positive, _ = roc_curve(y_true, values)
        axis.plot(
            false_positive,
            true_positive,
            linewidth=2,
            label=f"{name} (AUC {roc_auc_score(y_true, values):.3f})",
        )
    axis.plot([0, 1], [0, 1], "k--")
    axis.set(title="ROC Curves (out-of-time test)", xlabel="False-positive rate", ylabel="True-positive rate")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "roc_curves_oot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, probabilities: dict[str, np.ndarray], save_dir: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    for name, values in probabilities.items():
        CalibrationDisplay.from_predictions(y_true, values, n_bins=8, strategy="quantile", name=name, ax=axis)
    axis.set_title("Calibration (out-of-time test)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "calibration_curves_oot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(panel: pd.DataFrame, model_results: dict, config: dict) -> dict:
    traffic = traffic_light_test(panel)
    y_test = model_results["y_test"].to_numpy()
    probabilities = model_results["y_probs_test"]
    discrimination, hosmer_lemeshow, calibration = {}, {}, {}
    for name, values in probabilities.items():
        discrimination[name] = discriminatory_power(y_test, values)
        hosmer_lemeshow[name] = hosmer_lemeshow_test(y_test, values)
        calibration[name] = calibration_intercept_slope(y_test, values)
        logger.info(
            "%s OOT AUC %.3f | Brier %.4f | calibration intercept %.3f slope %.3f",
            name,
            discrimination[name]["auc"],
            discrimination[name]["brier_score"],
            calibration[name]["intercept"],
            calibration[name]["slope"],
        )
    save_dir = config.get("viz", {}).get("output_dir", "reports/figures")
    plot_roc(y_test, probabilities, save_dir)
    plot_calibration(y_test, probabilities, save_dir)
    report_dir = Path(config.get("reports", {}).get("output_dir", "reports"))
    traffic.to_csv(report_dir / "rating_calibration_traffic_light.csv", index=False)
    pd.DataFrame(discrimination).T.to_csv(report_dir / "model_discrimination.csv")
    pd.DataFrame(calibration).T.to_csv(report_dir / "model_calibration.csv")
    return {
        "traffic_light": traffic,
        "discriminatory_stats": discrimination,
        "hosmer_lemeshow": hosmer_lemeshow,
        "calibration_intercept_slope": calibration,
    }
