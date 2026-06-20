"""OOT PD models: LR, RF, HGB trained on t to predict default at t+1."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Keep every obligor in a single fold so repeated quarterly observations cannot
# leak borrower-specific information across development and validation samples.

NUMERIC_FEATURES = [
    "rating_grade",
    "leverage_ratio",
    "interest_coverage",
    "current_ratio",
    "return_on_assets",
    "revenue_growth_yoy",
    "gdp_growth_yoy_lag1",
    "UNRATE_lag1",
    "FEDFUNDS_lag1",
    "DGS10_lag1",
    "cpi_yoy_lag1",
]
CATEGORICAL_FEATURES = ["sector"]


def prepare_features(
    panel: pd.DataFrame, macro: pd.DataFrame, test_size: float = 0.25
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Use information at t to predict a new default event at t+1."""
    data = panel.sort_values(["obligor_id", "snapshot_date"]).copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])
    # shift(-1): use info at t to predict what happens at t+1
    data["default_next_quarter"] = data.groupby("obligor_id")["is_new_default"].shift(-1)
    data["rating_grade"] = data["rating"].map(
        {rating: idx for idx, rating in enumerate(["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"])}
    )

    macro_data = macro.copy()
    macro_data["snapshot_date"] = pd.to_datetime(macro_data["date"])
    rename = {
        "gdp_growth_yoy": "gdp_growth_yoy_lag1",
        "UNRATE": "UNRATE_lag1",
        "FEDFUNDS": "FEDFUNDS_lag1",
        "DGS10": "DGS10_lag1",
        "cpi_yoy": "cpi_yoy_lag1",
    }
    macro_data = macro_data[["snapshot_date", *rename]].rename(columns=rename)
    data = data.merge(macro_data, on="snapshot_date", how="left", validate="many_to_one")
    data = data[(data["rating"] != "D") & data["default_next_quarter"].notna()].copy()
    data = data.reset_index(drop=True)

    feature_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    features = data[feature_columns]
    target = data["default_next_quarter"].astype(int)
    dates = sorted(data["snapshot_date"].unique())
    cutoff = dates[max(1, int(len(dates) * (1 - test_size)))]
    train_mask = data["snapshot_date"] < cutoff
    test_mask = ~train_mask
    metadata = data[["obligor_id", "snapshot_date", "sector", "rating", "ead"]].copy()

    if target[train_mask].nunique() != 2 or target[test_mask].nunique() != 2:
        raise ValueError(
            "OOT split must contain defaults and non-defaults in train and test; "
            f"train={target[train_mask].value_counts().to_dict()}, "
            f"test={target[test_mask].value_counts().to_dict()}"
        )
    logger.info(
        "Model sample: %s train (%s defaults) | %s OOT (%s defaults) | cutoff %s",
        f"{train_mask.sum():,}",
        int(target[train_mask].sum()),
        f"{test_mask.sum():,}",
        int(target[test_mask].sum()),
        pd.Timestamp(cutoff).date(),
    )
    return features, target, train_mask, test_mask, metadata


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("sector", categorical, CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def build_models(seed: int = 42, model_names: list[str] | None = None) -> dict[str, Pipeline]:
    requested = set(model_names or ["logistic_regression", "random_forest", "hist_gradient_boosting"])
    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("preprocess", _preprocessor()),
                ("classifier", LogisticRegression(max_iter=2_000, C=0.5, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=7,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        max_leaf_nodes=15,
                        learning_rate=0.05,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    models = {name: model for name, model in models.items() if name in requested}
    if "xgboost" in requested:
        try:
            import xgboost as xgb

            models["xgboost"] = Pipeline(
                [
                    ("preprocess", _preprocessor()),
                    (
                        "classifier",
                        xgb.XGBClassifier(
                            n_estimators=250,
                            max_depth=3,
                            learning_rate=0.04,
                            subsample=0.75,
                            colsample_bytree=0.75,
                            reg_alpha=0.1,
                            reg_lambda=1.0,
                            eval_metric="logloss",
                            random_state=seed,
                        ),
                    ),
                ]
            )
        except Exception as exc:
            logger.warning("XGBoost unavailable (%s); portable sklearn models will run", exc)
    unknown = requested - {"logistic_regression", "random_forest", "hist_gradient_boosting", "xgboost"}
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    if not models:
        raise ValueError("At least one available model must be requested")
    return models


def train_models(
    features: pd.DataFrame,
    target: pd.Series,
    train_mask: pd.Series,
    groups: pd.Series,
    seed: int = 42,
    cv_folds: int = 5,
    model_names: list[str] | None = None,
) -> dict:
    x_train, y_train = features[train_mask], target[train_mask]
    group_train = groups[train_mask]
    cross_validator = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fitted, scores = {}, {}
    for name, model in build_models(seed, model_names).items():
        aucs = cross_val_score(
            model,
            x_train,
            y_train,
            groups=group_train,
            cv=cross_validator,
            scoring="roc_auc",
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        fitted[name] = model
        scores[name] = {"cv_auc_mean": float(aucs.mean()), "cv_auc_std": float(aucs.std())}
        logger.info("%s grouped-CV AUC %.3f ± %.3f", name, aucs.mean(), aucs.std())
    return {"models": fitted, "cv_scores": scores}


def evaluate_on_test(
    models: dict[str, Pipeline], features: pd.DataFrame, target: pd.Series, test_mask: pd.Series
) -> dict:
    y_test = target[test_mask]
    event_rate = float(y_test.mean())
    probabilities, metrics = {}, {}
    for name, model in models.items():
        probability = model.predict_proba(features[test_mask])[:, 1]
        probabilities[name] = probability
        average_precision = float(average_precision_score(y_test, probability))
        metrics[name] = {
            "auc": float(roc_auc_score(y_test, probability)),
            "average_precision": average_precision,
            "event_rate": event_rate,
            "average_precision_lift": average_precision / event_rate,
            "brier_score": float(brier_score_loss(y_test, probability)),
        }
        logger.info(
            "%s OOT AUC %.3f | AP %.3f | Brier %.4f",
            name,
            metrics[name]["auc"],
            metrics[name]["average_precision"],
            metrics[name]["brier_score"],
        )
    return {"metrics": metrics, "probabilities": probabilities, "y_test": y_test}


def model_feature_importance(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    seed: int,
) -> pd.DataFrame:
    sample_size = min(5_000, len(features))
    sample = features.sample(sample_size, random_state=seed)
    result = permutation_importance(
        model,
        sample,
        target.loc[sample.index],
        scoring="roc_auc",
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def sector_pd_forecast(
    model: Pipeline,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    test_mask: pd.Series,
    gdp_delta: float = -3.0,
    unemployment_delta: float = 2.0,
) -> pd.DataFrame:
    baseline_features = features[test_mask].copy()
    stressed_features = baseline_features.copy()
    stressed_features["gdp_growth_yoy_lag1"] += gdp_delta
    stressed_features["UNRATE_lag1"] += unemployment_delta
    output = metadata[test_mask].copy()
    output["pd_base"] = model.predict_proba(baseline_features)[:, 1]
    output["pd_stressed"] = model.predict_proba(stressed_features)[:, 1]
    output["pd_uplift"] = output["pd_stressed"] - output["pd_base"]
    summary = (
        output.groupby("sector")
        .apply(
            lambda group: pd.Series(
                {
                    "oot_ead_sum": group["ead"].sum(),
                    "wa_pd_base": np.average(group["pd_base"], weights=group["ead"]),
                    "wa_pd_stressed": np.average(group["pd_stressed"], weights=group["ead"]),
                    "wa_pd_uplift": np.average(group["pd_uplift"], weights=group["ead"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return summary.sort_values("wa_pd_uplift", ascending=False)


def plot_model_performance(cv_scores: dict, oot_metrics: dict, save_dir: str) -> None:
    names = list(oot_metrics)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cv_values = [cv_scores[name]["cv_auc_mean"] for name in names]
    cv_errors = [cv_scores[name]["cv_auc_std"] for name in names]
    oot_values = [oot_metrics[name]["auc"] for name in names]
    axes[0].barh(names, cv_values, xerr=cv_errors, color="#2678B2")
    axes[1].barh(names, oot_values, color="#3A8E5B")
    for axis, values, title in zip(
        axes,
        [cv_values, oot_values],
        ["Grouped Cross-Validation AUC", "Out-of-Time AUC"],
        strict=True,
    ):
        axis.axvline(0.5, color="#B33A3A", linestyle="--")
        axis.set_xlim(0.45, 1.0)
        axis.set_title(title)
        for row, value in enumerate(values):
            axis.text(value + 0.008, row, f"{value:.3f}", va="center", fontsize=9)
    fig.suptitle("PD Model Validation", fontsize=14)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "model_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, save_dir: str) -> None:
    plot_data = importance.head(12).sort_values("importance_mean")
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        plot_data["feature"], plot_data["importance_mean"], xerr=plot_data["importance_std"], color="#2678B2"
    )
    axis.set(title="OOT Permutation Importance", xlabel="Decrease in ROC AUC")
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "model_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(panel: pd.DataFrame, macro: pd.DataFrame, config: dict) -> dict:
    seed = int(config.get("project", {}).get("seed", 42))
    cv_folds = int(config.get("macro_model", {}).get("cv_folds", 5))
    test_size = float(config.get("macro_model", {}).get("test_size", 0.25))
    model_names = config.get("macro_model", {}).get("models")
    save_dir = config.get("viz", {}).get("output_dir", "reports/figures")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    features, target, train_mask, test_mask, metadata = prepare_features(panel, macro, test_size)
    trained = train_models(features, target, train_mask, metadata["obligor_id"], seed, cv_folds, model_names)
    evaluated = evaluate_on_test(trained["models"], features, target, test_mask)
    best_name = max(evaluated["metrics"], key=lambda name: evaluated["metrics"][name]["auc"])
    best_model = trained["models"][best_name]
    importance = model_feature_importance(best_model, features[test_mask], target[test_mask], seed)
    sector_forecast = sector_pd_forecast(best_model, features, metadata, test_mask)

    plot_model_performance(trained["cv_scores"], evaluated["metrics"], save_dir)
    plot_feature_importance(importance, save_dir)
    report_dir = Path(config.get("reports", {}).get("output_dir", "reports"))
    importance.to_csv(report_dir / "model_feature_importance.csv", index=False)
    sector_forecast.to_csv(report_dir / "sector_pd_forecast.csv", index=False)

    return {
        "models": trained["models"],
        "cv_scores": trained["cv_scores"],
        "oot_metrics": evaluated["metrics"],
        "best_model": best_name,
        "feature_matrix": features,
        "target": target,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "metadata": metadata,
        "y_probs_test": evaluated["probabilities"],
        "y_test": evaluated["y_test"],
        "feature_importance": importance,
        "sector_pd_forecast": sector_forecast,
    }
