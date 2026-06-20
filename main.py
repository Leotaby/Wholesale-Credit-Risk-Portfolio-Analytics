#!/usr/bin/env python3
"""Run the complete wholesale portfolio analytics workflow."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import colorlog
import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, text


def configure_logging(level: str = "INFO") -> None:
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red"},
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wholesale Credit Risk Portfolio Analytics")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    return parser.parse_args()


def run_sql_queries(config: dict) -> dict[str, pd.DataFrame]:
    """Execute named statements from the version-controlled SQL file."""
    query_file = Path("sql/analytics_queries.sql")
    sections = query_file.read_text(encoding="utf-8").split("-- name:")[1:]
    engine = create_engine(f"sqlite:///{config['database']['sqlite_path']}")
    outputs = {}
    with engine.connect() as connection:
        for section in sections:
            name, statement = section.split("\n", 1)
            outputs[name.strip()] = pd.read_sql(text(statement.strip().rstrip(";")), connection)
            logging.getLogger("sql").info("%s: %d rows", name.strip(), len(outputs[name.strip()]))
    return outputs


def persist_stress_results(stress_summary: pd.DataFrame, config: dict) -> None:
    output = stress_summary.copy()
    output["run_timestamp"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    engine = create_engine(f"sqlite:///{config['database']['sqlite_path']}")
    output.to_sql("stress_results", engine, if_exists="append", index=False)


def _fmt(value: float | None, pattern: str = ".2f") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return format(float(value), pattern)


def write_executive_summary(panel: pd.DataFrame, results: dict, sql_results: dict, config: dict) -> Path:
    from src.eda_portfolio import latest_snapshot

    snapshot = latest_snapshot(panel)
    performing = snapshot[snapshot["rating"] != "D"]
    total_ead = snapshot["ead"].sum()
    performing_ead = performing["ead"].sum()
    wa_pd = np.average(performing["pd"], weights=performing["ead"])
    wa_lgd = np.average(snapshot["lgd"], weights=snapshot["ead"])
    performing_el_rate = performing["expected_loss"].sum() / performing_ead
    defaulted_ead_share = snapshot.loc[snapshot["rating"] == "D", "ead"].sum() / total_ead

    eda = results["eda"]
    concentration = results["concentration"]
    migration = results["migration"]
    stress = results["stress"]
    models = results.get("models", {})
    backtest = results.get("backtest", {})
    llm_output = results.get("llm", pd.DataFrame())

    latest_dr = eda["quarterly_default_rate"].iloc[-1]
    sectors = eda["sector_summary"]
    top_sector = sectors.iloc[0]
    top_20_share = concentration["top_20_obligors"]["share_pct"].sum()
    watch = eda["watchlist"]
    watch_share = watch["ead"].sum() / total_ead * 100
    annual_matrix = migration["annual_transition"]
    stress_summary = stress["summary"]
    severe = stress_summary[stress_summary["scenario"] == "severe_recession"].iloc[0]
    baseline = stress_summary[stress_summary["scenario"] == "baseline"].iloc[0]
    vulnerable = stress["sector_sensitivity"].head(3)

    best_model = models.get("best_model")
    best_metrics = models.get("oot_metrics", {}).get(best_model, {}) if best_model else {}
    calibration = backtest.get("calibration_intercept_slope", {}).get(best_model, {}) if best_model else {}
    traffic = backtest.get("traffic_light", pd.DataFrame())
    red_yellow = traffic[traffic["traffic_light"].isin(["Red", "Yellow"])] if not traffic.empty else traffic
    llm_method = (
        llm_output["method"].iloc[0]
        if isinstance(llm_output, pd.DataFrame) and not llm_output.empty
        else "not run"
    )

    limits = config["credit_risk"]["risk_appetite"]
    breaches = []
    if top_sector["ead_share_pct"] > limits["max_sector_share_pct"]:
        breaches.append("sector concentration")
    if concentration["top_20_obligors"]["share_pct"].iloc[0] > limits["max_single_name_share_pct"]:
        breaches.append("single-name concentration")
    if watch_share > limits["max_watchlist_share_pct"]:
        breaches.append("watchlist EAD")
    if severe["el_delta_pct_ead"] > limits["severe_el_uplift_limit_ppts"]:
        breaches.append("severe-stress EL uplift")
    risk_status = "Within illustrative limits" if not breaches else "Breaches: " + ", ".join(breaches)
    stress_table_rows = "\n".join(
        f"| {row.scenario} | {row.el_base_pct_ead:.3f}% | "
        f"{row.el_stressed_pct_ead:.3f}% | {row.el_delta_pct_ead:.3f}ppts |"
        for row in stress_summary.itertuples()
    )
    stress_table = (
        "| Scenario | Baseline EL/EAD | Stressed EL/EAD | Delta |\n|---|---:|---:|---:|\n" + stress_table_rows
    )

    report = f"""# Wholesale Credit Risk Portfolio Analytics: Executive Summary

**As of:** {pd.Timestamp(snapshot["snapshot_date"].max()).date()}<br>
**Data:** reproducible synthetic obligor panel; no proprietary information<br>
**Risk appetite status:** **{risk_status}**

## Management summary

The £{total_ead / 1e9:.2f}bn portfolio contains {snapshot["obligor_id"].nunique():,} obligors. Performing
weighted-average one-year PD is {wa_pd:.2%}, performing EL/EAD is {performing_el_rate:.2%}, and
defaulted EAD represents {defaulted_ead_share:.2%} of current exposure. The latest quarterly default
rate is {latest_dr["default_rate_pct"]:.2f}% ({int(latest_dr["n_defaults"])} events from
{int(latest_dr["n_at_risk"]):,} beginning performing names).

The largest sector is **{top_sector["sector"]}** at {top_sector["ead_share_pct"]:.1f}% of EAD;
sector HHI is {concentration["hhi_sector"]:.3f}. The top 20 names represent {top_20_share:.1f}% of EAD.
The rules-based watchlist contains {len(watch):,} obligors and {watch_share:.1f}% of EAD.

Under severe recession, EL increases from {baseline["el_base_pct_ead"]:.2f}% to
{severe["el_stressed_pct_ead"]:.2f}% of EAD (+{severe["el_delta_pct_ead"]:.2f}ppts). The largest
incremental-EL contributors are {", ".join(vulnerable["sector"].tolist())}.

## Portfolio dashboard

| Measure | Result | Illustrative limit |
|---|---:|---:|
| Total EAD | £{total_ead / 1e9:.2f}bn | n/a |
| Performing WA PD | {wa_pd:.2%} | n/a |
| WA LGD | {wa_lgd:.1%} | n/a |
| Defaulted EAD share | {defaulted_ead_share:.2%} | n/a |
| Largest sector share | {top_sector["ead_share_pct"]:.1f}% | {limits["max_sector_share_pct"]:.1f}% |
| Largest single-name share | {concentration["top_20_obligors"]["share_pct"].iloc[0]:.2f}% | {
        limits["max_single_name_share_pct"]:.1f}% |
| Watchlist EAD share | {watch_share:.1f}% | {limits["max_watchlist_share_pct"]:.1f}% |
| Severe stress EL uplift | {severe["el_delta_pct_ead"]:.2f}ppts | {
        limits["severe_el_uplift_limit_ppts"]:.1f}ppts |

## Rating migration and performance

| Annual transition | Probability |
|---|---:|
| BBB->D | {annual_matrix.loc["BBB", "D"]:.2%} |
| BB->B | {annual_matrix.loc["BB", "B"]:.2%} |
| B->D | {annual_matrix.loc["B", "D"]:.2%} |
| CCC->D | {annual_matrix.loc["CCC", "D"]:.2%} |

Transition confidence intervals are produced with an obligor-cluster bootstrap. Cumulative default
rates exclude cohorts without sufficient observation time at each horizon.

## Stress results

{stress_table}

The scenario engine stresses PD, collateral recovery/LGD and revolver utilisation/EAD. Results are
sensitivity estimates, not regulatory or IFRS 9 forecasts.

## PD model validation

{
        (
            "Best OOT model: **"
            + str(best_model)
            + "** AUC "
            + _fmt(best_metrics.get("auc"), ".3f")
            + ", average precision "
            + _fmt(best_metrics.get("average_precision"), ".3f")
            + ", Brier "
            + _fmt(best_metrics.get("brier_score"), ".4f")
            + ". Average precision is "
            + _fmt(best_metrics.get("average_precision_lift"), ".1f")
            + "× the OOT event-rate baseline. Calibration intercept "
            + _fmt(calibration.get("intercept"), ".3f")
            + ", slope "
            + _fmt(calibration.get("slope"), ".3f")
            + "."
        )
        if best_model
        else "Model stage skipped."
    }

Training uses information at quarter *t* to predict default at *t+1*, grouped CV by obligor, and a
strict final-quarter-block OOT test. {len(red_yellow)} rating grades are Yellow/Red in the
heterogeneous-PD calibration test.

## Unstructured risk signals and SQL controls

Text-classification method used: **{llm_method}**. The offline heuristic path is deliberately labelled
and its scores are not presented as probabilities. Model-based zero-shot classification is opt-in.

The SQL control layer executed {len(sql_results)} named, version-controlled queries against
{len(panel):,} panel rows and {len(sql_results["quarterly_transition_counts"]):,} populated transition cells.

## Recommended management actions

1. Review the top stressed sectors ({", ".join(vulnerable["sector"].tolist())}) and the largest watchlist
   names for refinancing, covenant and collateral mitigants.
2. Monitor B/CCC migration and rating-grade calibration each quarter; investigate any Yellow/Red grade.
3. Apply a documented recalibration or override review if the OOT calibration intercept or slope moves
   outside governance tolerances; discrimination alone is not sufficient for PD use.
4. Keep sector and single-name utilisation against the illustrative appetite thresholds above.
5. Replace synthetic calibration with governed internal histories before any credit decision, limit or
   accounting use.

## Method limitations

- Synthetic transitions start from a published-style annual matrix and are conditioned on simulated
  macro, industry and obligor drivers; they are not calibrated estimates for decision-making.
- Default dependence and contagion are simplified; stress results should not be interpreted as capital
  or IFRS 9 ECL outputs.
- The NLP demonstration uses four labelled examples only. Production use requires a governed corpus,
  precision/recall thresholds, evidence spans, privacy review and human oversight.
"""
    report_path = Path(config["reports"]["output_dir"]) / "executive_summary.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def write_manifest(panel: pd.DataFrame, results: dict, config: dict, elapsed: float) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_version": config["project"]["version"],
        "seed": config["project"]["seed"],
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "panel_rows": len(panel),
        "obligors": int(panel["obligor_id"].nunique()),
        "snapshot_date": str(pd.to_datetime(panel["snapshot_date"]).max().date()),
        "elapsed_seconds": round(elapsed, 2),
        "best_model": results.get("models", {}).get("best_model"),
    }
    path = Path(config["reports"]["output_dir"]) / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    configure_logging(config["project"].get("log_level", "INFO"))
    logger = logging.getLogger("main")
    started = time.time()
    Path(config["reports"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    logger.info("1/8 Data ingestion and SQL persistence")
    from src.data_ingestion import run as ingestion_run

    panel, macro = ingestion_run(config)
    results: dict = {}

    logger.info("2/8 Portfolio performance and emerging risks")
    from src.eda_portfolio import run as eda_run

    results["eda"] = eda_run(panel, config)

    logger.info("3/8 Rating migration")
    from src.rating_migration import run as migration_run

    results["migration"] = migration_run(panel, config)

    logger.info("4/8 Concentration")
    from src.concentration_risk import run as concentration_run
    from src.eda_portfolio import latest_snapshot

    current = latest_snapshot(panel)
    results["concentration"] = concentration_run(current, config)

    logger.info("5/8 Stress testing")
    from src.stress_testing import run as stress_run

    results["stress"] = stress_run(current, config)
    persist_stress_results(results["stress"]["summary"], config)

    if args.skip_models:
        logger.warning("6/8 Model stage skipped by request")
        results["models"], results["backtest"] = {}, {}
    else:
        logger.info("6/8 OOT PD modelling and backtesting")
        from src.macro_industry_models import run as model_run
        from src.model_backtesting import run as backtest_run

        results["models"] = model_run(panel, macro, config)
        results["backtest"] = backtest_run(panel, results["models"], config)

    if args.skip_llm:
        logger.warning("7/8 Unstructured-risk stage skipped by request")
    else:
        logger.info("7/8 Unstructured-risk classification")
        from src.llm_insights import run as llm_run

        results["llm"] = llm_run(config)

    logger.info("8/8 Shared visualisations, SQL controls and reporting")
    from src.visualization import run as visualisation_run

    visualisation_run(panel, macro, config)
    sql_results = run_sql_queries(config)
    report_path = write_executive_summary(panel, results, sql_results, config)
    elapsed = time.time() - started
    write_manifest(panel, results, config, elapsed)
    logger.info("Pipeline complete in %.1fs -> %s", elapsed, report_path)


if __name__ == "__main__":
    main()
