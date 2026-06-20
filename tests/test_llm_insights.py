"""Controls for explicitly labelled offline unstructured-risk signals."""

import pytest

from src.llm_insights import RISK_LABELS, classify_texts, heuristic_classify, run


def test_heuristic_scores_are_bounded_and_labelled() -> None:
    scores, method = heuristic_classify("Liquidity is thin and a covenant breach may trigger default.")
    assert method == "heuristic"
    assert set(scores) == set(RISK_LABELS)
    assert all(0 <= score <= 1 for score in scores.values())
    assert scores["liquidity risk"] > 0
    assert scores["credit risk"] > 0


def test_classifier_rejects_ambiguous_mode() -> None:
    with pytest.raises(ValueError, match="llm.mode"):
        classify_texts(["sample"], mode="automatic")


def test_offline_run_persists_method_and_outputs(tmp_path) -> None:
    summary = run(
        {
            "llm": {"mode": "heuristic", "score_threshold": 0.25},
            "reports": {"output_dir": str(tmp_path)},
        }
    )
    assert len(summary) == 4
    assert summary["method"].eq("heuristic").all()
    assert (tmp_path / "llm_risk_scores.csv").is_file()
    assert (tmp_path / "llm_risk_summary.csv").is_file()
