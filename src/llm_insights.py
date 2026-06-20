"""Extract structured risk flags from short credit-related text.

The default path is a deterministic keyword heuristic so the core project runs
offline. BART zero-shot classification is optional and must be selected in the
configuration. Outputs always retain the method because heuristic hit ratios
and model confidence scores are not directly comparable.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Risk taxonomy ─────────────────────────────────────────────────────────────

RISK_LABELS = [
    "liquidity risk",
    "credit risk",
    "market risk",
    "operational risk",
    "regulatory / compliance risk",
    "refinancing / leverage risk",
    "geopolitical risk",
    "sector-specific risk",
]

# keyword map used when transformers unavailable
HEURISTIC_MAP = {
    "liquidity risk": [
        "cash flow",
        "liquidity",
        "covenant breach",
        "going concern",
        "working capital",
        "revolving credit",
    ],
    "credit risk": [
        "default",
        "non-payment",
        "counterparty",
        "credit quality",
        "impairment",
        "write-off",
    ],
    "market risk": [
        "interest rate",
        "fx",
        "currency",
        "commodity",
        "volatility",
        "market risk",
    ],
    "operational risk": [
        "cyber",
        "fraud",
        "operational",
        "system failure",
        "data breach",
        "supply chain disruption",
    ],
    "regulatory / compliance risk": [
        "regulation",
        "compliance",
        "gdpr",
        "sanctions",
        "investigation",
        "antitrust",
        "litigation",
    ],
    "refinancing / leverage risk": [
        "refinancing",
        "maturity wall",
        "leverage",
        "debt service",
        "high yield",
        "covenant",
    ],
    "geopolitical risk": [
        "geopolitical",
        "trade war",
        "tariff",
        "sanctions",
        "conflict",
        "political instability",
    ],
    "sector-specific risk": [
        "energy transition",
        "commodity price",
        "sector headwinds",
        "disruption",
        "structural decline",
    ],
}

# ── Sample texts (bundled demo) ───────────────────────────────────────────────

SAMPLE_TEXTS = [
    {
        "obligor_id": "OBL-00042",
        "source": "10-K Risk Factor (Energy Co.)",
        "text": (
            "The Company faces significant refinancing risk as $2.5bn of senior "
            "unsecured notes mature in Q3 2026. Prevailing credit markets and "
            "elevated interest rates may result in substantially higher borrowing "
            "costs or an inability to refinance on acceptable terms. Additionally, "
            "the Company's revolving credit facility contains a net leverage covenant "
            "that, if breached, could trigger an event of default."
        ),
    },
    {
        "obligor_id": "OBL-00118",
        "source": "Analyst Note (Manufacturing Sector)",
        "text": (
            "Supply chain disruptions stemming from the ongoing geopolitical conflict "
            "have materially impacted input cost inflation. Operating margins have "
            "compressed by ~350bps YoY, and management acknowledged a risk of covenant "
            "breach on the leverage test if EBITDA does not recover in H2. "
            "We view liquidity headroom as thin at just 1.2x EBITDA in available facilities."
        ),
    },
    {
        "obligor_id": "OBL-00305",
        "source": "News Article (Healthcare)",
        "text": (
            "Shares of MedCorp fell 12% after the company disclosed a data breach "
            "affecting approximately 800,000 patient records. The company now faces "
            "potential GDPR fines of up to €20M and class-action litigation. "
            "Operational disruption to its billing systems is expected to delay "
            "cash collection by 4-6 weeks."
        ),
    },
    {
        "obligor_id": "OBL-00789",
        "source": "Credit Memo (Real Estate)",
        "text": (
            "The borrower operates a portfolio of commercial office assets in Central "
            "London. Occupancy rates have declined to 78% as hybrid working trends persist. "
            "Refinancing risk is elevated given the maturity of the senior facility "
            "in 18 months; the loan-to-value ratio has drifted to 73%, approaching "
            "the 75% LTV covenant threshold."
        ),
    },
]


# ── Classification methods ────────────────────────────────────────────────────


def heuristic_classify(text: str) -> tuple[dict[str, float], str]:
    """
    Keyword-based risk scoring.  Returns ({label: score}, method_tag).
    Score = keyword-hit ratio (0-1) per category. NOT a probability.
    """
    text_lower = text.lower()
    scores = {}
    for label, keywords in HEURISTIC_MAP.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[label] = round(hits / len(keywords), 3)
    return scores, "heuristic"


def bart_classify(
    texts: list[str],
    labels: list[str] = RISK_LABELS,
    model_name: str = "facebook/bart-large-mnli",
) -> tuple[list[dict[str, float]], str]:
    """
    Zero-shot classification via BART.
    Returns (list of score dicts, method_tag='bart_zero_shot').
    Raises RuntimeError if transformers / torch cannot be imported.
    """
    from transformers import pipeline as hf_pipeline  # may ImportError

    logger.info("Loading zero-shot model: %s (CPU only)", model_name)
    classifier = hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,
    )
    results = []
    for text in texts:
        out = classifier(text, candidate_labels=labels, multi_label=True)
        results.append(dict(zip(out["labels"], out["scores"], strict=True)))
    return results, "bart_zero_shot"


def classify_texts(
    texts: list[str],
    mode: str = "heuristic",
    model_name: str = "facebook/bart-large-mnli",
) -> tuple[list[dict[str, float]], str]:
    """
    Try BART; fall back to heuristic if unavailable.
    Logs clearly which method was actually used.
    """
    if mode == "heuristic":
        return [heuristic_classify(text)[0] for text in texts], "heuristic"
    if mode != "bart_zero_shot":
        raise ValueError("llm.mode must be 'heuristic' or 'bart_zero_shot'")
    try:
        scores, method = bart_classify(texts, model_name=model_name)
        logger.info("Classification method: %s", method)
        return scores, method
    except Exception as exc:
        logger.warning(
            "BART model unavailable (%s). "
            "Falling back to keyword heuristic (scores are keyword-hit ratios, "
            "NOT probabilities.  Label results as 'heuristic' in all outputs.",
            exc,
        )
        scores = [heuristic_classify(t)[0] for t in texts]
        return scores, "heuristic"


# ── Summary / reporting ───────────────────────────────────────────────────────


def summarise_risk_flags(
    texts_df: pd.DataFrame,
    score_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Produce a clean per-obligor risk summary.

    For 'bart_zero_shot' results the threshold is a NLI confidence score.
    For 'heuristic' results the threshold is a keyword-hit ratio.
    These are NOT directly comparable; check the 'method' column.
    """
    risk_cols = [c for c in texts_df.columns if c in RISK_LABELS]
    rows = []
    for _, row in texts_df.iterrows():
        flags = [c for c in risk_cols if row[c] >= score_threshold]
        top = max(risk_cols, key=lambda c: row[c]) if risk_cols else "N/A"
        rows.append(
            {
                "obligor_id": row["obligor_id"],
                "source": row["source"],
                "method": row.get("method", "unknown"),
                "top_risk": top,
                "flagged_risks": ", ".join(flags) if flags else "None",
                "risk_flag_count": len(flags),
            }
        )
    return pd.DataFrame(rows)


# ── Pipeline entry point ──────────────────────────────────────────────────────


def run(config: dict) -> pd.DataFrame:
    """
    Classify SAMPLE_TEXTS, produce risk summary, persist to reports/.

    Returns: summary DataFrame with 'method' column indicating which
    classification path was taken ('bart_zero_shot' or 'heuristic').
    """
    logger.info("LLM risk extraction: %d sample texts", len(SAMPLE_TEXTS))

    llm_config = config.get("llm", {})
    mode = llm_config.get("mode", "heuristic")
    model_name = llm_config.get("model_name", "facebook/bart-large-mnli")
    threshold = float(llm_config.get("score_threshold", 0.25))
    texts = [t["text"] for t in SAMPLE_TEXTS]
    scores_list, method = classify_texts(texts, mode=mode, model_name=model_name)

    records = []
    for meta, scores in zip(SAMPLE_TEXTS, scores_list, strict=True):
        row = {**meta, "method": method, **scores}
        records.append(row)

    df = pd.DataFrame(records)
    summary = summarise_risk_flags(df, score_threshold=threshold)

    logger.info(
        "Risk extraction complete (method=%s). Top risks per obligor:\n%s",
        method,
        summary[["obligor_id", "method", "top_risk", "flagged_risks"]].to_string(index=False),
    )

    if method == "heuristic":
        logger.warning(
            "Heuristic fallback was used. Scores are keyword-hit ratios, not "
            "NLI confidence scores. Treat outputs as indicative only."
        )

    out_dir = Path(config.get("reports", {}).get("output_dir", "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "llm_risk_scores.csv", index=False)
    summary.to_csv(out_dir / "llm_risk_summary.csv", index=False)
    logger.info("Saved llm_risk_scores.csv and llm_risk_summary.csv to %s", out_dir)

    return summary
