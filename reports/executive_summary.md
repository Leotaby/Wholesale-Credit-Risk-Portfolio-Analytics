# Wholesale Credit Risk Portfolio Analytics: Executive Summary

**As of:** 2023-10-01<br>
**Data:** reproducible synthetic obligor panel; no proprietary information<br>
**Risk appetite status:** **Breaches: watchlist EAD, severe-stress EL uplift**

## Management summary

The £13.48bn portfolio contains 2,500 obligors. Performing
weighted-average one-year PD is 1.34%, performing EL/EAD is 0.46%, and
defaulted EAD represents 4.03% of current exposure. The latest quarterly default
rate is 0.36% (8 events from
2,203 beginning performing names).

The largest sector is **Manufacturing** at 13.7% of EAD;
sector HHI is 0.105. The top 20 names represent 4.7% of EAD.
The rules-based watchlist contains 368 obligors and 15.1% of EAD.

Under severe recession, EL increases from 1.88% to
4.90% of EAD (+3.02ppts). The largest
incremental-EL contributors are Energy, Real Estate, Manufacturing.

## Portfolio dashboard

| Measure | Result | Illustrative limit |
|---|---:|---:|
| Total EAD | £13.48bn | n/a |
| Performing WA PD | 1.34% | n/a |
| WA LGD | 37.2% | n/a |
| Defaulted EAD share | 4.03% | n/a |
| Largest sector share | 13.7% | 20.0% |
| Largest single-name share | 0.35% | 3.0% |
| Watchlist EAD share | 15.1% | 15.0% |
| Severe stress EL uplift | 3.02ppts | 2.5ppts |

## Rating migration and performance

| Annual transition | Probability |
|---|---:|
| BBB->D | 0.40% |
| BB->B | 6.40% |
| B->D | 4.87% |
| CCC->D | 23.68% |

Transition confidence intervals are produced with an obligor-cluster bootstrap. Cumulative default
rates exclude cohorts without sufficient observation time at each horizon.

## Stress results

| Scenario | Baseline EL/EAD | Stressed EL/EAD | Delta |
|---|---:|---:|---:|
| baseline | 1.885% | 1.885% | 0.000ppts |
| mild_recession | 1.885% | 2.735% | 0.850ppts |
| severe_recession | 1.885% | 4.900% | 3.016ppts |
| stagflation | 1.885% | 3.388% | 1.503ppts |

The scenario engine stresses PD, collateral recovery/LGD and revolver utilisation/EAD. Results are
sensitivity estimates, not regulatory or IFRS 9 forecasts.

## PD model validation

Best OOT model: **logistic_regression** AUC 0.915, average precision 0.027, Brier 0.0029. Average precision is 8.9× the OOT event-rate baseline. Calibration intercept 0.796, slope 1.123.

Training uses information at quarter *t* to predict default at *t+1*, grouped CV by obligor, and a
strict final-quarter-block OOT test. 0 rating grades are Yellow/Red in the
heterogeneous-PD calibration test.

## Unstructured risk signals and SQL controls

Text-classification method used: **not run**. The offline heuristic path is deliberately labelled
and its scores are not presented as probabilities. Model-based zero-shot classification is opt-in.

The SQL control layer executed 6 named, version-controlled queries against
51,182 panel rows and 43 populated transition cells.

## Recommended management actions

1. Review the top stressed sectors (Energy, Real Estate, Manufacturing) and the largest watchlist
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
