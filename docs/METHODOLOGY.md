# Methodology and Analytical Controls

## Scope

This is a synthetic wholesale credit portfolio demonstrator. It is intended to show how a portfolio
analytics workflow can preserve event timing, exposure weighting, validation discipline and
management traceability. It is not a regulatory capital, IFRS 9/CECL or underwriting model.

## Portfolio construction

The generator creates obligors and facilities with sector, geography, seniority, currency, maturity,
financial ratios, limits, utilisation, PD, LGD and EAD. Current EAD is calibrated to a wholesale-scale
portfolio rather than retail-sized balances. Defaults are absorbing; post-default EAD runs off toward
a common reporting date.

The stated annual rating transition matrix is converted to a quarterly matrix with a matrix fourth
root. Small numerical artifacts are clipped and rows are renormalised. Tests require `P_quarter^4`
to reproduce the annual matrix within tolerance. Macro pressure, sector risk and obligor financials
then tilt the transition probabilities while retaining a valid row-stochastic distribution.

The panel row is the state at the beginning of a quarter. A transition is applied after that state is
recorded, and `is_new_default` is marked on the first row carrying the default state. This convention
keeps transition pairs, default denominators and predictive features aligned.

## Risk measures

- `PD`: one-year probability of default for performing names; 100% for defaulted names.
- `LGD`: downturn-style loss severity conditioned on seniority, collateral and scenario.
- `EAD`: drawn plus credit-conversion-weighted undrawn exposure, with scenario utilisation uplift for
  revolving facilities.
- `Expected loss = PD × LGD × EAD`.
- Concentration HHI is the sum of squared exposure shares. Name Gini and Lorenz curves supplement HHI.

Quarterly default rate is new defaults divided by performing obligors at the start of the interval.
Defaulted names are not recycled into the denominator. Cumulative default curves exclude cohorts
without sufficient observation time at a given horizon, avoiding right-censoring bias from treating
unobserved future quarters as survival.

## Migration estimation

Observed adjacent-quarter pairs are used to estimate the empirical quarterly matrix. The annual
matrix is the fourth power of that estimate. Confidence intervals use an obligor-cluster bootstrap:
an obligor is the resampling unit, repeated draws receive multiplicity weights, and complete histories
remain intact. This addresses within-obligor serial dependence better than resampling individual rows.

## Stress testing

Four deterministic scenarios pass through three channels:

1. PD multipliers from GDP contraction, unemployment and rates, with rating and sector sensitivity.
2. LGD uplift from lower collateral recovery and sector vulnerability.
3. EAD uplift from stressed revolver utilisation.

Baseline is an exact reconciliation to portfolio EL. Scenario outputs include total EL/EAD, change
from baseline and sector contribution. The engine is a sensitivity tool; it does not estimate scenario
probabilities, portfolio credit VaR, contagion or management actions.

## Macro-linked PD model

Features observed at quarter `t` predict a new default at `t+1`. Defaulted states and terminal rows are
removed. The final 25% of dates form a strict out-of-time set; training cross-validation is grouped by
obligor to prevent the same borrower appearing in train and validation folds.

The portable champion/challenger set is logistic regression, random forest and histogram gradient
boosting. XGBoost is optional. Champion selection uses OOT ROC AUC, while average precision, event-rate
lift, Brier score, KS and calibration diagnostics are reported to prevent AUC-only selection.

Permutation importance is computed on the OOT sample. A simple industry sensitivity forecast applies
a -3 percentage point GDP shock and +2 point unemployment shock to the champion pipeline; this is a
conditional sensitivity, not a causal macroeconomic forecast.

## Validation and governance

- Calibration-in-the-large and slope are estimated from the logit of predicted PD.
- A heterogeneous-PD normal approximation compares grade-level observed defaults with the sum of
  individual PDs and variance `Σ p(1-p)`.
- Hosmer-Lemeshow is included as a diagnostic, not treated as a standalone pass/fail test.
- CI runs lint, tests with branch coverage, and the complete offline pipeline on Python 3.10 and 3.11.
- The run manifest captures seed, versions, row counts, as-of date, elapsed time and selected model.

## Material limitations

- Synthetic calibration and simulated dependence cannot support a credit decision.
- Macro and financial-ratio relationships are intentionally simplified and not causal estimates.
- There is no competing-risk, cure, restructuring, multi-facility netting or guarantor hierarchy model.
- Stress output omits portfolio default correlation, second-order contagion, FX shocks and management
  actions.
- The heuristic text classifier is a demonstrator; production use needs labelled data, evidence spans,
  precision/recall monitoring, privacy controls and human review.
- SQLite/pandas is appropriate for a portable demonstration, not a substitute for governed bank data
  platforms, lineage, entitlements and independent model validation.
