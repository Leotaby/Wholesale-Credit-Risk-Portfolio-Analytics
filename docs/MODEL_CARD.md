# PD Model Card

## Intended use

Quarterly portfolio surveillance and demonstration of champion/challenger validation on synthetic
wholesale obligor data. The output can prioritise analytical review in this project. It must not be
used for approval, pricing, limits, provisioning, regulatory capital or customer treatment.

## Target and population

- Target: first transition to default in quarter `t+1`.
- Unit: performing obligor-quarter at `t`.
- Exclusions: defaulted observations, rows with no observable following quarter.
- Split: first 75% of dates for development; final 25% for out-of-time validation.
- Cross-validation: stratified group folds by obligor within development data.

## Inputs

Rating grade, sector, leverage, interest coverage, current ratio, return on assets, revenue growth and
lagged macro variables (GDP growth, unemployment, policy rate, 10-year yield and CPI growth). The code
uses a preprocessing pipeline with median imputation, standardisation and one-hot encoding.

## Models and selection

Logistic regression, random forest and histogram gradient boosting are core. XGBoost is an optional
extra. OOT ROC AUC selects the displayed champion, but the report also shows average precision and
event-rate lift, Brier score, KS, calibration intercept/slope, calibration plots and grade traffic
lights.

For seed 42, the logistic model is champion with OOT AUC 0.915, average precision 0.027 and Brier score
0.0029. The OOT event rate is approximately 0.003, so average precision is materially above a random
ranking baseline. Calibration intercept 0.796 and slope 1.123 indicate that recalibration/governance
review would be appropriate before use, even though discrimination is strong.

## Risks and controls

| Risk | Control |
|---|---|
| Temporal leakage | `t+1` target is shifted by obligor; final dates are held out |
| Borrower leakage | CV folds are grouped by obligor |
| Rare-event optimism | Average precision and event-rate lift supplement AUC |
| Miscalibration | Intercept, slope, Brier, reliability curves and grade traffic lights |
| Feature instability | OOT permutation importance and scenario sensitivity output |
| Non-reproducibility | Fixed seed, central config, run manifest and CI full-run gate |
| Overstatement | Synthetic-only limitation is repeated in README and management report |

## Monitoring proposal

Track quarterly data completeness, population stability, missingness, rating mix, event rate, AUC,
average precision, Brier score, calibration intercept/slope and grade observed-to-expected ratios.
Pre-agree escalation thresholds, require evidence for overrides, compare champion/challenger drift and
perform annual outcome analysis plus independent validation after any material redevelopment.

## Known limitations

The sample is synthetic, dependencies are simplified, macro paths are not consensus forecasts and
there is no formal parameter-uncertainty or economic-capital layer. Performance numbers demonstrate
code behavior only and must not be interpreted as evidence of real-world predictive performance.
