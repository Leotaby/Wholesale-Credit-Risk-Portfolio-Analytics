# Data Dictionary

## `portfolio`

| Field | Definition |
|---|---|
| `obligor_id` | Stable synthetic borrower identifier |
| `facility_id` | Synthetic facility identifier |
| `snapshot_date` | Quarterly beginning-of-period observation date |
| `origination_date`, `maturity_date` | Contract dates |
| `sector`, `geography` | Portfolio segmentation dimensions |
| `currency`, `facility_type`, `seniority` | Facility attributes |
| `rating_prev`, `rating` | Prior and current internal-style rating state |
| `is_new_default` | 1 on the first observed default state, else 0 |
| `pd_quarterly`, `pd` | Quarterly and one-year PD; defaulted names equal 1 |
| `lgd` | Loss given default after seniority/collateral effects |
| `ead` | Exposure at default in GBP-equivalent synthetic units |
| `ead_limit`, `undrawn_amount` | Facility limit and available undrawn commitment |
| `expected_loss` | `pd × lgd × ead` |
| `revenue`, `total_assets`, `total_debt` | Synthetic scale measures |
| `leverage_ratio` | Debt relative to earnings capacity proxy |
| `interest_coverage` | Earnings relative to interest burden |
| `current_ratio` | Current assets relative to current liabilities proxy |
| `return_on_assets` | Profitability ratio |
| `revenue_growth_yoy` | Year-on-year revenue growth proxy |
| `collateral_value` | Current collateral value proxy |

## `macro_series`

| Field | Definition |
|---|---|
| `date` | Quarterly observation date |
| `UNRATE` | Unemployment rate (%) |
| `GDPC1` | Real GDP level when retrieved; normalised synthetic index in fallback |
| `gdp_growth_yoy` | Year-on-year real GDP growth (%) |
| `FEDFUNDS` | Effective federal funds rate (%) |
| `DGS10` | 10-year US Treasury yield (%) |
| `CPIAUCSL` | CPI level/index |
| `cpi_yoy` | Year-on-year CPI growth (%) |
| `USREC` | Recession indicator |

## `rating_transitions`

One row per adjacent-quarter obligor pair: obligor, dates, prior/new rating, default flag, sector and
exposure. A composite foreign key reconciles it to the corresponding portfolio observation. This table
is the SQL audit trail behind migration counts and default-rate controls.

## `stress_results`

Run timestamp, scenario, baseline/stressed expected loss, EAD-normalised loss rates and delta. Repeated
runs append results so scenario output can be reconciled to the generated report.
