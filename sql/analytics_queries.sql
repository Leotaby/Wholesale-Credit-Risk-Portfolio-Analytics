-- name: current_portfolio_by_rating
WITH current_snapshot AS (
    SELECT * FROM portfolio
    WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM portfolio)
)
SELECT
    rating,
    COUNT(DISTINCT obligor_id) AS n_obligors,
    SUM(ead) AS total_ead,
    SUM(ead * pd) / NULLIF(SUM(ead), 0) AS wa_pd,
    SUM(expected_loss) / NULLIF(SUM(ead), 0) AS el_rate
FROM current_snapshot
GROUP BY rating
ORDER BY CASE rating
    WHEN 'AAA' THEN 1 WHEN 'AA' THEN 2 WHEN 'A' THEN 3 WHEN 'BBB' THEN 4
    WHEN 'BB' THEN 5 WHEN 'B' THEN 6 WHEN 'CCC' THEN 7 WHEN 'D' THEN 8 END;

-- name: current_sector_concentration
WITH current_snapshot AS (
    SELECT * FROM portfolio
    WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM portfolio)
), total AS (
    SELECT SUM(ead) AS total_ead FROM current_snapshot
)
SELECT
    sector,
    COUNT(DISTINCT obligor_id) AS n_obligors,
    SUM(ead) AS total_ead,
    100.0 * SUM(ead) / (SELECT total_ead FROM total) AS ead_share_pct,
    POWER(SUM(ead) / (SELECT total_ead FROM total), 2) AS hhi_contribution
FROM current_snapshot
GROUP BY sector
ORDER BY total_ead DESC;

-- name: quarterly_default_trend
SELECT
    snapshot_date,
    COUNT(DISTINCT obligor_id) AS n_at_risk,
    SUM(is_new_default) AS n_defaults,
    100.0 * SUM(is_new_default) / COUNT(DISTINCT obligor_id) AS default_rate_pct
FROM portfolio
WHERE rating_prev IS NOT NULL AND rating_prev <> 'D'
GROUP BY snapshot_date
ORDER BY snapshot_date;

-- name: current_watchlist
WITH current_snapshot AS (
    SELECT * FROM portfolio
    WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM portfolio)
)
SELECT
    obligor_id,
    sector,
    rating,
    ead,
    leverage_ratio,
    interest_coverage,
    maturity_date
FROM current_snapshot
WHERE rating <> 'D'
  AND (
      rating = 'CCC'
      OR (rating = 'B' AND (leverage_ratio > 4.0 OR interest_coverage < 3.0))
      OR leverage_ratio > 5.0
      OR interest_coverage < 2.0
      OR (rating IN ('BB','B','CCC') AND maturity_date <= DATE(snapshot_date, '+18 months'))
  )
ORDER BY ead DESC
LIMIT 50;

-- name: quarterly_transition_counts
SELECT
    rating_start,
    rating_end,
    COUNT(*) AS n_transitions,
    SUM(is_default) AS n_defaults,
    SUM(ead) AS transition_ead
FROM rating_transitions
GROUP BY rating_start, rating_end
ORDER BY rating_start, rating_end;

-- name: stress_results_latest
SELECT scenario, el_base_pct_ead, el_stressed_pct_ead, el_delta_pct_ead
FROM stress_results
WHERE run_timestamp = (SELECT MAX(run_timestamp) FROM stress_results)
ORDER BY scenario;
