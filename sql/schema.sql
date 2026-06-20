PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS stress_results;
DROP TABLE IF EXISTS rating_transitions;
DROP TABLE IF EXISTS macro_series;
DROP TABLE IF EXISTS portfolio;

CREATE TABLE portfolio (
    obligor_id             TEXT NOT NULL,
    facility_id            TEXT NOT NULL,
    snapshot_date          TEXT NOT NULL,
    origination_date       TEXT NOT NULL,
    maturity_date          TEXT NOT NULL,
    sector                 TEXT NOT NULL,
    geography              TEXT NOT NULL,
    currency               TEXT NOT NULL,
    facility_type          TEXT NOT NULL CHECK (facility_type IN ('term_loan','revolver')),
    seniority              TEXT NOT NULL,
    rating                 TEXT NOT NULL CHECK (rating IN ('AAA','AA','A','BBB','BB','B','CCC','D')),
    rating_prev            TEXT,
    rating_at_origination  TEXT NOT NULL,
    ead                    REAL NOT NULL CHECK (ead >= 0),
    ead_limit              REAL NOT NULL CHECK (ead_limit >= 0),
    undrawn_amount         REAL NOT NULL CHECK (undrawn_amount >= 0),
    lgd                    REAL NOT NULL CHECK (lgd BETWEEN 0 AND 1),
    pd                     REAL NOT NULL CHECK (pd BETWEEN 0 AND 1),
    pd_quarterly           REAL NOT NULL CHECK (pd_quarterly BETWEEN 0 AND 1),
    expected_loss          REAL NOT NULL CHECK (expected_loss >= 0),
    default_flag           INTEGER NOT NULL CHECK (default_flag IN (0,1)),
    is_new_default         INTEGER NOT NULL CHECK (is_new_default IN (0,1)),
    macro_pressure         REAL NOT NULL,
    sector_risk_score      REAL NOT NULL,
    leverage_ratio         REAL,
    interest_coverage      REAL,
    current_ratio          REAL,
    return_on_assets       REAL,
    revenue_growth_yoy     REAL,
    PRIMARY KEY (obligor_id, snapshot_date)
);

CREATE TABLE macro_series (
    date                TEXT PRIMARY KEY,
    gdp_growth_yoy      REAL NOT NULL,
    UNRATE              REAL NOT NULL,
    FEDFUNDS            REAL NOT NULL,
    DGS10               REAL NOT NULL,
    cpi_yoy             REAL NOT NULL,
    USREC               INTEGER NOT NULL CHECK (USREC IN (0,1))
);

CREATE TABLE rating_transitions (
    obligor_id          TEXT NOT NULL,
    period_start        TEXT NOT NULL,
    period_end          TEXT NOT NULL,
    rating_start        TEXT NOT NULL,
    rating_end          TEXT NOT NULL,
    is_default          INTEGER NOT NULL CHECK (is_default IN (0,1)),
    sector              TEXT NOT NULL,
    ead                 REAL NOT NULL CHECK (ead >= 0),
    PRIMARY KEY (obligor_id, period_end),
    FOREIGN KEY (obligor_id, period_end)
        REFERENCES portfolio(obligor_id, snapshot_date)
);

CREATE TABLE stress_results (
    scenario                TEXT NOT NULL,
    total_ead               REAL NOT NULL,
    el_base_abs             REAL NOT NULL,
    el_stressed_abs         REAL NOT NULL,
    el_delta_abs            REAL NOT NULL,
    el_base_pct_ead         REAL NOT NULL,
    el_stressed_pct_ead     REAL NOT NULL,
    el_delta_pct_ead        REAL NOT NULL,
    run_timestamp           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (scenario, run_timestamp)
);

CREATE INDEX idx_portfolio_snapshot_rating ON portfolio(snapshot_date, rating);
CREATE INDEX idx_portfolio_sector_snapshot ON portfolio(sector, snapshot_date);
CREATE INDEX idx_portfolio_default_event ON portfolio(is_new_default, snapshot_date);
CREATE INDEX idx_transition_start_end ON rating_transitions(rating_start, rating_end);
