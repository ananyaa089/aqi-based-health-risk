# Updated risk analysis (Milestone 2)

| # | Risk | Status | Evidence | Mitigation / what we did |
|---|---|---|---|---|
| R1 | Search interest does not track Delhi respiratory conditions | OCCURRED | Whole-year deseasonalised correlation about 0 or negative; weak winter-only association (held-out r = 0.36, CI [0.03, 0.65], n = 34 weeks) selected on overlapping data | Reframed: AQI forecasting is the core; health risk via literature-based burden index; Trends reported as exploratory |
| R2 | 2017-23 AQI is reported, 2024-25 AQI is computed by us (level shift) | OPEN, contained | Stable-station check shows no composition effect; sensor sanity good (PM10 below PM2.5 in at most 2% of days); models score similarly in both eras | Keep `aqi_source` flag; evaluate eras separately; document the unresolved 2024 level question |
| R3 | Peaks (AQI 400+) under-predicted at longer horizons | OCCURRED | Mean error about -54 (1 day), -101 (3 days), -109 (7 days) on 400+ days | Separate severe-episode classifier; two alert tiers; limit claims to 1-3 days |
| R4 | Alert fatigue from false alarms | OCCURRED | Standard 400+ alert fires on about 26% of station-days at precision 0.52 | Added high-confidence tier (precision 0.83, recall 0.42, about 7.5% of days) |
| R5 | External benchmark (CAMS) unreliable at station level | OCCURRED | Correlation fell from 0.77 (2022) to 0.23 (2025); seasonal bias drifts; adding it gave +8-11% in 2024 but -8 to -13% in 2025 | Reported as negative result; not integrated |
| R6 | Hospital data unavailable | ACCEPTED | No open respiratory admissions series for Delhi | Literature-based exposure-response index, labelled as relative and uncertain |
| R7 | Burden slope comes from one news-reported study | ACCEPTED | Six hospitals; not peer reviewed | Show 50%-150% slope range; present tiers not counts |
| R8 | Not a live system | ACCEPTED | No live AQI feed connected | Replay mode with out-of-sample forecasts; future work: live feed |
| R9 | Station coverage limited to 10 monitors; coordinates approximate | ACCEPTED | Map shows station points only | State clearly; no interpolated "locality" claims |
| R10 | Over-reliance on AI-generated code without understanding | NEW | Individual viva | Each member owns, runs and explains specific scripts; 24 unit tests; one-command rerun |
