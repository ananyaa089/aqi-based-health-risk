# Test cases and validation evidence

Run: `python -m pytest tests -v`. Result on the team's machine (Windows, Python 3.11.9, pytest 9.1.1, 5 Nov 2026): **24 passed, 0 failed.**
All tests use small synthetic inputs, so they test the code's logic, not the real data. Real-data validation is in the report (time-ordered test blocks, confidence intervals, calibration).

| ID | Module under test | What is checked | Expected result | Result |
|---|---|---|---|---|
| TC-01 | AQI formula | PM2.5 at each CPCB breakpoint (30/60/90/120/250/380) | AQI 50/100/200/300/400/500 | Pass |
| TC-02 | AQI formula | PM10 at each breakpoint (50/100/250/350/430/510) | AQI 50/100/200/300/400/500 | Pass |
| TC-03 | AQI formula | AQI is the max of the PM2.5 and PM10 sub-indices | PM10-driven 200; PM2.5-driven 400 | Pass |
| TC-04 | AQI formula | Concentrations above the top breakpoint | Clamped to 500 | Pass |
| TC-05 | AQI formula | Minimum 16 of 24 hours required | First 15 hours missing, then defined | Pass |
| TC-06 | AQI formula | Linear interpolation between breakpoints | PM2.5 = 45 gives AQI 75 | Pass |
| TC-07 | AQI formula | 24-hour rolling mean | AQI follows the rolled-over mean after a level change | Pass |
| TC-08 | Legacy parser | Wide month-block format, day label "2.0", blank hour, invalid date (31 Feb) | Correct timestamps and values; blanks skipped, invalid dates dropped, no duplicates | Pass |
| TC-09 | 2024-25 resampler | UTC to IST conversion, 15-minute to hourly mean, cp1252 header with unit symbols | 00:00Z maps to 05:30 IST; bin means 15 and 35 | Pass |
| TC-10 | Feature builder | Lags align with true calendar days | aqi_lag0/1/7 equal the daily means 0/1/7 days earlier; 3-day-max label uses days t+1..t+3 only | Pass |
| TC-11 | Feature builder | Regression target is h days ahead | y_h1 and y_h3 equal the AQI 1 and 3 days later | Pass |
| TC-12 | Leakage guard | Perturb AQI on one future day | Every feature of every earlier date is unchanged | Pass |
| TC-13 | Daily panel | Day with fewer than 12 valid hours | Day treated as missing | Pass |
| TC-14 | Time-ordered split | Train/tune/test blocks and horizon | Test rows only in the test window; target_date = date + 3; model beats climatology on a seasonal series | Pass |
| TC-15 | Alert threshold (standard tier) | F2-based threshold | Threshold in (0,1); recall higher than precision | Pass |
| TC-16 | Alert threshold (high-confidence tier) | Precision target reachable / unreachable | Reaches 80% precision when possible; falls back to 0.5 otherwise | Pass |
| TC-17 | Explanations | Friendly feature names; top-driver text; merged sin/cos | Correct names; one "time of year" driver; correct ordering | Pass |
| TC-18 | Burden conversion | AQI to PM2.5 inverse | Matches the breakpoints; clipped at 500 | Pass |
| TC-19 | Burden conversion | Excess admissions with literature slope 0.709/ug/m3, 60 ug/m3 reference | Zero below reference; (90-60) x 0.709 at AQI 200; scales with slope; monotone | Pass |
| TC-20 | Burden tiers | Tier boundaries | none / moderate / very high at 0 / 30 / 200 | Pass |
| TC-21 | Bootstrap CI | Brackets the sample mean; reproducible | Mean inside the interval; identical with the fixed seed | Pass |
| TC-22 | Bootstrap CI | Noisier data gives a wider interval | Width increases with noise | Pass |
| TC-23 | Dashboard | Renders all alert branches, including a high-confidence alert | No exception; headline metrics shown; high-confidence alert text appears | Pass |
| TC-24 | Dashboard | Old export missing new columns | Clear "re-run export" error instead of a crash | Pass |

## Real-data validation (see report, not unit tests)
- Forecast error vs persistence and climatology, 95% block-bootstrap intervals, two test blocks (2023; 2024-25).
- 80% prediction-range coverage, calibration (Brier skill, ECE), alert precision/recall by tier.
- Exploratory tests: Google Trends link (weak, winter-only), CAMS benchmark (no robust gain).
