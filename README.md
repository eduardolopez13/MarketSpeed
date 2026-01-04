# Market Speed 

## Overview

This project studies how liquid markets respond to **scheduled macroeconomic releases** using **free, institutional-grade public data**.  
Macro events (CPI, Non-Farm Payrolls) are treated as **exogenous, time-stamped shocks** to the market. The system response is measured through changes in returns, volatility, and cross-asset correlations.

The main purpose of this project was to learn core quant research skills by building an event‑driven market study with free data, clean code, and clear visuals.



---

## Research Questions

1. Do scheduled macro releases materially change market volatility?
2. Is the event-day move statistically large relative to recent price behavior?
3. Do macro events increase or decrease cross-asset co-movement?
4. Are effects symmetric across asset classes and event types?

---

## Data Sources

- **Daily prices:** Stooq via `pandas-datareader`
- **Macro event calendar:** FRED via `pandas-datareader`
  - CPI
  - Non-Farm Payrolls (NFP)

All data is pulled programmatically and cached locally to ensure full reproducibility.

---

## Methodology

Each macro release is treated as a **known event date**. For each asset and event:

1. Define a **pre-event window** and **post-event window** (±5 trading days).
2. Compute return-based statistics before and after the release.
3. Compare event-day returns to a rolling historical baseline.
4. Measure changes in cross-asset correlation structure.

No model fitting or forecasting is performed at this stage. The focus is on **empirical response diagnostics**, not prediction.

---

## Key Metrics

- **vol_delta**  
  Post − pre volatility, where volatility is the standard deviation of daily returns over the event window.

  - `> 0` → higher noise after the event  
  - `< 0` → volatility compression after the event  

- **ret_delta**  
  Post − pre mean daily return (drift shift).

- **impact_ratio**  

```impact_ratio = abs(event_day_return) / median(abs(returns[-20:]))```

  - `≈ 1` → event day is typical
  - `> 1` → outsized macro impact
  - `< 1` → muted reaction

---

## Repository Structure

```text
market-speed/
├─ README.md
├─ environment.yml
├─ .gitignore
├─ src/
│  ├─ data_loader.py         # Stooq daily loader + returns
│  ├─ event_loader.py        # CPI/NFP events from FRED → data/events.parquet
│  ├─ daily_event_min.py     # Pre/post (±5d) stats → CSV + console medians
│  ├─ event_day_impact.py    # Event-day impact ratios → CSV + console medians
│  ├─ daily_event_study.py   # Volatility boxplot + correlation heatmap → PNGs
│  └─ basic_tests.py         # t-tests + written summary → TXT
├─ data/                     # Cached Parquet/CSV (generated)
└─ figures/                  # PNGs/CSVs/TXT outputs (generated)

```


