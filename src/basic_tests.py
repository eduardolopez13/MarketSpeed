# filename: src/basic_tests.py
# Purpose: Quantify and explain statistical significance for your daily event study.
# Tests:
# - vol_delta vs 0 (no change in volatility across ±5d windows)
# - impact_ratio vs 1 (event-day move equals typical day)
#
# Inputs produced by your pipeline:
# - figures/daily_event_min_results.csv  (from daily_event_min.py)
# - figures/event_day_impact_results.csv (from event_day_impact.py)
#
# Outputs:
# - figures/basic_tests_summary.txt (human-readable conclusions)
# - Console prints of the same conclusions

import os
import math
import pandas as pd
from scipy import stats

PREPOST_CSV = "figures/daily_event_min_results.csv"
IMPACT_CSV = "figures/event_day_impact_results.csv"
SUMMARY_TXT = "figures/basic_tests_summary.txt"

MIN_N = 5  # minimum observations to run a test
ALPHA = 0.05  # significance threshold


def explain_vol_delta(symbol, event, mean_val, n, t, p):
    # Directional interpretation
    if math.isnan(mean_val):
        return f"{symbol} ({event}): Not enough data."
    direction = "higher" if mean_val > 0 else "lower" if mean_val < 0 else "no change"
    sig = p < ALPHA
    if sig:
        return (
            f"{symbol} ({event}): Statistically significant change in post vs pre volatility "
            f"(mean Δ={mean_val:.6f}, n={n}, t={t:.2f}, p={p:.3g}). "
            f"This means daily noise is {direction} in the 5 days after the event compared to the 5 days before."
        )
    else:
        return (
            f"{symbol} ({event}): Not statistically different from zero (mean Δ={mean_val:.6f}, n={n}, t={t:.2f}, p={p:.3g}). "
            f"This means typical events do not change daily volatility in a consistent way for this asset."
        )


def explain_impact_ratio(symbol, event, mean_ratio, n, t, p):
    # Directional interpretation relative to 1
    if math.isnan(mean_ratio):
        return f"{symbol} ({event}): Not enough data."
    if mean_ratio > 1:
        dir_text = "larger-than-usual moves on event days"
    elif mean_ratio < 1:
        dir_text = "smaller-than-usual moves on event days"
    else:
        dir_text = "typical-sized moves on event days"
    sig = p < ALPHA
    if sig:
        return (
            f"{symbol} ({event}): Statistically significant difference from typical day (mean ratio={mean_ratio:.3f}, "
            f"n={n}, t={t:.2f}, p={p:.3g}). This means {dir_text} relative to the prior 20-day baseline."
        )
    else:
        return (
            f"{symbol} ({event}): Not statistically different from typical day (mean ratio={mean_ratio:.3f}, "
            f"n={n}, t={t:.2f}, p={p:.3g}). This means event-day moves are broadly in line with recent norms."
        )


def run_vol_delta_tests():
    if not os.path.exists(PREPOST_CSV):
        raise FileNotFoundError(f"Missing {PREPOST_CSV}. Run daily_event_min.py first.")
    df = pd.read_csv(PREPOST_CSV)
    df = df.dropna(subset=["vol_delta"])
    conclusions = []
    for (sym, ev), grp in df.groupby(["symbol", "event"]):
        x = grp["vol_delta"].astype(float).dropna()
        if len(x) >= MIN_N:
            t, p = stats.ttest_1samp(x, 0.0, nan_policy="omit")
            mean_val = x.mean()
            conclusions.append(explain_vol_delta(sym, ev, mean_val, len(x), t, p))
        else:
            conclusions.append(f"{sym} ({ev}): Not enough observations (n={len(x)}).")
    return conclusions


def run_impact_ratio_tests():
    if not os.path.exists(IMPACT_CSV):
        raise FileNotFoundError(f"Missing {IMPACT_CSV}. Run event_day_impact.py first.")
    df = pd.read_csv(IMPACT_CSV)
    df = df.dropna(subset=["impact_ratio"])
    conclusions = []
    for (sym, ev), grp in df.groupby(["symbol", "event"]):
        x = grp["impact_ratio"].astype(float).dropna()
        if len(x) >= MIN_N:
            t, p = stats.ttest_1samp(x, 1.0, nan_policy="omit")  # H0: mean == 1
            mean_ratio = x.mean()
            conclusions.append(explain_impact_ratio(sym, ev, mean_ratio, len(x), t, p))
        else:
            conclusions.append(f"{sym} ({ev}): Not enough observations (n={len(x)}).")
    return conclusions


def main():
    vol_conc = run_vol_delta_tests()
    imp_conc = run_impact_ratio_tests()

    all_conc = (
        ["Volatility (post − pre) results:"]
        + vol_conc
        + ["", "Event-day impact results:"]
        + imp_conc
    )

    # Print to console
    print("\n".join(all_conc))

    # Save to summary file
    os.makedirs(os.path.dirname(SUMMARY_TXT), exist_ok=True)
    with open(SUMMARY_TXT, "w") as f:
        f.write("\n".join(all_conc))
    print(f"\nSaved {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
