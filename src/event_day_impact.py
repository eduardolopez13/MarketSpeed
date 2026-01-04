# filename: src/event_day_impact.py
# Purpose: Compare event-day absolute return to a 20-day baseline for cached symbols.
# Reads prices from data/*.parquet and events from data/events.parquet.

import os
import re
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def discover_symbols(data_dir: str = DATA_DIR):
    """Find cached daily parquet files and return Stooq-style symbols."""
    syms = []
    for f in os.listdir(data_dir):
        m = re.match(r"(.+)_daily\.parquet$", f)
        if m:
            syms.append(m.group(1).replace("_", "."))
    return sorted(syms)


def load_cached_price(symbol: str) -> pd.DataFrame:
    """Load cached daily data and ensure 'ret' exists."""
    path = os.path.join(DATA_DIR, f"{symbol.replace('.', '_')}_daily.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "ret" not in df.columns and "close" in df.columns:
        df["ret"] = df["close"].pct_change()
    return df.dropna()


def load_events(path: str = os.path.join(DATA_DIR, "events.parquet")) -> pd.DataFrame:
    """Load CPI/NFP events created by src/event_loader.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing events file: {path}. Run src/event_loader.py first."
        )
    ev = pd.read_parquet(path)
    ev["event_date"] = pd.to_datetime(ev["event_date"])  # ensure Timestamp
    return ev.sort_values("event_date")


def nearest_trading_pos(idx: pd.Index, event_date: pd.Timestamp) -> int | None:
    """Snap calendar event_date to the nearest trading date in the price index."""
    try:
        nearest = idx[idx.get_indexer([event_date], method="nearest")]
        if len(nearest) == 0:
            return None
        return idx.get_loc(nearest[0])
    except Exception:
        return None


def event_day_stats(price_df: pd.DataFrame, event_date: pd.Timestamp):
    """Event-day absolute return vs median absolute return over prior 20 trading days."""
    if price_df.empty or "ret" not in price_df.columns:
        return None
    idx = price_df.index
    pos = nearest_trading_pos(idx, pd.Timestamp(event_date))
    if pos is None or pos == 0:
        return None
    t0 = idx[pos]
    ret_t0 = float(price_df.loc[t0, "ret"])
    pre = price_df.iloc[max(0, pos - 20) : pos]["ret"]
    if pre.empty:
        return None
    baseline_abs = float(pre.abs().median())
    impact_ratio = (abs(ret_t0) / baseline_abs) if baseline_abs > 0 else np.nan
    return {
        "event_day_ret": ret_t0,
        "event_day_abs": abs(ret_t0),
        "baseline_abs20": baseline_abs,
        "impact_ratio": impact_ratio,
    }


def main():
    symbols = discover_symbols(DATA_DIR)
    if not symbols:
        print("No cached data found in data/. Run src/data_loader.py first.")
        return
    print("Symbols discovered:", symbols)

    events = load_events()
    events_tail = events.tail(30)  # recent months for concise output

    rows = []
    for sym in symbols:
        df = load_cached_price(sym)
        if df.empty:
            print(f"[WARN] Empty cache for {sym}, skipping.")
            continue
        for _, e in events_tail.iterrows():
            st = event_day_stats(df, e["event_date"])
            if st:
                rows.append(
                    {
                        "symbol": sym,
                        "event": e["event"],
                        "event_date": e["event_date"].date(),
                        **st,
                    }
                )

    res = pd.DataFrame(rows)
    if res.empty:
        print("No results. Check symbols or widen date range.")
        return

    print("\nMedian impact ratio (|event-day| / median |return| last 20d):")
    print(res.groupby(["symbol", "event"])["impact_ratio"].median())

    out_csv = os.path.join(OUT_DIR, "event_day_impact_results.csv")
    res.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
