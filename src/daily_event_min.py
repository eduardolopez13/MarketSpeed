# Minimal daily event study reading cached prices from data/ and event dates from data/events.parquet.
# Computes pre/post (±5 trading days) volatility and mean return around CPI/NFP.

import os
import re
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def discover_symbols(data_dir: str = DATA_DIR):
    """
    Find cached daily parquet files and return Stooq-style symbols.
    Expects filenames like SYMBOL_daily.parquet (where SYMBOL uses underscores).
    Example: 'SPY_US_daily.parquet' -> 'SPY.US'
    """
    syms = []
    for f in os.listdir(data_dir):
        m = re.match(r"(.+)_daily\.parquet$", f)
        if m:
            sym = m.group(1).replace("_", ".")
            syms.append(sym)
    return sorted(syms)


def load_cached_price(symbol: str) -> pd.DataFrame:
    """
    Load cached daily data (Parquet) and ensure it has 'close' and 'ret'.
    'ret' is computed if missing: ret = pct_change(close).
    """
    path = os.path.join(DATA_DIR, f"{symbol.replace('.', '_')}_daily.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Normalize columns to lowercase
    df.columns = [c.lower() for c in df.columns]
    # Compute returns if not present
    if "ret" not in df.columns and "close" in df.columns:
        df["ret"] = df["close"].pct_change()
    # Drop rows with NA after pct_change
    return df.dropna()


def nearest_trading_pos(idx: pd.Index, event_date: pd.Timestamp) -> int | None:
    """
    Map a calendar event_date to the nearest index position in price_df.
    Handles non-trading days by snapping to nearest trading date.
    """
    try:
        nearest = idx[idx.get_indexer([event_date], method="nearest")]
        if len(nearest) == 0:
            return None
        return idx.get_loc(nearest[0])
    except Exception:
        return None


def pre_post_stats(
    price_df: pd.DataFrame,
    event_date: pd.Timestamp,
    pre_days: int = 5,
    post_days: int = 5,
):
    """
    Compute pre/post volatility and mean return around event_date.
    pre: [-5d, -1d], post: [+1d, +5d] relative to the nearest trading day.
    Returns None if windows are empty or the event is at the start of the series.
    """
    if price_df.empty or "ret" not in price_df.columns:
        return None
    idx = price_df.index
    pos = nearest_trading_pos(idx, pd.Timestamp(event_date))
    if pos is None or pos == 0:
        return None

    pre = price_df.iloc[max(0, pos - pre_days) : pos]
    post = price_df.iloc[pos + 1 : min(len(price_df), pos + 1 + post_days)]
    if pre.empty or post.empty:
        return None

    pre_vol = pre["ret"].std(ddof=1)
    post_vol = post["ret"].std(ddof=1)
    pre_mean = pre["ret"].mean()
    post_mean = post["ret"].mean()

    return {
        "pre_vol": float(pre_vol),
        "post_vol": float(post_vol),
        "vol_delta": float(post_vol - pre_vol),
        "pre_mean": float(pre_mean),
        "post_mean": float(post_mean),
        "ret_delta": float(post_mean - pre_mean),
    }


def load_events(path: str = os.path.join(DATA_DIR, "events.parquet")) -> pd.DataFrame:
    """
    Load CPI/NFP events saved by src/event_loader.py.
    Expected columns: ['event','event_date','value'] where event_date is a date.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing events file: {path}. Run src/event_loader.py first."
        )
    ev = pd.read_parquet(path)
    # Ensure event_date is usable as a Timestamp later
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    return ev.sort_values("event_date")


def main():
    # 1) Symbols from your cached data
    symbols = discover_symbols(DATA_DIR)
    if not symbols:
        print("No cached data found in data/. Run src/data_loader.py first.")
        return
    print("Symbols discovered:", symbols)

    # 2) Load CPI/NFP event dates
    events = load_events()
    # Keep a small recent subset for concise output; adjust as needed
    events_tail = events.tail(20)

    # 3) Compute per-symbol stats across recent events
    rows = []
    for sym in symbols:
        df = load_cached_price(sym)
        if df.empty:
            print(f"[WARN] Empty or missing cache for {sym}, skipping.")
            continue
        for _, e in events_tail.iterrows():
            st = pre_post_stats(df, e["event_date"])
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
        print("No results (check symbols or widen event subset).")
        return

    # 4) Print concise medians
    print("\nMedian pre/post deltas (last ~20 events):")
    print(res.groupby(["symbol", "event"])[["vol_delta", "ret_delta"]].median())

    # 5) Save tidy results for your repo
    out_csv = os.path.join(OUT_DIR, "daily_event_min_results.csv")
    res.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
