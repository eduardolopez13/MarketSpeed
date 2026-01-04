# Purpose: Produce two figures from cached data + events:
#  - Volatility boxplot (post − pre) for CPI vs NFP
#  - Correlation delta heatmap around the most recent CPI
# Reads prices from data/*.parquet and events from data/events.parquet.
# Automatically discovers which assets exist in data/.

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def discover_symbols(data_dir: str = DATA_DIR) -> list[str]:
    """
    Find cached daily parquet files and return Stooq-style symbols.
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
    Load cached daily price from data/, ensure 'ret' exists.
    """
    path = os.path.join(DATA_DIR, f"{symbol.replace('.', '_')}_daily.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "ret" not in df.columns and "close" in df.columns:
        df["ret"] = df["close"].pct_change()
    return df.dropna()


def load_events(path: str = os.path.join(DATA_DIR, "events.parquet")) -> pd.DataFrame:
    """
    Load CPI/NFP events created by src/event_loader.py.
    Expected columns: ['event','event_date','value'].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing events file: {path}. Run src/event_loader.py first."
        )
    ev = pd.read_parquet(path)
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    return ev.sort_values("event_date")


def nearest_pos(index: pd.Index, when: pd.Timestamp) -> int | None:
    """
    Nearest trading index position to a calendar timestamp.
    """
    try:
        nearest = index[index.get_indexer([when], method="nearest")]
        if len(nearest) == 0:
            return None
        return index.get_loc(nearest[0])
    except Exception:
        return None


def window_stats(
    price_df: pd.DataFrame,
    event_date: pd.Timestamp,
    pre_days: int = 5,
    post_days: int = 5,
):
    """
    Pre [-5d,-1d] vs Post [+1d,+5d] stats around nearest trading day to event_date.
    Returns None if windows are empty.
    """
    if price_df.empty or "ret" not in price_df.columns:
        return None
    idx = price_df.index
    pos = nearest_pos(idx, pd.Timestamp(event_date))
    if pos is None:
        return None
    pre = price_df.iloc[max(0, pos - pre_days) : pos]
    post = price_df.iloc[pos + 1 : min(len(price_df), pos + 1 + post_days)]
    if pre.empty or post.empty:
        return None
    return {
        "pre_vol": pre["ret"].std(ddof=1),
        "post_vol": post["ret"].std(ddof=1),
        "vol_delta": post["ret"].std(ddof=1) - pre["ret"].std(ddof=1),
        "pre_mean": pre["ret"].mean(),
        "post_mean": post["ret"].mean(),
        "ret_delta": post["ret"].mean() - pre["ret"].mean(),
    }


def main():
    # Discover cached assets and load their prices
    symbols = discover_symbols(DATA_DIR)
    if not symbols:
        print("No cached prices found in data/. Run src/data_loader.py first.")
        return

    prices = {}
    for sym in symbols:
        df = load_cached_price(sym)
        if df.empty:
            print(f"[WARN] Empty cache for {sym}, skipping.")
            continue
        prices[sym] = df
    if not prices:
        print("No usable cached prices. Check data/ contents.")
        return
    print("Loaded symbols:", list(prices.keys()))

    # Load events
    events = load_events()

    # Compute pre/post stats for all assets and events
    rows = []
    for sym, df in prices.items():
        for _, e in events.iterrows():
            st = window_stats(df, e["event_date"])
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
    print("Rows:", len(res))
    if res.empty:
        print("No results. Check assets or event coverage.")
        return

    # Figure 1: Volatility delta boxplot (group assets by simple class label from symbol)
    def classify(sym: str) -> str:
        if sym.endswith(".US"):
            return "Equity"
        if sym in {"EURUSD", "USDJPY"} or "USD" in sym:
            return "FX/Commodity"
        return "Other"

    res["class"] = res["symbol"].map(classify)

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=res, x="class", y="vol_delta", hue="event")
    plt.axhline(0, color="gray", linewidth=1)
    plt.title("Post − Pre Volatility (daily windows)")
    plt.tight_layout()
    out_vol = os.path.join(FIG_DIR, "daily_vol_delta.png")
    plt.savefig(out_vol)
    print(f"Saved {out_vol}")

    # Figure 2: Correlation delta heatmap around most recent CPI
    # Build aligned returns panel from whatever assets are cached
    panel = []
    for sym, df in prices.items():
        panel.append(df[["ret"]].rename(columns={"ret": sym}))
    panel_df = pd.concat(panel, axis=1).dropna(how="any")
    last_cpi = events[events["event"] == "CPI"]["event_date"].max()
    last_cpi_ts = pd.Timestamp(last_cpi)
    idx = panel_df.index
    pos = nearest_pos(idx, last_cpi_ts)
    if pos is None:
        print("No CPI anchor found for correlation panel.")
        return
    pre = panel_df.iloc[max(0, pos - 20) : pos]
    post = panel_df.iloc[pos + 1 : min(len(panel_df), pos + 1 + 20)]
    if pre.empty or post.empty:
        print("Insufficient pre/post data for correlation panel.")
        return
    corr_delta = post.corr() - pre.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_delta, vmin=-0.5, vmax=0.5, cmap="coolwarm", annot=False)
    plt.title("Correlation Delta (Post − Pre) around last CPI")
    plt.tight_layout()
    out_corr = os.path.join(FIG_DIR, "daily_corr_delta_cpi.png")
    plt.savefig(out_corr)
    print(f"Saved {out_corr}")


if __name__ == "__main__":
    main()
