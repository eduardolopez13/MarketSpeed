# filename: src/data_loader.py
# Purpose: Fetch FREE daily prices from Stooq, compute returns, cache to data/, print a summary.

import os
from typing import List, Dict
import pandas as pd
from pandas_datareader import data as web

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# US listings on Stooq must end with .US (e.g., SPY.US). FX pairs have no suffix (EURUSD, USDJPY).
ASSETS: Dict[str, str] = {
    "SPY.US": "Equity",
    "AAPL.US": "Equity",
    "MSFT.US": "Equity",
    "GLD.US": "CommodityETF",
    "SLV.US": "CommodityETF",
    "USO.US": "CommodityETF",
    "UNG.US": "CommodityETF",
    "DBA.US": "CommodityETF",
    "DBC.US": "CommodityETF",
}

START_DATE = "2021-01-01"  # adjust as needed


def fetch_stooq(symbol: str, start: str = START_DATE) -> pd.DataFrame:
    """
    Download daily data from Stooq, sort ascending, lowercase columns, compute daily returns.
    Returns a tidy DataFrame with columns: ['open','high','low','close','volume','ret'].
    """
    try:
        df = web.DataReader(symbol, "stooq", start=start)
    except Exception as e:
        print(f"[ERROR] Stooq fetch failed for {symbol}: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        print(f"[WARN] No data for {symbol} from Stooq.")
        return pd.DataFrame()

    # Stooq returns latest-first; sort ascending and normalize columns
    df = df.sort_index().rename(columns=str.lower)

    # Ensure 'close' exists (defensive)
    if "close" not in df.columns:
        print(
            f"[WARN] 'close' column missing for {symbol}. Columns found: {list(df.columns)}"
        )
        return pd.DataFrame()

    # Daily returns
    df["ret"] = df["close"].pct_change()
    df = df.dropna()

    # Cache to Parquet
    out_path = os.path.join(DATA_DIR, f"{symbol.replace('.', '_')}_daily.parquet")
    try:
        df.to_parquet(out_path)
    except Exception as e:
        print(f"[WARN] Parquet write failed for {symbol}: {e}. Writing CSV instead.")
        df.to_csv(out_path.replace(".parquet", ".csv"))

    return df


def main():
    print("Stooq daily loader: START")
    print("Assets:", list(ASSETS.keys()))
    print("Data directory:", os.path.abspath(DATA_DIR))

    shapes = {}
    for sym in ASSETS.keys():
        df = fetch_stooq(sym, start=START_DATE)
        shapes[sym] = df.shape
        print(f"{sym}: {df.shape}")

    print("\nSummary (rows, cols):")
    for sym, shp in shapes.items():
        print(f"  {sym}: {shp}")

    # Quick sanity: show last few returns if available
    for sym in ASSETS.keys():
        out_path = os.path.join(DATA_DIR, f"{sym.replace('.', '_')}_daily.parquet")
        if os.path.exists(out_path):
            df = pd.read_parquet(out_path)
            tail = df[["close", "ret"]].tail(3)
            print(f"\n{sym} tail:")
            print(tail)

    print(
        "\nDone. Next: run events_loader.py, then daily_event_min.py or event_day_impact.py."
    )


if __name__ == "__main__":
    main()
