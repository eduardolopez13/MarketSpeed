import os
import pandas as pd
from pandas_datareader import data as web

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def fred(series_id: str) -> pd.DataFrame:
    """Fetch a FRED series and return a DataFrame with a DatetimeIndex and 'value' column."""
    df = web.DataReader(series_id, "fred")
    df = df.rename(columns={series_id: "value"})
    df.index = pd.to_datetime(df.index)
    return df


def cpi_yoy_events() -> pd.DataFrame:
    """
    CPI as YoY %:
    - Start with CPI level (CPIAUCSL)
    - Convert to year-over-year percent change
    - Keep monthly dates as event_date
    """
    cpi = fred("CPIAUCSL").asfreq("MS")
    cpi["yoy_pct"] = cpi["value"].pct_change(12) * 100
    cpi = cpi.dropna()[["yoy_pct"]]
    out = pd.DataFrame(
        {
            "event": "CPI",
            "event_date": cpi.index.date,
            "value": cpi["yoy_pct"].values,  # YoY % inflation
        }
    )
    return out


def nfp_events() -> pd.DataFrame:
    """
    NFP level (PAYEMS):
    - Use monthly level (thousands)
    - Keep monthly dates as event_date
    """
    nfp = fred("PAYEMS").asfreq("MS").dropna()[["value"]]
    out = pd.DataFrame(
        {
            "event": "NFP",
            "event_date": nfp.index.date,
            "value": nfp["value"].values,  # employment level (thousands)
        }
    )
    return out


def build_events() -> pd.DataFrame:
    """Combine CPI and NFP into a single tidy table and save to data/events.parquet."""
    cpi_df = cpi_yoy_events()
    nfp_df = nfp_events()
    events = pd.concat([cpi_df, nfp_df], ignore_index=True).sort_values("event_date")
    events.to_parquet(os.path.join(DATA_DIR, "events.parquet"))
    return events


if __name__ == "__main__":
    ev = build_events()
    print("Saved data/events.parquet")
    print("Preview:")
    print(ev.tail(6))
    print("\nDone. Next: run daily_event_min.py or daily_event_study.py.")
