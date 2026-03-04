"""Yahoo Finance bar fetcher using yfinance.

Interval limits (yfinance):
  1m  - last 7 days max
  5m  - last 60 days max
  15m - last 60 days max
  60m - last 730 days max
  1d  - full history

Use `period` for quick backfills or `start`/`end` for precise ranges.
"""

import math
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

# Map our canonical timeframe -> yfinance interval string
YAHOO_INTERVAL_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "60m",
    "1d":  "1d",
}

# Sensible defaults that stay within yfinance intraday limits
YAHOO_DEFAULT_PERIOD = {
    "1m":  "5d",
    "5m":  "1mo",
    "15m": "1mo",
    "1h":  "1y",
    "1d":  "2y",
}


def fetch_yahoo_bars(
    symbol: str,
    timeframe: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_bars: Optional[int] = None,
) -> list[dict]:
    """Return a list of OHLCV dicts from Yahoo Finance.

    Timestamps are naive UTC datetimes.
    Bars with zero close are dropped.
    """
    yf_interval = YAHOO_INTERVAL_MAP[timeframe]
    ticker = yf.Ticker(symbol)

    kwargs: dict = {
        "interval": yf_interval,
        "auto_adjust": True,
        "prepost": False,
    }

    if start or end:
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        kwargs["period"] = period or YAHOO_DEFAULT_PERIOD[timeframe]

    df = ticker.history(**kwargs)

    if df is None or df.empty:
        return []

    bars = []
    for idx, row in df.iterrows():
        ts = idx.to_pydatetime()
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

        close_val = float(row["Close"])
        if close_val == 0:
            continue

        vol = row["Volume"]
        if vol is None or (isinstance(vol, float) and math.isnan(vol)):
            volume = None
        else:
            volume = int(vol)

        bars.append({
            "ts":     ts,
            "open":   float(row["Open"]),
            "high":   float(row["High"]),
            "low":    float(row["Low"]),
            "close":  close_val,
            "volume": volume,
        })

    if max_bars:
        bars = bars[:max_bars]

    return bars
