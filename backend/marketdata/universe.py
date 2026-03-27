"""
Symbol universe and asset type classification.

ETFs/indexes and individual stocks are scored differently:
- ETFs are mean-reverting baskets — gap fades, structural bounces, and VWAP
  reclaims work best. ATH breakouts and gap-and-go are low-signal.
- Stocks have catalyst-driven momentum — sweeps, ATH breakouts, and gap-go
  work best. Gap fades are less reliable.
"""

SYMBOLS = [
    # Broad market ETFs
    "SPY", "QQQ", "IWM", "DIA",

    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "AVGO", "AMD", "NFLX", "INTC", "ORCL", "ADBE", "CRM",

    # Semiconductors
    "SMH", "SOXX", "MU", "QCOM", "TXN", "AMAT", "KLAC", "LRCX",

    # Financials
    "JPM", "BAC", "GS", "MS", "C", "WFC", "BLK", "AXP",

    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",

    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY",

    # Consumer
    "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "WMT",

    # Industrials
    "CAT", "DE", "BA", "HON", "GE", "RTX",

    # Sector ETFs
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",

    # Fixed income & macro
    "TLT", "IEF", "HYG", "LQD",

    # Commodities
    "GLD", "SLV", "USO", "UNG",
]

# All ETFs/indexes in the universe — scored differently from individual stocks.
ETF_SYMBOLS = frozenset({
    # Broad market
    "SPY", "QQQ", "IWM", "DIA",
    # Sector ETFs
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
    # Semiconductor ETFs
    "SMH", "SOXX",
    # Fixed income
    "TLT", "IEF", "HYG", "LQD",
    # Commodities
    "GLD", "SLV", "USO", "UNG",
})


def get_asset_type(symbol: str) -> str:
    """Return 'etf' or 'stock' for a given symbol."""
    return "etf" if symbol.upper() in ETF_SYMBOLS else "stock"
