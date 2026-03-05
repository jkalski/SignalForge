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
    "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "WMT",

    # Industrials
    "CAT", "DE", "BA", "HON", "GE", "RTX",

    # Sector ETFs
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",

    # Fixed income & macro
    "TLT", "IEF", "HYG", "LQD",

    # Commodities
    "GLD", "SLV", "USO", "UNG",
]

# Deduplicate while preserving order (AMZN appears in tech + consumer)
_seen: set = set()
_deduped = []
for _s in SYMBOLS:
    if _s not in _seen:
        _seen.add(_s)
        _deduped.append(_s)
SYMBOLS = _deduped
