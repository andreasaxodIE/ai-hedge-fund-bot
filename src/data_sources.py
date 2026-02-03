import os
import time
import requests
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

BASE = "https://api.massive.com"


def _http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
                   timeout: int = 60, retries: int = 3, backoff: float = 1.5) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(backoff * (2 ** attempt))
    return {"error": "http_get_json_failed", "details": str(last_err), "url": url}


def _http_get_text(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
                   timeout: int = 60, retries: int = 3, backoff: float = 1.5) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return {"ok": True, "text": r.text}
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(backoff * (2 ** attempt))
    return {"ok": False, "error": "http_get_text_failed", "details": str(last_err), "url": url}


def _get_json_massive(path: str, params=None):
    """
    Tries Massive with query-param auth first, then Bearer auth.
    Returns JSON dict (or error dict).
    """
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        return {"error": "missing_massive_api_key"}

    params = params or {}
    url = BASE + path

    # Try query param auth
    qp = dict(params)
    qp["apiKey"] = key
    j1 = _http_get_json(url, params=qp, headers={"Accept": "application/json"}, retries=1)
    if not (isinstance(j1, dict) and j1.get("error")):
        return j1

    # If forbidden or failed, try Bearer auth
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    j2 = _http_get_json(url, params=params, headers=headers, retries=1)
    return j2


def fetch_snapshot(ticker: str):
    return _get_json_massive(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")


def fetch_daily_aggs_last_30d(ticker: str):
    end = datetime.utcnow().date()
    start = end - timedelta(days=30)
    return _get_json_massive(f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}")


def fetch_ticker_overview(ticker: str):
    return _get_json_massive(f"/v3/reference/tickers/{ticker}")


def fetch_prices_stooq(ticker: str):
    """
    Free fallback. Stooq is inconsistent for some tickers, we try a few patterns.
    """
    base = ticker.lower()
    candidates = []

    if "." in base:
        left, right = base.split(".", 1)
        candidates += [
            f"{left}.{right}",
            f"{left}",
            f"{left}.us",
            f"{left}.fr",
            f"{left}.de",
            f"{left}.uk",
        ]
    else:
        candidates += [f"{base}.us", base]

    for sym in candidates:
        url = "https://stooq.com/q/d/l/"
        params = {"s": sym, "i": "d"}
        resp = _http_get_text(url, params=params, timeout=60, retries=2)
        if resp.get("ok") and "Date,Open,High,Low,Close,Volume" in (resp.get("text") or ""):
            return {"source": "stooq", "symbol": sym, "csv": (resp["text"] or "")[:20000], "tried": candidates}

    return {"source": "stooq", "error": "no_data_found", "tried": candidates}


def fetch_prices_yahoo(ticker: str):
    """
    Free fallback with strong global coverage.
    Uses Yahoo chart endpoint for 1 month daily data.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "1mo", "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    j = _http_get_json(url, params=params, headers=headers, timeout=60, retries=2)
    if j.get("error"):
        return {"source": "yahoo_chart", "error": j.get("details") or j.get("error"), "ticker": ticker}

    chart = (j.get("chart") or {})
    result = (chart.get("result") or [None])[0]
    if not result:
        return {"source": "yahoo_chart", "error": "no_chart_result", "ticker": ticker}

    timestamps = result.get("timestamp") or []
    indicators = (result.get("indicators") or {})
    quote0 = (indicators.get("quote") or [None])[0] or {}
    closes = quote0.get("close") or []

    clean_closes = []
    clean_ts = []
    for ts, c in zip(timestamps, closes):
        if c is None:
            continue
        try:
            val = float(c)
            if val > 0:
                clean_closes.append(val)
                clean_ts.append(int(ts))
        except Exception:
            continue

    if len(clean_closes) < 5:
        return {"source": "yahoo_chart", "error": "not_enough_points", "ticker": ticker}

    return {"source": "yahoo_chart", "ticker": ticker, "closes": clean_closes[-30:], "timestamps": clean_ts[-30:]}


def fetch_quote_yahoo(ticker: str):
    """
    Free Yahoo quote endpoint for light fundamentals:
      shortName, longName, marketCap, currency, exchange, quoteType, etc.
    """
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    params = {"symbols": ticker}
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    j = _http_get_json(url, params=params, headers=headers, timeout=60, retries=2)
    if j.get("error"):
        return {"source": "yahoo_quote", "error": j.get("details") or j.get("error"), "ticker": ticker}

    result = (((j.get("quoteResponse") or {}).get("result")) or [])
    item = result[0] if result else None
    if not item:
        return {"source": "yahoo_quote", "error": "no_quote_result", "ticker": ticker}

    # Keep it clean: only a few fields
    keep = {}
    for k in [
        "shortName", "longName", "marketCap", "currency", "exchange",
        "quoteType", "regularMarketPrice", "regularMarketPreviousClose",
        "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "trailingPE", "forwardPE",
        "dividendYield", "sector", "industry"
    ]:
        if k in item:
            keep[k] = item[k]
    return {"source": "yahoo_quote", "ticker": ticker, "data": keep}


def fetch_benchmark_yahoo(benchmark: str):
    """
    Benchmark chart data via Yahoo.
    """
    return fetch_prices_yahoo(benchmark) | {"benchmark": benchmark}


def fetch_market_bundle(ticker: str) -> dict:
    benchmark = os.environ.get("BENCHMARK_TICKER", "^GSPC")

    massive_enabled = bool(os.environ.get("MASSIVE_API_KEY"))

    snapshot = fetch_snapshot(ticker) if massive_enabled else {"error": "missing_massive_api_key"}
    aggs_30d = fetch_daily_aggs_last_30d(ticker) if massive_enabled else {"error": "missing_massive_api_key"}
    overview = fetch_ticker_overview(ticker) if massive_enabled else {"error": "missing_massive_api_key"}

    massive_failed = any(isinstance(x, dict) and x.get("error") for x in [snapshot, aggs_30d, overview])

    bundle = {
        "ticker": ticker,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",

        # Keep Massive raw (optional)
        "snapshot": snapshot,
        "aggs_30d": aggs_30d,
        "overview": overview,

        # Free fallbacks
        "yahoo_chart": fetch_prices_yahoo(ticker),
        "stooq": fetch_prices_stooq(ticker),
        "yahoo_quote": fetch_quote_yahoo(ticker),

        # Benchmark
        "benchmark": benchmark,
        "benchmark_chart": fetch_benchmark_yahoo(benchmark),
    }

    if massive_failed:
        bundle["note"] = "Massive returned errors (common for some tickers). Using free Yahoo/Stooq fallbacks."

    return bundle
