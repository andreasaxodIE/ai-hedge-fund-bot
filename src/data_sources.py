import os
import requests
from datetime import datetime, timedelta

BASE = "https://api.massive.com"

def _get_json_massive(path: str, params=None):
    """
    Tries Massive with query-param auth first, then Bearer header auth.
    Returns JSON dict (or error dict).
    """
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        return {"error": "Missing MASSIVE_API_KEY"}

    params = params or {}
    url = BASE + path

    # Try query param auth
    try:
        qp = dict(params)
        qp["apiKey"] = key
        r = requests.get(url, params=qp, timeout=60)
        if r.status_code != 403:
            r.raise_for_status()
            return r.json()
    except Exception as e:
        last_err = str(e)
    else:
        last_err = f"403 Forbidden (query-param auth) at {url}"

    # Try header auth
    try:
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        r2 = requests.get(url, params=params, headers=headers, timeout=60)
        r2.raise_for_status()
        return r2.json()
    except Exception as e2:
        return {
            "error": "Massive request failed",
            "details": str(e2),
            "note": last_err,
            "url": url,
            "status_code": getattr(getattr(e2, "response", None), "status_code", None),
        }

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
    Free fallback. Stooq often uses:
      aapl.us (US)
    EU tickers are inconsistent; we try a few patterns.
    """
    base = ticker.lower()

    # If ticker has dot (e.g. bnp.pa), try variants
    candidates = []
    if "." in base:
        left, right = base.split(".", 1)
        candidates += [
            f"{left}.{right}",     # as-is
            f"{left}",             # base only
            f"{left}.us",          # sometimes ADRs etc
            f"{left}.fr",          # guess for Paris (not guaranteed)
        ]
    else:
        candidates += [f"{base}.us", base]

    for sym in candidates:
        try:
            url = "https://stooq.com/q/d/l/"
            params = {"s": sym, "i": "d"}
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200 and "Date,Open,High,Low,Close,Volume" in r.text:
                return {"source": "stooq", "symbol": sym, "csv": r.text[:20000], "tried": candidates}
        except Exception:
            continue

    return {"source": "stooq", "error": "No data found", "tried": candidates}

def fetch_prices_yahoo(ticker: str):
    """
    Free fallback with good global coverage.
    Uses Yahoo chart endpoint for 1 month daily data.
    """
    # Yahoo expects e.g. BNP.PA, NESN.SW, TSLA
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "1mo", "interval": "1d"}

    try:
        r = requests.get(url, params=params, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        j = r.json()

        chart = (j.get("chart") or {})
        result = (chart.get("result") or [None])[0]
        if not result:
            return {"source": "yahoo", "error": "No chart result", "raw": j}

        timestamps = result.get("timestamp") or []
        indicators = (result.get("indicators") or {})
        quote0 = (indicators.get("quote") or [None])[0] or {}
        closes = quote0.get("close") or []

        # closes can include None values; filter aligned
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
            return {"source": "yahoo", "error": "Not enough data points", "ticker": ticker}

        return {"source": "yahoo", "ticker": ticker, "closes": clean_closes[-30:], "timestamps": clean_ts[-30:]}
    except Exception as e:
        return {"source": "yahoo", "error": str(e), "ticker": ticker}

def fetch_market_bundle(ticker: str) -> dict:
    snapshot = fetch_snapshot(ticker)
    aggs_30d = fetch_daily_aggs_last_30d(ticker)
    overview = fetch_ticker_overview(ticker)

    massive_failed = any(isinstance(x, dict) and x.get("error") for x in [snapshot, aggs_30d, overview])

    bundle = {
        "ticker": ticker,
        "snapshot": snapshot,
        "aggs_30d": aggs_30d,
        "overview": overview,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

    # Free fallbacks
    stooq = fetch_prices_stooq(ticker)
    bundle["fallback_prices"] = stooq

    # If Stooq failed OR Massive failed (often for non-US), try Yahoo
    if stooq.get("error") or massive_failed:
        bundle["fallback_prices_yahoo"] = fetch_prices_yahoo(ticker)

    if massive_failed:
        bundle["note"] = "Massive returned errors (likely permissions/403). Using free fallbacks (Stooq/Yahoo) where possible."

    return bundle
