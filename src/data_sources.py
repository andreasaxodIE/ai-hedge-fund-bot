import os
import requests
from datetime import datetime, timedelta

BASE = "https://api.massive.com"

def _get_json(path: str, params=None):
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        return {"error": "Missing MASSIVE_API_KEY"}

    params = params or {}

    url = BASE + path

    # Try query param auth first (Polygon-style)
    try:
        qp = dict(params)
        qp["apiKey"] = key
        r = requests.get(url, params=qp, timeout=60)
        if r.status_code != 403:
            r.raise_for_status()
            return r.json()
    except Exception as e:
        # continue to header attempt
        last_err = str(e)
    else:
        last_err = f"403 Forbidden (query param auth) at {url}"

    # Try header auth (some providers require this)
    try:
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }
        r2 = requests.get(url, params=params, headers=headers, timeout=60)
        r2.raise_for_status()
        return r2.json()
    except Exception as e2:
        return {
            "error": "Massive request failed",
            "status": getattr(getattr(e2, "response", None), "status_code", None),
            "details": str(e2),
            "note": "Check MASSIVE_API_KEY validity/plan/endpoint permissions. Some endpoints may require a paid tier.",
            "url": url,
        }

def fetch_snapshot(ticker: str):
    return _get_json(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")

def fetch_daily_aggs_last_30d(ticker: str):
    end = datetime.utcnow().date()
    start = end - timedelta(days=30)
    return _get_json(f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}")

def fetch_ticker_overview(ticker: str):
    return _get_json(f"/v3/reference/tickers/{ticker}")

def fetch_prices_stooq(ticker: str):
    # free fallback if Massive blocks you
    candidates = [f"{ticker.lower()}.us", ticker.lower()]
    for sym in candidates:
        url = "https://stooq.com/q/d/l/"
        params = {"s": sym, "i": "d"}
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200 and "Date,Open,High,Low,Close,Volume" in r.text:
                return {"source": "stooq", "symbol": sym, "csv": r.text[:20000]}
        except Exception:
            pass
    return {"source": "stooq", "error": "No data found"}

def fetch_market_bundle(ticker: str) -> dict:
    snapshot = fetch_snapshot(ticker)
    aggs_30d = fetch_daily_aggs_last_30d(ticker)
    overview = fetch_ticker_overview(ticker)

    massive_failed = any(
        isinstance(x, dict) and x.get("error")
        for x in [snapshot, aggs_30d, overview]
    )

    bundle = {
        "ticker": ticker,
        "snapshot": snapshot,
        "aggs_30d": aggs_30d,
        "overview": overview,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

    if massive_failed:
        bundle["fallback_prices"] = fetch_prices_stooq(ticker)
        bundle["note"] = "Massive returned errors (likely 403/permissions). Included Stooq fallback prices so the analysis can still run."

    return bundle
