import os
import requests
from datetime import datetime, timedelta

BASE = "https://api.massive.com"

def _get_json(path: str, params=None):
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        return {"error": "Missing MASSIVE_API_KEY"}

    params = params or {}
    # Massive examples typically use apiKey query param (Polygon-style)
    params["apiKey"] = key

    url = BASE + path
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def fetch_snapshot(ticker: str):
    return _get_json(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")

def fetch_daily_aggs_last_30d(ticker: str):
    end = datetime.utcnow().date()
    start = end - timedelta(days=30)
    return _get_json(f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}")

def fetch_ticker_overview(ticker: str):
    return _get_json(f"/v3/reference/tickers/{ticker}")

def fetch_market_bundle(ticker: str) -> dict:
    snapshot = fetch_snapshot(ticker)
    aggs_30d = fetch_daily_aggs_last_30d(ticker)
    overview = fetch_ticker_overview(ticker)

    return {
        "ticker": ticker,
        "snapshot": snapshot,
        "aggs_30d": aggs_30d,
        "overview": overview,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

