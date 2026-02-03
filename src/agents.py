# src/agents.py
# Gemini committee version with "clean context" + free-data first.
# - Avoids poisoning the prompt with Massive 403 error blobs
# - Uses Yahoo fallback closes (or Stooq) to compute stats
# - Sends Gemini a compact factsheet instead of raw JSON
#
# Requires: google-genai
# Env:
#   GEMINI_API_KEY (required)
#   GEMINI_MODEL (optional, default gemini-2.5-flash)
#   GEMINI_TEMPERATURE (optional, default 0.4)
#   GEMINI_MAX_OUTPUT_TOKENS (optional, default 1200)

import os
import time
import math
import csv
import io
from typing import Optional, Tuple, List, Dict, Any

from google import genai
from google.genai import types

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.4"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1200"))

MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
BACKOFF_BASE_SECONDS = float(os.getenv("GEMINI_BACKOFF_BASE_SECONDS", "1.5"))

# Optional benchmarks for relative context (still free, from your Yahoo fallback layer)
# If you later add benchmark fetching to data_sources.py, you can pass it in bundle.
DEFAULT_BENCHMARK = os.getenv("BENCHMARK_TICKER", "^FCHI")  # CAC 40


def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


def _safe_get(d: Any, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_stooq_csv(csv_text: str) -> List[float]:
    closes: List[float] = []
    if not csv_text:
        return closes
    reader = csv.DictReader(io.StringIO(csv_text))
    for r in reader:
        try:
            c = r.get("Close")
            if c is None:
                continue
            val = float(c)
            if val > 0:
                closes.append(val)
        except Exception:
            continue
    return closes


def _extract_closes(bundle: Dict[str, Any]) -> Tuple[List[float], str]:
    """
    Prefer Yahoo fallback closes, then Stooq CSV closes.
    We intentionally DO NOT use Massive here because Massive often returns errors for EU tickers.
    """
    y = bundle.get("fallback_prices_yahoo") or {}
    y_closes = y.get("closes")
    if isinstance(y_closes, list) and len(y_closes) >= 10:
        cleaned = [float(x) for x in y_closes if x is not None]
        if len(cleaned) >= 10:
            return cleaned[-30:], "yahoo_fallback"

    st = bundle.get("fallback_prices") or {}
    csv_text = st.get("csv", "")
    st_closes = _parse_stooq_csv(csv_text)
    if len(st_closes) >= 10:
        return st_closes[-30:], "stooq_fallback"

    # last resort: try Massive aggs only if they look valid
    aggs = bundle.get("aggs_30d") or {}
    results = aggs.get("results")
    if isinstance(results, list) and results:
        closes = []
        for r in results:
            c = r.get("c")
            if isinstance(c, (int, float)) and c > 0:
                closes.append(float(c))
        if len(closes) >= 10:
            return closes[-30:], "massive_aggs_30d"

    return [], "none"


def _calc_stats(closes: List[float]) -> Dict[str, Any]:
    if len(closes) < 10:
        return {"ok": False}

    ret_30d = (closes[-1] / closes[0]) - 1.0

    daily = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev > 0:
            daily.append((cur / prev) - 1.0)

    vol_annual = None
    if len(daily) >= 2:
        mean = sum(daily) / len(daily)
        var = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)
        vol_daily = math.sqrt(var)
        vol_annual = vol_daily * math.sqrt(252)

    # simple regime
    regime = "UNKNOWN"
    if vol_annual is not None:
        if vol_annual < 0.25:
            regime = "CALM"
        elif vol_annual < 0.40:
            regime = "NORMAL"
        else:
            regime = "ELEVATED"

    return {
        "ok": True,
        "last_close": closes[-1],
        "ret_30d": ret_30d,
        "vol_annual": vol_annual,
        "regime": regime,
        "n_points": len(closes),
    }


def _make_factsheet(ticker: str, bundle: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    closes, source = _extract_closes(bundle)
    stats = _calc_stats(closes)

    # Try to pull minimal overview details if they exist (even if Massive errors, overview may be absent)
    ov = bundle.get("overview") or {}
    name = _safe_get(ov, "results", "name", default=None)
    market_cap = _safe_get(ov, "results", "market_cap", default=None)
    sector = _safe_get(ov, "results", "sic_description", default=None)

    # Clean “data quality” message
    if stats.get("ok"):
        dq = f"OK: {source} with {stats['n_points']} daily closes"
    else:
        dq = "LOW: Could not retrieve enough daily closes from Yahoo/Stooq/Massive"

    facts = []
    facts.append(f"TICKER: {ticker}")
    if name:
        facts.append(f"COMPANY: {name}")
    if sector:
        facts.append(f"INDUSTRY: {sector}")
    if market_cap:
        facts.append(f"MARKET_CAP: {market_cap}")

    facts.append(f"DATA_QUALITY: {dq}")

    if stats.get("ok"):
        facts.append(f"LAST_CLOSE: {stats['last_close']:.2f}")
        facts.append(f"RETURN_30D: {stats['ret_30d']*100:.1f}%")
        if stats["vol_annual"] is not None:
            facts.append(f"VOLATILITY_ANNUALIZED: {stats['vol_annual']*100:.1f}%")
        facts.append(f"VOL_REGIME: {stats['regime']}")

    # Avoid dumping raw JSON. Gemini should reason over this compact sheet.
    factsheet = "\n".join(facts)

    meta = {
        "source": source,
        "stats": stats,
        "name": name,
        "market_cap": market_cap,
        "sector": sector,
    }
    return factsheet, meta


def _gemini_call(system_instruction: str, user_content: str, max_output_tokens: int) -> str:
    client = _client()
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=DEFAULT_TEMPERATURE,
        max_output_tokens=max_output_tokens,
    )

    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=user_content,
                config=config,
            )
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            transient = any(x in msg for x in ["429", "rate", "quota", "timeout", "503", "unavailable", "internal"])
            if attempt >= MAX_RETRIES or not transient:
                raise
            time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))

    raise last_err or RuntimeError("Gemini call failed.")


def run_committee(ticker: str, bundle: dict) -> str:
    factsheet, meta = _make_factsheet(ticker, bundle)
    stats_ok = bool(meta["stats"].get("ok"))

    # IMPORTANT: steer Gemini away from "REJECT due to missing data"
    # We explicitly tell it to fall back to "MONITOR" if data quality is low.
    common_instructions = f"""
You are part of an investment committee. You MUST base your reasoning primarily on the FACTSHEET below.
Do NOT complain about missing upstream APIs (e.g. Massive 403). The factsheet already reflects usable data.
If DATA_QUALITY is LOW, you must output a conservative MONITOR decision, not a catastrophic rejection.
Keep outputs structured and concise.
"""

    user_context = f"""{common_instructions}

FACTSHEET:
{factsheet}
"""

    buffett = _gemini_call(
        system_instruction=(
            "You are Warren Buffett. Long-term value investing: moat, management, intrinsic value, margin of safety. "
            "If fundamentals are missing, say what additional info you'd want, but still provide a conservative stance."
        ),
        user_content=user_context + """
Output format:
BUFFETT RECOMMENDATION: [BUY/HOLD/SELL]
CONVICTION LEVEL: [1-10]
TARGET ALLOCATION: [0-25%]
TIME HORIZON: [5-10+ years]
INVESTMENT THESIS: bullet points
RISK FACTORS: bullet points
""",
        max_output_tokens=900,
    )

    munger = _gemini_call(
        system_instruction=(
            "You are Charlie Munger. Apply mental models, incentives, second-order effects. Be skeptical."
        ),
        user_content=user_context + """
Output format:
MUNGER ANALYSIS: [OPPORTUNITY/CAUTION/AVOID]
RISK RATING: [1-10]
CIRCLE OF COMPETENCE: [IN/OUT/BORDERLINE]
KEY MENTAL MODELS APPLIED: bullet points
CONTRARIAN INSIGHTS: bullet points
""",
        max_output_tokens=800,
    )

    ackman = _gemini_call(
        system_instruction=(
            "You are Bill Ackman. Focus on catalysts, governance, concentrated sizing. If data is limited, keep it high-level."
        ),
        user_content=user_context + """
Output format:
ACKMAN POSITION: [LONG/SHORT/ACTIVIST LONG/NO-TRADE]
CONVICTION LEVEL: [1-10]
CATALYST TIMELINE: [6-36 months]
TARGET POSITION SIZE: [0-20%]
ACTIVIST THESIS: bullet points
CATALYSTS: bullet points
""",
        max_output_tokens=800,
    )

    cohen = _gemini_call(
        system_instruction=(
            "You are Steve Cohen. Trading-oriented: entry/exit, stops, catalysts, sizing, risk control."
        ),
        user_content=user_context + """
Output format:
COHEN TRADE: [LONG/SHORT/HEDGE/NO-TRADE]
CONVICTION: [1-10]
POSITION SIZE: [%]
ENTRY: levels/conditions
TAKE PROFIT: levels
STOP LOSS: levels
KEY DRIVERS: bullet points
""",
        max_output_tokens=800,
    )

    dalio = _gemini_call(
        system_instruction=(
            "You are Ray Dalio. Macro regimes, cycles, diversification, correlation, risk contribution."
        ),
        user_content=user_context + """
Output format:
DALIO STRATEGY: [ALLOCATE/REDUCE/HEDGE/AVOID]
MACRO ENVIRONMENT SCORE: [1-10]
DIVERSIFICATION VALUE: [HIGH/MEDIUM/LOW]
RISK CONTRIBUTION: [LOW/MEDIUM/HIGH]
ECONOMIC SEASON: [GROWTH/RECESSION/REFLATION/STAGFLATION/UNKNOWN]
MACRO ANALYSIS: bullet points
PORTFOLIO FIT: bullet points
""",
        max_output_tokens=850,
    )

    committee_blob = f"""
=== FACTSHEET (CLEAN INPUT) ===
{factsheet}

=== BUFFETT ===
{buffett}

=== MUNGER ===
{munger}

=== ACKMAN ===
{ackman}

=== COHEN ===
{cohen}

=== DALIO ===
{dalio}
""".strip()

    risk_officer = _gemini_call(
        system_instruction=(
            "You are the Chief Risk Officer & Portfolio Manager. Synthesize the committee into one actionable decision. "
            "If DATA_QUALITY is LOW, choose MONITOR and propose what to fetch next. "
            "If DATA_QUALITY is OK, you may IMPLEMENT a small position with clear stops."
        ),
        user_content=committee_blob + f"""

Ticker: {ticker}

Output format:
PORTFOLIO DECISION: [IMPLEMENT/MODIFY/MONITOR/REJECT]
FINAL POSITION SIZE: X.X%
RISK RATING: [1-10]
EXPECTED ANNUAL RETURN: X.X% (rough estimate)
MAXIMUM EXPECTED LOSS: -X.X% (rough estimate)
COMMITTEE CONSENSUS: bullet points (who agrees/disagrees)
IMPLEMENTATION PLAN: bullet points (entry, stop, targets, monitoring)
RISK OFFICER SUMMARY: 2-3 lines
""",
        max_output_tokens=1200,
    )

    # Small hint at the top so you can see the data source without reading the whole thing
    source = meta.get("source", "unknown")
    dq = "OK" if stats_ok else "LOW"

    return f"""## {ticker} — Gemini Committee Report

_Data quality: **{dq}** | Price source: **{source}**_

{risk_officer}

---

## Full Committee Outputs

{committee_blob}
"""
