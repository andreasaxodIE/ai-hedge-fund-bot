# src/agents.py
# Gemini (Google Gen AI SDK) implementation of the multi-agent “hedge fund committee”.
# Requires: pip install google-genai
# Env vars:
#   GEMINI_API_KEY   (required)
#   GEMINI_MODEL     (optional, default: gemini-2.5-flash)
#   GEMINI_TEMPERATURE (optional, default: 0.4)
#   GEMINI_MAX_OUTPUT_TOKENS (optional, default: 1200)

import os
import time
import json
from typing import Optional

from google import genai
from google.genai import types


# --------- configuration ---------

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.4"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1200"))

# Basic exponential backoff for transient errors (429/503/etc.)
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
BACKOFF_BASE_SECONDS = float(os.getenv("GEMINI_BACKOFF_BASE_SECONDS", "1.5"))


def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it as a GitHub Actions secret and pass it to the workflow.")
    return genai.Client(api_key=api_key)


def _pretty_json(obj, max_chars: int = 120_000) -> str:
    """
    Keep prompt size reasonable; Gemini can handle large context, but you don't want
    to dump megabytes of JSON and blow latency/cost.
    """
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)

    if len(s) > max_chars:
        return s[:max_chars] + "\n... (truncated) ..."
    return s


def chat(
    *,
    system_instruction: str,
    user_content: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """
    Single Gemini call with retries. Returns response text.
    """
    client = _client()

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_content,
                config=config,
            )
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            last_err = e
            # crude transient detection; Gemini SDK wraps HTTP errors
            msg = str(e).lower()
            is_transient = any(x in msg for x in ["429", "rate", "quota", "timeout", "503", "unavailable", "internal"])
            if attempt >= MAX_RETRIES or not is_transient:
                raise
            sleep_s = BACKOFF_BASE_SECONDS * (2 ** attempt)
            time.sleep(sleep_s)

    # should never reach here
    raise last_err or RuntimeError("Gemini call failed.")


def run_committee(ticker: str, bundle: dict) -> str:
    """
    Produces the same style of output you had with the OpenAI version:
      Buffett, Munger, Ackman, Cohen, Dalio + Risk Officer synthesis
    """
    context = f"""You are given a MARKET DATA BUNDLE (JSON) for {ticker}.
Extract key signals (price trend, volatility proxy, market cap if available, basic risk factors).
If some fields are missing or errors are present, explicitly say so.

MARKET BUNDLE (JSON):
{_pretty_json(bundle)}
"""

    # --- committee members ---

    buffett = chat(
        system_instruction=(
            "You are Warren Buffett. Focus on business quality, economic moat, management, "
            "intrinsic value vs price, and margin of safety. Be conservative and explainable."
        ),
        user_content=context
        + """
Output format:
BUFFETT RECOMMENDATION: [BUY/HOLD/SELL]
CONVICTION LEVEL: [1-10]
TARGET ALLOCATION: [0-25%]
TIME HORIZON: [5-10+ years]
INVESTMENT THESIS: bullet points
RISK FACTORS: bullet points
""",
        max_output_tokens=1000,
    )

    munger = chat(
        system_instruction=(
            "You are Charlie Munger. Apply mental models, incentives, second-order effects. "
            "Be skeptical and emphasize what could go wrong."
        ),
        user_content=context
        + """
Output format:
MUNGER ANALYSIS: [OPPORTUNITY/CAUTION/AVOID]
RISK RATING: [1-10]
CIRCLE OF COMPETENCE: [IN/OUT/BORDERLINE]
KEY MENTAL MODELS APPLIED: bullet points
CONTRARIAN INSIGHTS: bullet points
""",
        max_output_tokens=900,
    )

    ackman = chat(
        system_instruction=(
            "You are Bill Ackman. Focus on concentrated positions, catalysts, governance, and "
            "what would need to change for value to be unlocked."
        ),
        user_content=context
        + """
Output format:
ACKMAN POSITION: [LONG/SHORT/ACTIVIST LONG]
CONVICTION LEVEL: [1-10]
CATALYST TIMELINE: [6-36 months]
TARGET POSITION SIZE: [5-20%]
ENGAGEMENT PROBABILITY: [HIGH/MEDIUM/LOW]
ACTIVIST THESIS: bullet points
CATALYSTS: bullet points
""",
        max_output_tokens=900,
    )

    cohen = chat(
        system_instruction=(
            "You are Steve Cohen. Trading-oriented. Focus on positioning, timing, entry/exit, "
            "stops, catalysts, and risk control."
        ),
        user_content=context
        + """
Output format:
COHEN TRADE: [LONG/SHORT/HEDGE/NO-TRADE]
CONVICTION: [1-10]
POSITION SIZE: [%]
ENTRY: levels/conditions
TAKE PROFIT: levels
STOP LOSS: levels
KEY DRIVERS: bullet points
""",
        max_output_tokens=900,
    )

    dalio = chat(
        system_instruction=(
            "You are Ray Dalio. Macro regimes, cycles, diversification, correlation, and "
            "risk contribution. Think in scenarios."
        ),
        user_content=context
        + """
Output format:
DALIO STRATEGY: [ALLOCATE/REDUCE/HEDGE/AVOID]
MACRO ENVIRONMENT SCORE: [1-10]
DIVERSIFICATION VALUE: [HIGH/MEDIUM/LOW]
RISK CONTRIBUTION: [LOW/MEDIUM/HIGH]
ECONOMIC SEASON: [GROWTH/RECESSION/REFLATION/STAGFLATION/UNKNOWN]
MACRO ANALYSIS: bullet points
PORTFOLIO FIT: bullet points
""",
        max_output_tokens=900,
    )

    committee_blob = f"""
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

    # --- synthesis (Risk Officer / PM) ---

    risk_officer = chat(
        system_instruction=(
            "You are the Chief Risk Officer & Portfolio Manager. Synthesize the committee into "
            "a single actionable decision with explicit risk controls. If data quality is weak, "
            "reduce position sizing and confidence."
        ),
        user_content=committee_blob
        + f"""

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

    return f"""## {ticker} — Gemini Committee Report

{risk_officer}

---

## Full Committee Outputs

{committee_blob}
"""
