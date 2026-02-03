import os
import time
import math
import csv
import io
import re
from typing import Optional, Tuple, List, Dict, Any

from google import genai
from google.genai import types

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.4"))

MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
BACKOFF_BASE_SECONDS = float(os.getenv("GEMINI_BACKOFF_BASE_SECONDS", "1.5"))

# Token budgets (tuned for reliability)
TOKENS_AGENT = int(os.getenv("GEMINI_AGENT_TOKENS", "850"))
TOKENS_CRO = int(os.getenv("GEMINI_CRO_TOKENS", "1700"))
TOKENS_REPAIR = int(os.getenv("GEMINI_REPAIR_TOKENS", "650"))

# How many repair passes allowed per section
MAX_REPAIR_PASSES = int(os.getenv("GEMINI_MAX_REPAIR_PASSES", "2"))


# -------------------------
# Gemini Client + Call
# -------------------------

def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


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


# -------------------------
# Data helpers
# -------------------------

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
    Prefer Yahoo chart closes, then Stooq CSV closes, then Massive aggs closes.
    """
    y = bundle.get("yahoo_chart") or {}
    y_closes = y.get("closes")
    if isinstance(y_closes, list) and len(y_closes) >= 10:
        cleaned = [float(x) for x in y_closes if x is not None]
        if len(cleaned) >= 10:
            return cleaned[-30:], "yahoo_chart"

    st = bundle.get("stooq") or {}
    csv_text = st.get("csv", "")
    st_closes = _parse_stooq_csv(csv_text)
    if len(st_closes) >= 10:
        return st_closes[-30:], "stooq"

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


def _daily_returns_from_closes(closes: List[float]) -> List[float]:
    rets: List[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev and prev > 0:
            rets.append((cur / prev) - 1.0)
    return rets


def _max_drawdown(closes: List[float]) -> Optional[float]:
    if len(closes) < 3:
        return None
    peak = closes[0]
    mdd = 0.0
    for p in closes[1:]:
        if p > peak:
            peak = p
        dd = (p / peak) - 1.0
        if dd < mdd:
            mdd = dd
    return mdd


def _best_worst_day(daily_rets: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not daily_rets:
        return None, None
    return max(daily_rets), min(daily_rets)


def _corr(x: List[float], y: List[float]) -> Optional[float]:
    n = min(len(x), len(y))
    if n < 5:
        return None
    x = x[-n:]
    y = y[-n:]
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (n - 1)
    vx = sum((a - mx) ** 2 for a in x) / (n - 1)
    vy = sum((b - my) ** 2 for b in y) / (n - 1)
    if vx <= 0 or vy <= 0:
        return None
    return cov / (math.sqrt(vx) * math.sqrt(vy))


def _beta(asset_rets: List[float], bench_rets: List[float]) -> Optional[float]:
    n = min(len(asset_rets), len(bench_rets))
    if n < 5:
        return None
    a = asset_rets[-n:]
    b = bench_rets[-n:]
    mb = sum(b) / n
    vb = sum((x - mb) ** 2 for x in b) / (n - 1)
    if vb <= 0:
        return None
    ma = sum(a) / n
    cov = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b)) / (n - 1)
    return cov / vb


def _calc_stats(closes: List[float]) -> Dict[str, Any]:
    if len(closes) < 10:
        return {"ok": False}

    ret_30d = (closes[-1] / closes[0]) - 1.0
    daily = _daily_returns_from_closes(closes)

    vol_annual = None
    if len(daily) >= 2:
        mean = sum(daily) / len(daily)
        var = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)
        vol_daily = math.sqrt(var)
        vol_annual = vol_daily * math.sqrt(252)

    regime = "UNKNOWN"
    if vol_annual is not None:
        if vol_annual < 0.25:
            regime = "CALM"
        elif vol_annual < 0.40:
            regime = "NORMAL"
        else:
            regime = "ELEVATED"

    mdd = _max_drawdown(closes)
    best_day, worst_day = _best_worst_day(daily)

    return {
        "ok": True,
        "last_close": closes[-1],
        "ret_30d": ret_30d,
        "vol_annual": vol_annual,
        "regime": regime,
        "n_points": len(closes),
        "max_drawdown_30d": mdd,
        "best_day": best_day,
        "worst_day": worst_day,
    }


# -------------------------
# Factsheet builder
# -------------------------

def _make_factsheet(ticker: str, bundle: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    closes, source = _extract_closes(bundle)
    stats = _calc_stats(closes) if closes else {"ok": False}

    yq = bundle.get("yahoo_quote") or {}
    yq_data = yq.get("data") if isinstance(yq, dict) else {}
    yq_data = yq_data if isinstance(yq_data, dict) else {}

    name = yq_data.get("longName") or yq_data.get("shortName")
    market_cap = yq_data.get("marketCap")
    currency = yq_data.get("currency")
    exchange = yq_data.get("exchange")
    sector = yq_data.get("sector")
    industry = yq_data.get("industry")

    benchmark = bundle.get("benchmark", "^GSPC")
    bench_bundle = bundle.get("benchmark_chart") or {}
    bench_closes = bench_bundle.get("closes") if isinstance(bench_bundle, dict) else None
    bench_closes = bench_closes if isinstance(bench_closes, list) else []
    bench_stats = _calc_stats([float(x) for x in bench_closes]) if bench_closes else {"ok": False}

    rel = None
    corr = None
    beta = None
    if stats.get("ok") and bench_stats.get("ok"):
        asset_rets = _daily_returns_from_closes(closes)
        bench_rets = _daily_returns_from_closes([float(x) for x in bench_closes])
        corr = _corr(asset_rets, bench_rets)
        beta = _beta(asset_rets, bench_rets)
        rel = stats["ret_30d"] - bench_stats["ret_30d"]

    dq = f"OK: {source} with {stats.get('n_points', 0)} daily closes" if stats.get("ok") else "LOW: not enough daily closes"

    facts = []
    facts.append(f"TICKER: {ticker}")
    if name:
        facts.append(f"NAME: {name}")
    if exchange:
        facts.append(f"EXCHANGE: {exchange}")
    if currency:
        facts.append(f"CURRENCY: {currency}")
    if sector:
        facts.append(f"SECTOR: {sector}")
    if industry:
        facts.append(f"INDUSTRY: {industry}")
    if market_cap:
        facts.append(f"MARKET_CAP: {market_cap}")

    facts.append(f"DATA_QUALITY: {dq}")

    if stats.get("ok"):
        facts.append(f"LAST_CLOSE: {stats['last_close']:.2f}")
        facts.append(f"RETURN_30D: {stats['ret_30d']*100:.1f}%")
        if stats["vol_annual"] is not None:
            facts.append(f"VOLATILITY_ANNUALIZED: {stats['vol_annual']*100:.1f}%")
        facts.append(f"VOL_REGIME: {stats['regime']}")
        if stats.get("max_drawdown_30d") is not None:
            facts.append(f"MAX_DRAWDOWN_30D: {stats['max_drawdown_30d']*100:.1f}%")
        if stats.get("best_day") is not None:
            facts.append(f"BEST_DAY: {stats['best_day']*100:.1f}%")
        if stats.get("worst_day") is not None:
            facts.append(f"WORST_DAY: {stats['worst_day']*100:.1f}%")

    facts.append(f"BENCHMARK: {benchmark}")
    if bench_stats.get("ok"):
        facts.append(f"BENCHMARK_RETURN_30D: {bench_stats['ret_30d']*100:.1f}%")
        if rel is not None:
            facts.append(f"RELATIVE_RETURN_30D: {rel*100:.1f}%")
        if corr is not None:
            facts.append(f"CORRELATION_TO_BENCH: {corr:.2f}")
        if beta is not None:
            facts.append(f"BETA_TO_BENCH: {beta:.2f}")
    else:
        facts.append("BENCHMARK_DATA: LOW")

    meta = {"source": source, "stats": stats, "bench_stats": bench_stats}
    return "\n".join(facts), meta


# -------------------------
# Validators + Repair
# -------------------------

def _count_bullets(block: str) -> int:
    return len(re.findall(r"(?m)^\s*-\s+", block or ""))


def _has_fields(text: str, fields: List[str]) -> bool:
    return all(f in (text or "") for f in fields)


def _repair_section(section_name: str, factsheet: str, draft: str, required_fields: List[str], min_bullets: int) -> str:
    prompt = f"""
You are repairing a section named {section_name}.
Rules:
- Use ONLY the FACTSHEET (no invented facts).
- Output MUST contain ALL required fields exactly as labels.
- Any missing/unknown value => write UNKNOWN (but keep the field).
- Any bullet section must have at least {min_bullets} bullets.
- Keep concise.

FACTSHEET:
{factsheet}

DRAFT OUTPUT TO REPAIR:
{draft}

REQUIRED FIELDS (must all appear):
{chr(10).join(required_fields)}

Return ONLY the repaired section.
"""
    return _gemini_call(
        system_instruction="You are a strict formatter. Never omit required fields.",
        user_content=prompt,
        max_output_tokens=TOKENS_REPAIR,
    ).strip()


def _ensure_valid(section_name: str, factsheet: str, draft: str, required_fields: List[str], min_bullets: int) -> str:
    text = (draft or "").strip()
    for _ in range(MAX_REPAIR_PASSES + 1):
        ok_fields = _has_fields(text, required_fields)
        ok_bullets = _count_bullets(text) >= min_bullets
        if ok_fields and ok_bullets:
            return text
        text = _repair_section(section_name, factsheet, text, required_fields, min_bullets)
    return text


# -------------------------
# Committee runner
# -------------------------

def run_committee(ticker: str, bundle: dict) -> str:
    factsheet, meta = _make_factsheet(ticker, bundle)
    stats_ok = bool(meta["stats"].get("ok"))
    benchmark = bundle.get("benchmark", "^GSPC")
    vol_regime = meta.get("stats", {}).get("regime", "UNKNOWN")

    common = f"""
You are part of an investment committee.
STRICT RULES:
- Use ONLY the FACTSHEET (no outside knowledge, no news).
- If DATA_QUALITY is LOW => conservative MONITOR.
- For any bullet list: minimum 3 bullets.
- If you don't know something: write UNKNOWN.
- Keep concise and structured.
FACTSHEET:
{factsheet}
"""

    buffett_raw = _gemini_call(
        system_instruction="You are Warren Buffett. Value investing, moat, quality, patience.",
        user_content=common + """
Output format (exact labels required):
BUFFETT RECOMMENDATION: [BUY/HOLD/SELL/MONITOR]
CONVICTION LEVEL: [1-10]
TARGET ALLOCATION: [0-25%]
TIME HORIZON: [5-10+ years/UNKNOWN]
INVESTMENT THESIS:
- ...
RISK FACTORS:
- ...
""",
        max_output_tokens=TOKENS_AGENT,
    )

    buffett = _ensure_valid(
        "BUFFETT",
        factsheet,
        buffett_raw,
        required_fields=[
            "BUFFETT RECOMMENDATION:",
            "CONVICTION LEVEL:",
            "TARGET ALLOCATION:",
            "TIME HORIZON:",
            "INVESTMENT THESIS:",
            "RISK FACTORS:",
        ],
        min_bullets=6,  # 3 thesis + 3 risks
    )

    munger_raw = _gemini_call(
        system_instruction="You are Charlie Munger. Mental models, incentives, second-order thinking.",
        user_content=common + """
Output format (exact labels required):
MUNGER ANALYSIS: [OPPORTUNITY/CAUTION/AVOID/MONITOR]
RISK RATING: [1-10]
CIRCLE OF COMPETENCE: [IN/OUT/BORDERLINE]
KEY MENTAL MODELS APPLIED:
- ...
CONTRARIAN INSIGHTS:
- ...
""",
        max_output_tokens=TOKENS_AGENT,
    )

    munger = _ensure_valid(
        "MUNGER",
        factsheet,
        munger_raw,
        required_fields=[
            "MUNGER ANALYSIS:",
            "RISK RATING:",
            "CIRCLE OF COMPETENCE:",
            "KEY MENTAL MODELS APPLIED:",
            "CONTRARIAN INSIGHTS:",
        ],
        min_bullets=6,
    )

    ackman_raw = _gemini_call(
        system_instruction="You are Bill Ackman. Catalysts, governance, sizing discipline.",
        user_content=common + """
Output format (exact labels required):
ACKMAN POSITION: [LONG/SHORT/ACTIVIST LONG/NO-TRADE/MONITOR]
CONVICTION LEVEL: [1-10]
CATALYST TIMELINE: [6-36 months/UNKNOWN]
TARGET POSITION SIZE: [0-20%/UNKNOWN]
THESIS:
- ...
CATALYSTS:
- ...
""",
        max_output_tokens=TOKENS_AGENT,
    )

    ackman = _ensure_valid(
        "ACKMAN",
        factsheet,
        ackman_raw,
        required_fields=[
            "ACKMAN POSITION:",
            "CONVICTION LEVEL:",
            "CATALYST TIMELINE:",
            "TARGET POSITION SIZE:",
            "THESIS:",
            "CATALYSTS:",
        ],
        min_bullets=6,
    )

    cohen_raw = _gemini_call(
        system_instruction="You are Steve Cohen. Tactical trading, risk control, stops.",
        user_content=common + """
Output format (exact labels required):
COHEN TRADE: [LONG/SHORT/HEDGE/NO-TRADE/MONITOR]
CONVICTION: [1-10]
POSITION SIZE: [%]
ENTRY:
- ...
TAKE PROFIT:
- ...
STOP LOSS:
- ...
KEY DRIVERS:
- ...
""",
        max_output_tokens=TOKENS_AGENT,
    )

    cohen = _ensure_valid(
        "COHEN",
        factsheet,
        cohen_raw,
        required_fields=[
            "COHEN TRADE:",
            "CONVICTION:",
            "POSITION SIZE:",
            "ENTRY:",
            "TAKE PROFIT:",
            "STOP LOSS:",
            "KEY DRIVERS:",
        ],
        min_bullets=12,  # 3 bullets per list x4
    )

    dalio_raw = _gemini_call(
        system_instruction="You are Ray Dalio. Macro regimes, risk balance, diversification.",
        user_content=common + """
Output format (exact labels required):
DALIO STRATEGY: [ALLOCATE/REDUCE/HEDGE/AVOID/MONITOR]
MACRO ENVIRONMENT SCORE: [1-10]
DIVERSIFICATION VALUE: [HIGH/MEDIUM/LOW]
RISK CONTRIBUTION: [LOW/MEDIUM/HIGH]
ECONOMIC SEASON: [GROWTH/RECESSION/REFLATION/STAGFLATION/UNKNOWN]
MACRO ANALYSIS:
- ...
PORTFOLIO FIT:
- ...
""",
        max_output_tokens=TOKENS_AGENT,
    )

    dalio = _ensure_valid(
        "DALIO",
        factsheet,
        dalio_raw,
        required_fields=[
            "DALIO STRATEGY:",
            "MACRO ENVIRONMENT SCORE:",
            "DIVERSIFICATION VALUE:",
            "RISK CONTRIBUTION:",
            "ECONOMIC SEASON:",
            "MACRO ANALYSIS:",
            "PORTFOLIO FIT:",
        ],
        min_bullets=6,
    )

    # CRO: include HARD risk rules so it doesn't output nonsense
    cro_prompt = f"""
You are the Chief Risk Officer & Portfolio Manager.
You must output all required fields.

Hard rules (apply deterministically):
- If DATA_QUALITY is LOW => PORTFOLIO DECISION must be MONITOR and FINAL POSITION SIZE must be 0.0%.
- If VOL_REGIME is ELEVATED and decision is IMPLEMENT/MODIFY => FINAL POSITION SIZE must be <= 2.0% and include a hard STOP LOSS in plan.
- MAXIMUM EXPECTED LOSS must be a negative percent like -12.0%.
- EXPECTED ANNUAL RETURN must be a percent like 15.0%.
- Do not invent fundamentals or news; use only FACTSHEET + committee.

FACTSHEET:
{factsheet}

COMMITTEE (verbatim):
BUFFETT:
{buffett}

MUNGER:
{munger}

ACKMAN:
{ackman}

COHEN:
{cohen}

DALIO:
{dalio}

Output format (exact labels required):
PORTFOLIO DECISION: [IMPLEMENT/MODIFY/MONITOR/REJECT]
FINAL POSITION SIZE: X.X%
RISK RATING: [1-10]
EXPECTED ANNUAL RETURN: X.X%
MAXIMUM EXPECTED LOSS: -X.X%
COMMITTEE CONSENSUS:
- ...
IMPLEMENTATION PLAN:
- ...
RISK OFFICER SUMMARY: 2-3 lines
"""

    cro_raw = _gemini_call(
        system_instruction="You are a strict CRO. Never omit fields. Follow hard rules.",
        user_content=cro_prompt,
        max_output_tokens=TOKENS_CRO,
    )

    cro = _ensure_valid(
        "CRO",
        factsheet,
        cro_raw,
        required_fields=[
            "PORTFOLIO DECISION:",
            "FINAL POSITION SIZE:",
            "RISK RATING:",
            "EXPECTED ANNUAL RETURN:",
            "MAXIMUM EXPECTED LOSS:",
            "COMMITTEE CONSENSUS:",
            "IMPLEMENTATION PLAN:",
            "RISK OFFICER SUMMARY:",
        ],
        min_bullets=6,  # 3 consensus + 3 plan
    )

    dq = "OK" if stats_ok else "LOW"
    source = meta.get("source", "unknown")

    full_committee = f"""
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

    return f"""## {ticker} — Gemini Committee Report

_Data quality: **{dq}** | Price source: **{source}** | Benchmark: **{benchmark}** | Vol regime: **{vol_regime}**_

{cro}

---

## Full Committee Outputs

{full_committee}
"""
