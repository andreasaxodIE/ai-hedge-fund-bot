import csv
import io
import math
from datetime import datetime

def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _parse_stooq_csv(csv_text: str):
    """
    Returns list of dict rows with Date, Close, Volume.
    """
    rows = []
    if not csv_text:
        return rows
    reader = csv.DictReader(io.StringIO(csv_text))
    for r in reader:
        try:
            close = float(r.get("Close", "") or 0)
            vol = float(r.get("Volume", "") or 0)
            date = r.get("Date")
            if close > 0 and date:
                rows.append({"date": date, "close": close, "volume": vol})
        except Exception:
            continue
    return rows

def _calc_stats_from_closes(closes):
    """
    closes: list[float] oldest->newest
    """
    if len(closes) < 5:
        return {}

    ret = (closes[-1] / closes[0]) - 1.0

    # daily returns for volatility proxy
    daily = []
    for i in range(1, len(closes)):
        if closes[i-1] > 0:
            daily.append((closes[i] / closes[i-1]) - 1.0)

    if len(daily) >= 2:
        mean = sum(daily) / len(daily)
        var = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)
        vol_daily = math.sqrt(var)
        vol_annual = vol_daily * math.sqrt(252)
    else:
        vol_daily = None
        vol_annual = None

    return {
        "period_return": ret,
        "vol_daily": vol_daily,
        "vol_annual": vol_annual,
        "last_close": closes[-1],
    }

def _extract_prices(bundle: dict):
    """
    Prefer Massive aggs if they look valid; else use Stooq fallback CSV.
    Returns (closes, source_str)
    """
    aggs = bundle.get("aggs_30d")
    results = _safe_get(aggs, "results", default=None)
    if isinstance(results, list) and results:
        closes = []
        for r in results:
            c = r.get("c")
            if isinstance(c, (int, float)) and c > 0:
                closes.append(float(c))
        if len(closes) >= 5:
            return closes, "massive_aggs_30d"

    fb = bundle.get("fallback_prices", {})
    csv_text = fb.get("csv", "")
    rows = _parse_stooq_csv(csv_text)
    rows = rows[-30:]  # last ~30 rows
    closes = [x["close"] for x in rows]
    if len(closes) >= 5:
        return closes, "stooq_fallback"

    return [], "none"

def _score_company_quality(bundle: dict):
    """
    Very rough heuristic based on what we can get without paid fundamentals.
    Uses Massive overview if available.
    """
    ov = bundle.get("overview") or {}
    name = _safe_get(ov, "results", "name", default=None)
    market_cap = _safe_get(ov, "results", "market_cap", default=None)
    sic_desc = _safe_get(ov, "results", "sic_description", default=None)

    score = 5  # neutral
    notes = []

    if market_cap:
        try:
            mc = float(market_cap)
            if mc >= 200e9:
                score += 1; notes.append("Large-cap stability")
            elif mc <= 2e9:
                score -= 1; notes.append("Small-cap risk")
        except Exception:
            pass

    if sic_desc:
        # weak signal: regulated/defensive industries tend to be steadier
        defensive_keywords = ["Utilities", "Insurance", "Banks", "Health", "Consumer Staples"]
        if any(k.lower() in sic_desc.lower() for k in defensive_keywords):
            score += 1; notes.append("Defensive industry signal")

    if not name:
        notes.append("Limited fundamentals available (free mode)")

    score = max(1, min(10, score))
    return score, notes

def _make_buffett(stats, quality_score, quality_notes):
    """
    Free-mode Buffett style: focuses on quality + volatility + margin of safety proxy.
    """
    if not stats:
        return {
            "rec": "HOLD",
            "conv": 4,
            "alloc": "0-5%",
            "thesis": [
                "Insufficient price data to estimate trend/volatility.",
                "In free mode, fundamentals may be missing."
            ],
            "risks": ["Data availability", "Model simplicity"],
        }

    vol = stats.get("vol_annual")
    ret = stats.get("period_return")

    rec = "HOLD"
    conv = 5
    alloc = "0-10%"

    # Simple interpretation:
    # - Prefer high quality score + reasonable volatility + not massively extended in 30d
    if quality_score >= 7 and vol and vol < 0.45 and ret is not None and ret < 0.25:
        rec = "BUY"; conv = 7; alloc = "5-15%"
    if vol and vol > 0.80:
        rec = "HOLD"; conv = 4; alloc = "0-5%"
    if ret is not None and ret > 0.40 and vol and vol > 0.60:
        rec = "HOLD"; conv = 4; alloc = "0-5%"

    thesis = [
        f"Quality score (proxy): {quality_score}/10",
        f"30-day return: {ret*100:.1f}%",
    ]
    if vol:
        thesis.append(f"Volatility (annualized proxy): {vol*100:.1f}%")

    thesis += [f"Note: {n}" for n in quality_notes]

    risks = [
        "This is a heuristic model (not financial advice).",
        "Limited fundamentals in free mode.",
        "Short window (30 days) can mislead on long-term value."
    ]

    return {"rec": rec, "conv": conv, "alloc": alloc, "thesis": thesis, "risks": risks}

def _make_trader(stats):
    if not stats:
        return {
            "stance": "NO-TRADE",
            "conv": 3,
            "plan": ["Not enough price history in free mode."]
        }

    last = stats["last_close"]
    ret = stats["period_return"]
    vol = stats.get("vol_annual") or 0

    # Basic momentum/trend proxy
    if ret > 0.08:
        stance = "LONG"
    elif ret < -0.08:
        stance = "SHORT"
    else:
        stance = "NO-TRADE"

    # crude position sizing: smaller if volatile
    size = "1-2%" if vol > 0.70 else "2-5%"

    stop = "5-8%" if vol > 0.70 else "3-5%"
    take = "8-15%" if vol > 0.70 else "5-10%"

    return {
        "stance": stance,
        "conv": 5 if stance != "NO-TRADE" else 4,
        "plan": [
            f"Last close: {last:.2f}",
            f"30d return: {ret*100:.1f}%",
            f"Suggested size: {size}",
            f"Stop loss: {stop}",
            f"Take profit: {take}",
            "Rule-based momentum only (free mode)."
        ]
    }

def _risk_officer(buffett, trader, data_source):
    # Simple synthesis
    if buffett["rec"] == "BUY" and trader["stance"] in ("LONG", "NO-TRADE"):
        decision = "IMPLEMENT"
        size = buffett["alloc"]
        risk = 6
    else:
        decision = "MONITOR"
        size = "0-5%"
        risk = 5

    summary = [
        f"Data source: {data_source}",
        f"Buffett proxy: {buffett['rec']} (conv {buffett['conv']}/10, alloc {buffett['alloc']})",
        f"Trader proxy: {trader['stance']} (conv {trader['conv']}/10)",
    ]

    return decision, size, risk, summary

def run_committee(ticker: str, bundle: dict) -> str:
    closes, source = _extract_prices(bundle)
    stats = _calc_stats_from_closes(closes) if closes else {}

    quality_score, quality_notes = _score_company_quality(bundle)
    buffett = _make_buffett(stats, quality_score, quality_notes)
    trader = _make_trader(stats)
    decision, size, risk, summary = _risk_officer(buffett, trader, source)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    report = []
    report.append(f"## {ticker} — FREE MODE AI Hedge Fund Report")
    report.append(f"_Generated: {now}_")
    report.append("")
    report.append("### Portfolio Decision (Rule-Based)")
    report.append(f"**PORTFOLIO DECISION:** {decision}")
    report.append(f"**FINAL POSITION SIZE:** {size}")
    report.append(f"**RISK RATING:** {risk}/10")
    report.append("")
    report.append("**RISK OFFICER SUMMARY:**")
    report += [f"- {x}" for x in summary]
    report.append("")
    report.append("---")
    report.append("### Buffett-Style (Heuristic)")
    report.append(f"**RECOMMENDATION:** {buffett['rec']}")
    report.append(f"**CONVICTION:** {buffett['conv']}/10")
    report.append(f"**TARGET ALLOCATION:** {buffett['alloc']}")
    report.append("**THESIS:**")
    report += [f"- {x}" for x in buffett["thesis"]]
    report.append("**RISKS:**")
    report += [f"- {x}" for x in buffett["risks"]]
    report.append("")
    report.append("---")
    report.append("### Trader-Style (Momentum Heuristic)")
    report.append(f"**STANCE:** {trader['stance']}")
    report.append(f"**CONVICTION:** {trader['conv']}/10")
    report.append("**PLAN:**")
    report += [f"- {x}" for x in trader["plan"]]
    report.append("")
    report.append("---")
    report.append("### Notes")
    report.append("- Free mode does not use OpenAI or any paid LLM.")
    report.append("- If Massive endpoints return 403/permission errors, the bot uses Stooq fallback prices.")
    report.append("- This is a demo / educational tool, not financial advice.")

    return "\n".join(report)
