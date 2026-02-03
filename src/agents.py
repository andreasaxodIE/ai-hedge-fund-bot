import csv
import io
import math
from datetime import datetime

# ---------- helpers ----------

def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _parse_stooq_csv(csv_text: str):
    rows = []
    if not csv_text:
        return rows
    reader = csv.DictReader(io.StringIO(csv_text))
    for r in reader:
        try:
            close = float(r.get("Close", "") or 0)
            date = r.get("Date")
            if close > 0 and date:
                rows.append({"date": date, "close": close})
        except Exception:
            continue
    return rows

# ---------- price extraction ----------

def _extract_prices(bundle: dict):
    # 1) Yahoo fallback (best global coverage)
    y = bundle.get("fallback_prices_yahoo") or {}
    y_closes = y.get("closes")
    if isinstance(y_closes, list) and len(y_closes) >= 10:
        return [float(x) for x in y_closes if x], "yahoo_fallback"

    # 2) Stooq fallback
    fb = bundle.get("fallback_prices", {})
    rows = _parse_stooq_csv(fb.get("csv", ""))
    rows = rows[-30:]
    closes = [x["close"] for x in rows]
    if len(closes) >= 10:
        return closes, "stooq_fallback"

    return [], "none"

# ---------- stats ----------

def _calc_stats(closes):
    if len(closes) < 10:
        return {}

    ret_30d = (closes[-1] / closes[0]) - 1.0

    daily = [(closes[i] / closes[i-1]) - 1.0 for i in range(1, len(closes)) if closes[i-1] > 0]
    vol = math.sqrt(sum((x - sum(daily)/len(daily))**2 for x in daily)/(len(daily)-1)) * math.sqrt(252)

    return {
        "last": closes[-1],
        "ret_30d": ret_30d,
        "vol": vol
    }

# ---------- fundamentals (bank-aware, free) ----------

def _bank_quality_score(bundle):
    score = 5
    notes = []

    ov = bundle.get("overview") or {}
    mcap = _safe_get(ov, "results", "market_cap")
    sector = _safe_get(ov, "results", "sic_description", "")

    if mcap:
        try:
            if float(mcap) > 50e9:
                score += 1; notes.append("Large-cap European bank")
        except Exception:
            pass

    if "bank" in sector.lower():
        score += 1; notes.append("Core banking business")

    # conservative cap
    score = max(1, min(8, score))
    return score, notes

# ---------- Buffett heuristic ----------

def _buffett(stats, q_score, q_notes):
    if not stats:
        return {"rec": "HOLD", "conv": 4, "alloc": "0–5%", "thesis": ["Insufficient price data"], "risks": ["Data limits"]}

    ret = stats["ret_30d"]
    vol = stats["vol"]

    rec = "HOLD"
    alloc = "0–10%"
    conv = 5

    if q_score >= 6 and ret < 0.25 and vol < 0.35:
        rec = "BUY"; alloc = "5–15%"; conv = 7

    return {
        "rec": rec,
        "conv": conv,
        "alloc": alloc,
        "thesis": [
            f"Quality score (bank-aware): {q_score}/8",
            f"30d return: {ret*100:.1f}%",
            f"Volatility: {vol*100:.1f}%"
        ] + q_notes,
        "risks": [
            "Short lookback window",
            "Free-mode fundamentals",
            "Banking sector cyclicality"
        ]
    }

# ---------- trader heuristic ----------

def _trader(stats):
    if not stats:
        return {"stance": "NO-TRADE", "conv": 3, "levels": []}

    ret = stats["ret_30d"]
    last = stats["last"]
    vol = stats["vol"]

    stance = "LONG" if ret > 0.05 else "NO-TRADE"

    stop = last * (1 - 0.04)
    tp = last * (1 + 0.08)

    return {
        "stance": stance,
        "conv": 5 if stance == "LONG" else 4,
        "levels": [
            f"Entry ~ {last:.2f}",
            f"Stop ~ {stop:.2f}",
            f"Target ~ {tp:.2f}"
        ]
    }

# ---------- risk officer ----------

def _risk_officer(buffett, trader, stats, source):
    vol = stats.get("vol", 0)

    regime = "Normal volatility" if vol < 0.35 else "Elevated volatility"

    if buffett["rec"] == "BUY" and trader["stance"] == "LONG":
        decision = "IMPLEMENT (TACTICAL)"
        size = "2–5%"
    elif buffett["rec"] == "HOLD" and trader["stance"] == "LONG":
        decision = "IMPLEMENT (SMALL)"
        size = "1–3%"
    else:
        decision = "MONITOR"
        size = "0–5%"

    return decision, size, regime, [
        f"Data source: {source}",
        f"Buffett: {buffett['rec']} ({buffett['alloc']})",
        f"Trader: {trader['stance']}",
        f"Market regime: {regime}"
    ]

# ---------- main ----------

def run_committee(ticker: str, bundle: dict) -> str:
    closes, source = _extract_prices(bundle)
    stats = _calc_stats(closes) if closes else {}

    q_score, q_notes = _bank_quality_score(bundle)
    buffett = _buffett(stats, q_score, q_notes)
    trader = _trader(stats)

    decision, size, regime, summary = _risk_officer(buffett, trader, stats, source)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    out = []
    out.append(f"## {ticker} — FREE MODE AI Hedge Fund Report")
    out.append(f"_Generated: {now}_\n")

    out.append("### Portfolio Decision")
    out.append(f"**DECISION:** {decision}")
    out.append(f"**POSITION SIZE:** {size}")
    out.append(f"**MARKET REGIME:** {regime}\n")

    out.append("**RISK OFFICER SUMMARY:**")
    out += [f"- {x}" for x in summary]

    out.append("\n---\n### Buffett-Style View")
    out.append(f"**Recommendation:** {buffett['rec']} (conv {buffett['conv']}/10)")
    out.append(f"**Allocation:** {buffett['alloc']}")
    out.append("**Thesis:**")
    out += [f"- {x}" for x in buffett["thesis"]]
    out.append("**Risks:**")
    out += [f"- {x}" for x in buffett["risks"]]

    out.append("\n---\n### Trader View")
    out.append(f"**Stance:** {trader['stance']} (conv {trader['conv']}/10)")
    if trader["levels"]:
        out.append("**Levels:**")
        out += [f"- {x}" for x in trader["levels"]]

    out.append("\n---\n### Notes")
    out.append("- Fully free, rule-based model")
    out.append("- Yahoo Finance used for global price coverage")
    out.append("- Educational/demo use only")

    return "\n".join(out)
