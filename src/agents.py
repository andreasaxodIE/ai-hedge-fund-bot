import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def chat(system: str, user: str, model="gpt-4o-mini") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content

def run_committee(ticker: str, bundle: dict) -> str:
    context = f"""You are given a MARKET DATA BUNDLE (JSON) from Massive.com for {ticker}.
Use it to extract key financial/market details. If fields are missing, say so.

MARKET BUNDLE (JSON):
{bundle}
"""

    buffett = chat(
        "You are Warren Buffett. Long-term value investing: moat, management, intrinsic value, margin of safety.",
        context + """
Output:
BUFFETT RECOMMENDATION: [BUY/HOLD/SELL]
CONVICTION LEVEL: [1-10]
TARGET ALLOCATION: [0-25%]
TIME HORIZON: [5-10+ years]
INVESTMENT THESIS: bullets
RISK FACTORS: bullets
BUFFETT WISDOM: one short quote
"""
    )

    munger = chat(
        "You are Charlie Munger. Mental models, incentives, second-order effects. Be skeptical.",
        context + """
Output:
MUNGER ANALYSIS: [OPPORTUNITY/CAUTION/AVOID]
RISK RATING: [1-10]
MENTAL MODEL SCORE: [1-10]
CIRCLE OF COMPETENCE: [IN/OUT/BORDERLINE]
KEY MENTAL MODELS APPLIED: bullets
CONTRARIAN INSIGHTS: bullets
MUNGER WISDOM: one short quote
"""
    )

    ackman = chat(
        "You are Bill Ackman. Activist/catalyst investing, governance improvements, concentrated sizing.",
        context + """
Output:
ACKMAN POSITION: [LONG/SHORT/ACTIVIST LONG]
CONVICTION LEVEL: [1-10]
CATALYST TIMELINE: [6-36 months]
TARGET POSITION SIZE: [5-20%]
ENGAGEMENT PROBABILITY: [HIGH/MEDIUM/LOW]
ACTIVIST THESIS: bullets
CATALYSTS: bullets
ENGAGEMENT STRATEGY: bullets
"""
    )

    cohen = chat(
        "You are Steve Cohen. Trading-focused: entry/exit, sizing, stops, key drivers.",
        context + """
Output:
COHEN TRADE: [LONG/SHORT/HEDGE/NO-TRADE]
CONVICTION: [1-10]
POSITION SIZE: [%]
ENTRY: levels/conditions
TAKE PROFIT: levels
STOP LOSS: levels
KEY DRIVERS: bullets
"""
    )

    dalio = chat(
        "You are Ray Dalio. Macro regimes, cycles, diversification, correlation, risk contribution.",
        context + """
Output:
DALIO STRATEGY: [ALLOCATE/REDUCE/HEDGE/AVOID]
MACRO ENVIRONMENT SCORE: [1-10]
DIVERSIFICATION VALUE: [HIGH/MEDIUM/LOW]
RISK CONTRIBUTION: [LOW/MEDIUM/HIGH]
ECONOMIC SEASON: [GROWTH/RECESSION/REFLATION/STAGFLATION]
MACRO ANALYSIS: bullets
PORTFOLIO FIT: bullets
"""
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

    risk_officer = chat(
        "You are the Chief Risk Officer & Portfolio Manager. Synthesize the committee into a single decision with risk controls.",
        committee_blob + f"""

Ticker: {ticker}

Output:
PORTFOLIO DECISION: [IMPLEMENT/MODIFY/MONITOR/REJECT]
FINAL POSITION SIZE: X.X%
RISK RATING: [1-10]
EXPECTED ANNUAL RETURN: X.X% (rough)
MAXIMUM EXPECTED LOSS: -X.X% (rough)
COMMITTEE CONSENSUS: bullets (who agrees/disagrees)
IMPLEMENTATION PLAN: bullets (entry, stop, targets, monitoring)
RISK OFFICER SUMMARY: 2-3 lines
"""
    )

    return f"""## {ticker} — Risk Officer Decision

{risk_officer}

---

## Full Committee Outputs

{committee_blob}
"""

