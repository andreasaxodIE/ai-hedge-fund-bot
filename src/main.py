import os
import json
import re
import requests
from data_sources import fetch_market_bundle
from agents import run_committee
from formatters import chunk_text

GITHUB_API = "https://api.github.com"

def load_event():
    path = os.environ["GITHUB_EVENT_PATH"]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def gh_headers():
    token = os.environ["GITHUB_TOKEN"]
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def post_comment(repo_full_name: str, issue_number: int, body: str):
    url = f"{GITHUB_API}/repos/{repo_full_name}/issues/{issue_number}/comments"
    r = requests.post(url, headers=gh_headers(), json={"body": body}, timeout=60)
    r.raise_for_status()
    return r.json()

def extract_ticker(text: str) -> str | None:
    """
    Supports:
      TSLA
      BNPQY
      BNP.PA
      AIR.PA
      NESN.SW
      $BNP.PA
      /analyze BNP.PA
    """
    if not text:
        return None

    t = text.strip().upper()

    # allow tickers with optional dot suffix (e.g. BNP.PA, NESN.SW)
    candidates = re.findall(r"\$?[A-Z0-9]{1,6}(?:\.[A-Z0-9]{1,4})?", t)
    if not candidates:
        return None

    sym = candidates[0].lstrip("$")

    # final sanity check
    if 1 <= len(sym) <= 12 and sym.replace(".", "").isalnum():
        return sym

    return None


    t = text.strip().upper()

    # allow tickers with optional dot suffix (e.g. BNP.PA, NESN.SW)
    candidates = re.findall(r"\$?[A-Z0-9]{1,6}(?:\.[A-Z0-9]{1,4})?", t)
    if not candidates:
        return None

    sym = candidates[0].lstrip("$")

    # final sanity check
    if 1 <= len(sym) <= 12 and sym.replace(".", "").isalnum():
        return sym

    return None


def is_allowed_user(sender_login: str) -> bool:
    allowed = os.environ.get("ALLOWED_USERS", "").strip()
    if not allowed:
        return True
    allowed_set = {u.strip().lower() for u in allowed.split(",") if u.strip()}
    return sender_login.lower() in allowed_set

def main():
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    repo = os.environ["GITHUB_REPOSITORY"]
    event = load_event()

    sender = (event.get("sender") or {}).get("login", "unknown")
    if not is_allowed_user(sender):
        return

    issue = event.get("issue")
    if not issue:
        return
    issue_number = issue["number"]

    if event_name == "issues":
        text_source = (issue.get("title") or "") + "\n" + (issue.get("body") or "")
    elif event_name == "issue_comment":
        comment = event.get("comment") or {}
        text_source = comment.get("body") or ""
    else:
        return

    ticker = extract_ticker(text_source)
    if not ticker:
        post_comment(repo, issue_number, "I couldn’t detect a ticker. Try: `TSLA` or `$TSLA`.")
        return

    post_comment(repo, issue_number, f"✅ Running AI Hedge Fund analysis for **{ticker}**…")

    # Fetch data safely
    try:
        bundle = fetch_market_bundle(ticker)
    except Exception as e:
        post_comment(
            repo,
            issue_number,
            f"❌ Data fetch failed for **{ticker}**.\n\n"
            f"Error: `{type(e).__name__}: {e}`\n\n"
            f"Check your `MASSIVE_API_KEY` secret and whether your plan allows the snapshot/aggs endpoints."
        )
        return

    # Run committee safely
    try:
        report = run_committee(ticker, bundle)
    except Exception as e:
        post_comment(
            repo,
            issue_number,
            f"❌ LLM analysis failed for **{ticker}**.\n\n"
            f"Error: `{type(e).__name__}: {e}`\n\n"
            f"Check `OPENAI_API_KEY` secret and model access."
        )
        return

    chunks = chunk_text(report, chunk_size=60000)
    for i, c in enumerate(chunks, start=1):
        header = f"### AI Hedge Fund Report — {ticker} (Part {i}/{len(chunks)})\n"
        post_comment(repo, issue_number, header + "\n" + c)

if __name__ == "__main__":
    main()
