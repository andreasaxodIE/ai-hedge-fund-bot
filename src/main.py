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
    Accepts inputs like:
      TSLA
      $TSLA
      analyze TSLA
      /analyze TSLA
    """
    if not text:
        return None
    t = text.strip().upper()

    # try to find first plausible ticker token
    # allow $TSLA too
    candidates = re.findall(r"\$?[A-Z0-9]{1,6}", t)
    if not candidates:
        return None

    # choose first, strip $
    sym = candidates[0].lstrip("$")
    if 1 <= len(sym) <= 6 and sym.isalnum():
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
    repo = os.environ["GITHUB_REPOSITORY"]  # owner/repo
    event = load_event()

    # Identify issue + sender + text depending on trigger type
    sender = (event.get("sender") or {}).get("login", "unknown")

    if not is_allowed_user(sender):
        # silently ignore (or you can comment a denial)
        return

    issue = event.get("issue")
    if not issue:
        return

    issue_number = issue["number"]

    text_source = ""
    if event_name == "issues":
        # opening an issue: use title + body
        text_source = (issue.get("title") or "") + "\n" + (issue.get("body") or "")
    elif event_name == "issue_comment":
        # new comment on an issue: use comment body
        comment = event.get("comment") or {}
        text_source = comment.get("body") or ""
    else:
        return

    ticker = extract_ticker(text_source)
    if not ticker:
        post_comment(
            repo,
            issue_number,
            "I couldn’t detect a ticker. Try: `TSLA` or `/analyze TSLA` or `$TSLA`."
        )
        return

    # Acknowledge
    post_comment(repo, issue_number, f"✅ Running AI Hedge Fund analysis for **{ticker}**…")

    # Fetch data from Massive
    bundle = fetch_market_bundle(ticker)

    # Run committee + risk officer
    report = run_committee(ticker, bundle)

    # GitHub comment limit is generous, but chunk anyway to be safe
    chunks = chunk_text(report, chunk_size=60000)

    for i, c in enumerate(chunks, start=1):
        header = f"### AI Hedge Fund Report — {ticker} (Part {i}/{len(chunks)})\n"
        post_comment(repo, issue_number, header + "\n" + c)

if __name__ == "__main__":
    main()

