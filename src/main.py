import os
import json
import re
import requests
from typing import Optional, Tuple

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


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def extract_ticker(text: str) -> Optional[str]:
    """
    Robust ticker extraction from issue title/body or comment.

    Supported examples:
      TSLA
      BNP.PA
      AIR.PA
      NESN.SW
      BRK-B
      BTC-USD
      ^GSPC
      ^FCHI
      $BNP.PA
      /analyze BNP.PA
      analyze TSLA
    """
    if not text:
        return None

    t = _normalize_spaces(text).upper()

    # Prefer explicit "analyze" commands
    m = re.search(r"(?:^|[\s>])(?:/ANALYZE|ANALYSE|ANALYZE:|ANALYSE:|ANALYZE\s|ANALYSE\s)([^\s]+)", t)
    if m:
        cand = m.group(1).strip().rstrip(".,;:!?)\"]}'")
        sym = cand.lstrip("$")
        if _is_plausible_symbol(sym):
            return sym

    # Otherwise scan for plausible symbols; take first good one
    # Pattern allows: optional ^, alnum, optional separators . - with alnum chunks
    candidates = re.findall(r"\^?\$?[A-Z0-9]{1,10}(?:[.\-][A-Z0-9]{1,10}){0,3}", t)
    for c in candidates:
        sym = c.lstrip("$")
        sym = sym.rstrip(".,;:!?)\"]}'")
        if _is_plausible_symbol(sym):
            return sym

    return None


def _is_plausible_symbol(sym: str) -> bool:
    if not sym:
        return False

    # allow leading ^
    if sym.startswith("^"):
        core = sym[1:]
        if not core:
            return False
    else:
        core = sym

    # must be made of A-Z0-9 . -
    if not re.fullmatch(r"[A-Z0-9.\-]+", core):
        return False

    # avoid extremely long noise
    if len(core) < 1 or len(core) > 20:
        return False

    # must contain at least one letter or digit; already true
    # disallow sequences that are only dashes/dots
    if all(ch in ".-" for ch in core):
        return False

    return True


def is_allowed_user(sender_login: str) -> bool:
    allowed = os.environ.get("ALLOWED_USERS", "").strip()
    if not allowed:
        return True
    allowed_set = {u.strip().lower() for u in allowed.split(",") if u.strip()}
    return sender_login.lower() in allowed_set


def _get_issue_and_number(event: dict) -> Tuple[Optional[dict], Optional[int]]:
    issue = event.get("issue")
    if not issue:
        return None, None
    return issue, issue.get("number")


def main():
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    repo = os.environ["GITHUB_REPOSITORY"]
    event = load_event()

    sender = (event.get("sender") or {}).get("login", "unknown")
    if not is_allowed_user(sender):
        return

    issue, issue_number = _get_issue_and_number(event)
    if not issue or not issue_number:
        return

    if event_name == "issues":
        text_source = (issue.get("title") or "") + "\n" + (issue.get("body") or "")
    elif event_name == "issue_comment":
        comment = event.get("comment") or {}
        text_source = comment.get("body") or ""
    else:
        return

    ticker = extract_ticker(text_source)
    if not ticker:
        post_comment(
            repo,
            issue_number,
            "I couldn’t detect a ticker.\n\nTry examples:\n- `TSLA`\n- `BNP.PA`\n- `BRK-B`\n- `BTC-USD`\n- `^FCHI`\n- `/analyze AIR.PA`"
        )
        return

    post_comment(repo, issue_number, f"✅ Running Gemini committee analysis for **{ticker}**…")

    try:
        bundle = fetch_market_bundle(ticker)
    except Exception as e:
        post_comment(
            repo,
            issue_number,
            f"❌ Data fetch failed for **{ticker}**.\n\n"
            f"Error: `{type(e).__name__}: {e}`\n\n"
            f"Check connectivity and optional `MASSIVE_API_KEY`."
        )
        return

    try:
        report = run_committee(ticker, bundle)
    except Exception as e:
        post_comment(
            repo,
            issue_number,
            f"❌ Gemini analysis failed for **{ticker}**.\n\n"
            f"Error: `{type(e).__name__}: {e}`\n\n"
            f"Check `GEMINI_API_KEY` secret, Gemini quota, and model name."
        )
        return

    chunks = chunk_text(report, chunk_size=60000)
    for i, c in enumerate(chunks, start=1):
        header = f"### AI Hedge Fund Report — {ticker} (Part {i}/{len(chunks)})\n"
        post_comment(repo, issue_number, header + "\n" + c)


if __name__ == "__main__":
    main()
