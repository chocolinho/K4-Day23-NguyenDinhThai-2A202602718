"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import random
import re
import threading
import time  # noqa: F401
from urllib.parse import quote
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)

import httpx  # noqa: F401
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_arxiv_lock = threading.Lock()
_last_arxiv_call = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    for attempt in range(max(1, attempts)):
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in (429, 500, 502, 503, 504):
                raise
            error = RetryableError(str(exc), exc.response.headers.get("Retry-After"))
        except httpx.TransportError as exc:
            error = RetryableError(str(exc))
        except RetryableError as exc:
            error = exc
        if attempt == max(1, attempts) - 1:
            raise error
        try:
            retry_after = float(error.retry_after) if error.retry_after is not None else None
        except (TypeError, ValueError):
            retry_after = None
        delay = min(cap, max(0.0, retry_after)) if retry_after is not None else min(
            cap, base * 2 ** attempt + random.uniform(0, base)
        )
        time.sleep(delay)


def _error(exc):
    """Never expose a configured Exa key to the agent through an error string."""
    message = f"{type(exc).__name__}: {exc}"
    key = os.getenv("EXA_API_KEY", "")
    if key:
        message = message.replace(key, "[REDACTED]").replace(quote(key, safe=""), "[REDACTED]")
    message = re.sub(r"exaApiKey=[^&\s]+", "exaApiKey=[REDACTED]", message, flags=re.I)
    return "ERROR: " + message


def _get(url, params, *, attempts=5, cap=30.0):
    def request():
        response = httpx.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response
    return with_retry(request, attempts=attempts, cap=cap)


def _clean(value, limit=600):
    return " ".join(str(value or "").split())[:limit]


def _hf_record(item, prefer_ai=False):
    paper = item.get("paper", item) if isinstance(item, dict) else {}
    if not isinstance(paper, dict) or not paper.get("id"):
        return None
    identifier = str(paper["id"])
    summary = (paper.get("ai_summary") or item.get("ai_summary")) if prefer_ai else None
    summary = summary or paper.get("summary") or item.get("summary")
    return {
        "id": identifier,
        "url": f"https://huggingface.co/papers/{identifier}",
        "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
        "title": _clean(paper.get("title") or item.get("title")),
        "summary": _clean(summary),
        "upvotes": paper.get("upvotes") or item.get("upvotes") or 0,
        "github": paper.get("githubRepo") or item.get("githubRepo") or "",
        "stars": paper.get("githubStars") or item.get("githubStars") or 0,
    }


def _exa_call(name, arguments):
    key = os.getenv("EXA_API_KEY", "").strip()
    params = {"exaApiKey": key} if key else None
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": arguments}}

    def request():
        response = httpx.post(EXA_URL, params=params, json=payload, timeout=45,
                              headers={"Accept": "application/json, text/event-stream"})
        response.raise_for_status()
        data = None
        for line in response.text.splitlines():
            if line.startswith("data:"):
                try:
                    event = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if isinstance(event, dict) and ("result" in event or "error" in event):
                    data = event
        if data is None:
            data = response.json()
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        result = data.get("result") or {}
        meta = result.get("_meta") or {}
        content = result.get("content") or []
        text = "\n".join(str(part.get("text", "")) for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
        # Exa can send a rate-limit notice with HTTP 200; detect both its flag and message.
        limited = any(re.search(r"rate.?limit|too.?many", str(k), re.I) and v not in (False, None, 0, "", "false")
                      for k, v in meta.items()) if isinstance(meta, dict) else False
        if limited or re.search(
            r"rate[ -]?limit|too many requests|try again.*(?:second|minute)", text, re.I
        ):
            raise RetryableError("Exa rate limit")
        if result.get("isError"):
            raise RuntimeError(text or "Exa tool error")
        return text

    return with_retry(request, attempts=7, cap=60)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    try:
        terms = re.findall(r"[^\W_]+(?:-[^\W_]+)*", query, re.UNICODE)
        terms = [term for term in terms if term.upper() not in {"AND", "OR", "NOT"}]
        if not terms:
            return "NO RESULTS"

        def request():
            global _last_arxiv_call
            with _arxiv_lock:
                pause = 3.0 - (time.monotonic() - _last_arxiv_call)
                if pause > 0:
                    time.sleep(pause)
                _last_arxiv_call = time.monotonic()
                return _get(ARXIV_URL, {
                    "search_query": " AND ".join(f"all:{term}" for term in terms),
                    "sortBy": "submittedDate", "sortOrder": "descending",
                    "max_results": max(1, min(int(max_results), 30)), "start": 0,
                }, attempts=1)

        response = with_retry(request, attempts=7, cap=60)
        root = xml.etree.ElementTree.fromstring(response.text)
        namespace = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("a:entry", namespace):
            raw_id = entry.findtext("a:id", default="", namespaces=namespace)
            identifier = re.sub(r"v\d+$", "", raw_id.rsplit("/", 1)[-1])
            if not identifier:
                continue
            records.append({
                "id": identifier, "url": f"https://arxiv.org/abs/{identifier}",
                "published": entry.findtext("a:published", default="", namespaces=namespace)[:10],
                "title": _clean(entry.findtext("a:title", default="", namespaces=namespace)),
                "summary": _clean(entry.findtext("a:summary", default="", namespaces=namespace)),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    try:
        params = {"limit": max(1, min(int(limit), 100))}
        if date:
            params["date"] = date
        items = _get(HF_DAILY_URL, params).json()
        records = [record for item in items if (record := _hf_record(item)) is not None]
        if keyword:
            words = keyword.casefold().split()
            records = [r for r in records if all(w in (r["title"] + " " + r["summary"]).casefold()
                                                 for w in words)]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    try:
        if not query.strip():
            return "NO RESULTS"
        items = _get(HF_SEARCH_URL, {"q": query, "limit": max(1, min(int(limit), 50))}).json()
        records = [record for item in items if (record := _hf_record(item, prefer_ai=True)) is not None]
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    try:
        if not query.strip():
            return "NO RESULTS"
        result = _exa_call("web_search_exa", {"query": query,
            "objective": objective or f"Find reliable sources about {query}",
            "numResults": max(1, min(int(num_results), 20))})
        return result or "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    try:
        if not url.strip():
            return "NO RESULTS"
        result = _exa_call("web_fetch_exa", {"urls": [url]})
        return result[:12000] if result else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
