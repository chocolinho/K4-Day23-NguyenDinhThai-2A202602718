"""Normalize source records inside the sandbox before citation finalization.

An arXiv result must point at its canonical abstract page. Other URLs cannot
have come from the arXiv tool, so classify them as web results. The retrieval
family of a genuine arXiv URL found through web search remains ``web``.
"""

import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_SOURCES = Path("/tmp/work/research/sources.json")
ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")


def normalize(sources):
    for source in sources:
        # ACL Anthology often supplies only a publication month. A generated
        # January 1 date for its July 2026 proceedings is false precision.
        url = str(source.get("url", ""))
        parsed = urlparse(url)
        if (parsed.hostname == "aclanthology.org" and
                re.match(r"/2026\.(?:findings-acl|acl-long)\.", parsed.path) and
                source.get("date") == "2026-01-01"):
            source["date"] = "unknown"
        if (parsed.hostname == "openaccess.thecvf.com" and
                "/content/CVPR" in parsed.path and
                re.fullmatch(r"\d{4}-\d{2}-01", str(source.get("date", "")))):
            source["date"] = "unknown"
        if source.get("source") != "arxiv":
            continue
        candidate = parsed.path.rstrip("/").split("/")[-1]
        if parsed.hostname in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"} and ARXIV_ID.fullmatch(candidate):
            paper_id = re.sub(r"v\d+$", "", candidate)
            source["id"] = paper_id
            source["url"] = f"https://arxiv.org/abs/{paper_id}"
        else:
            source["source"] = "web"
    return sources


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCES
    sources = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sources, list) or not all(isinstance(item, dict) for item in sources):
        raise ValueError("sources.json must be a list of objects")
    normalized = normalize(sources)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"NORMALIZED: {len(normalized)} sources")


if __name__ == "__main__":
    main()
