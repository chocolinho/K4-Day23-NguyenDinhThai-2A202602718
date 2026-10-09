"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    by_number = {}
    urls = set()
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            problems.append(f"source {index} is not an object")
            continue
        number, url = source.get("n"), source.get("url")
        if type(number) is not int:
            problems.append(f"source {index} has a non-integer n")
        elif number in by_number:
            problems.append(f"source [{number}] occurs more than once")
        else:
            by_number[number] = source
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source {index} has an invalid url")
        elif url in urls:
            problems.append(f"source {index} duplicates url {url}")
        else:
            urls.add(url)

    heading = re.search(r"(?m)^##[ \t]+References[ \t]*$", report_text)
    if heading is None:
        problems.append("missing ## References heading")
        body, references = report_text, ""
    else:
        body, references = report_text[:heading.start()], report_text[heading.end():]

    # Code and Markdown links can contain [n] without citing a source.
    body = re.sub(r"```.*?```|`[^`\n]*`", "", body, flags=re.S)
    body = re.sub(r"\[\d+\]\([^)]*\)", "", body)
    cited = set()
    for match in re.finditer(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()", body):
        for part in re.split(r"\s*,\s*", match.group(1)):
            span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
            if span:
                start, end = map(int, span.groups())
                if end < start or end - start > 200:
                    problems.append(f"invalid citation range [{part}]")
                else:
                    cited.update(range(start, end + 1))
            else:
                cited.add(int(part))
    for number in sorted(cited - by_number.keys()):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(by_number.keys() - cited):
        problems.append(f"source [{number}] never cited")

    ref_counts = {}
    for line in references.splitlines():
        match = re.match(r"^\[(\d+)\]\s+", line)
        if not match:
            continue
        number = int(match.group(1))
        ref_counts[number] = ref_counts.get(number, 0) + 1
        found_urls = re.findall(r"https?://[^\s<>]+", line)
        if len(found_urls) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL")
        elif number in by_number and found_urls[0] != by_number[number].get("url"):
            problems.append(f"reference [{number}] URL differs from sources.json")
        if number not in by_number:
            problems.append(f"reference [{number}] has no source")
    for number in sorted(by_number):
        if ref_counts.get(number, 0) != 1:
            problems.append(f"source [{number}] needs exactly one reference line")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
