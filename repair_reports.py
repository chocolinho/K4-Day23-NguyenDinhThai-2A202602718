"""Recheck existing reports in a fresh sandbox and download the validated files.

Usage: python repair_reports.py <report-slug> [<report-slug> ...]
"""

import json
import sys

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH
from research import (FINALIZER_SOURCE, NORMALIZER_PATH, NORMALIZER_SOURCE,
                      REPORTS, VALIDATOR_SOURCE)
from sandbox import download, open_sandbox, upload


def repair(slug):
    report_path = REPORTS / f"{slug}.md"
    sources_path = REPORTS / f"{slug}.sources.json"
    meta_path = REPORTS / f"{slug}.meta.json"
    original = {REPORT_PATH: report_path.read_bytes(), SOURCES_PATH: sources_path.read_bytes()}
    with open_sandbox() as backend:
        upload(backend, {**original, VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                         FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                         NORMALIZER_PATH: NORMALIZER_SOURCE.read_bytes()})
        for command in (f"sed -i 's/[[:blank:]]*$//' {REPORT_PATH}",
                        f"python3 {NORMALIZER_PATH}", f"python3 {FINALIZER_PATH}",
                        f"python3 {VALIDATOR_PATH}"):
            result = backend.execute(command)
            if result.exit_code:
                raise RuntimeError(f"{slug}: {command}: {result.output}")
            print(f"{slug}: {result.output.strip()}")
        files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, source_bytes = files[REPORT_PATH], files[SOURCES_PATH]
    if not report_bytes or not source_bytes:
        raise RuntimeError(f"{slug}: failed to download validated files")
    sources = json.loads(source_bytes)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["n_sources"] = len(sources)
    metadata["source_families"] = sorted({source["source"] for source in sources})
    report_path.write_bytes(report_bytes)
    sources_path.write_bytes(source_bytes)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python repair_reports.py <report-slug> [<report-slug> ...]")
    for report_slug in sys.argv[1:]:
        repair(report_slug)
