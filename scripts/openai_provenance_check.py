#!/usr/bin/env python3
"""One-shot OpenAI Content Provenance oracle check per file.

Usage: python3 scripts/openai_provenance_check.py FILE [FILE...]

Prints ``basename<TAB>synthid_outcome`` per file (DETECTED / NOT_DETECTED).
Reads OPENAI_API_KEY from the environment or the main checkout's .env.

This is the documented programmatic twin of openai.com/verify and reports the
SynthID and C2PA outcomes separately, which the web page collapses.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.request
import uuid

API_URL = "https://api.openai.com/v1/content_provenance_checks"


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_file = pathlib.Path.home() / "Documents/GitHub/remove-ai-watermarks/.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("no OPENAI_API_KEY in env and no main-checkout .env")


def check(path: pathlib.Path) -> str:
    boundary = uuid.uuid4().hex
    payload = path.read_bytes()
    body = b"".join(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            "Content-Type: image/png\r\n\r\n".encode(),
            payload,
            f"\r\n--{boundary}--\r\n".encode(),
        )
    )
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.load(response)
    for entry in data.get("results", []):
        if entry.get("type") == "synthid":
            return str(entry.get("outcome", "?")).upper()
    return "NO_SYNTHID_ENTRY"


def main() -> None:
    for name in sys.argv[1:]:
        path = pathlib.Path(name)
        try:
            outcome = check(path)
        except Exception as exc:
            outcome = f"ERROR:{exc}"
        print(f"{path.name}\t{outcome}", flush=True)


if __name__ == "__main__":
    main()
