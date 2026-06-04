#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def pretty(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", help="PDF path to upload")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--async-mode", action="store_true")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    print("HEALTH")
    pretty(requests.get(f"{base}/health", timeout=30).json())

    if not args.pdf:
        print("\nSAMPLE")
        pretty(requests.get(f"{base}/test-sample", timeout=30).json())
        return

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    endpoint = "/validate-batch-async" if args.async_mode else "/validate-batch"
    params = {} if args.async_mode else {"mode": "sync"}
    with pdf.open("rb") as f:
        response = requests.post(
            f"{base}{endpoint}",
            params=params,
            files=[("files", (pdf.name, f, "application/pdf"))],
            timeout=60 if args.async_mode else 900,
        )
    print(f"\nUPLOAD STATUS: {response.status_code}")
    data = response.json()
    pretty(data)

    if data.get("status") == "processing":
        result_url = data.get("result_url") or data.get("poll_url")
        if not result_url:
            return
        for i in range(180):
            time.sleep(5)
            status_data = requests.get(f"{base}{data['poll_url']}", timeout=30).json()
            print(f"poll {i + 1}: {status_data.get('status')}")
            if status_data.get("status") == "done":
                pretty(requests.get(f"{base}{result_url}", timeout=60).json())
                return
            if status_data.get("status") == "failed":
                pretty(status_data)
                return


if __name__ == "__main__":
    main()
