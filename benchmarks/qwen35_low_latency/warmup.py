# SPDX-License-Identifier: Apache-2.0
"""Publish a deployment readiness marker only after a short inference succeeds."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from benchmark import controlled_prompt, post, request_one


def warmup(base_url, model, timeout):
    started = time.monotonic()
    while True:
        try:
            with urllib.request.urlopen(base_url + "/v1/models", timeout=5) as response:
                models = json.load(response)
            if not any(entry["id"] == model for entry in models["data"]):
                raise RuntimeError(f"Requested model {model!r} is not served")
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() - started >= timeout:
                raise TimeoutError("Server did not start before the warmup deadline") from None
            time.sleep(1)
    with post(
        base_url + "/tokenize", dict(model=model, prompt="Correct this query: green apples.", add_special_tokens=False)
    ) as response:
        seed = json.load(response)["tokens"]
    result = request_one(base_url, model, controlled_prompt(seed, 64, 0), 4)
    return dict(ready=True, model=model, wait_and_warmup_s=time.monotonic() - started, warmup=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    if args.ready_file.exists():
        parser.error("Remove the previous deployment's readiness marker before starting a new deployment")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    report = warmup(args.base_url, args.model, args.timeout)
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.ready_file.with_suffix(args.ready_file.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.ready_file)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
