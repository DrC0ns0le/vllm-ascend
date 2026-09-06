# SPDX-License-Identifier: Apache-2.0
"""Streaming per-request latency and repeatability harness (stdlib only)."""

import argparse
import concurrent.futures
import json
import math
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BOUNDARIES = (
    1,
    7,
    31,
    32,
    33,
    63,
    64,
    65,
    95,
    96,
    97,
    127,
    128,
    129,
    159,
    160,
    191,
    192,
    193,
    223,
    224,
    255,
    256,
    257,
    383,
    384,
    385,
    511,
    512,
    513,
    767,
    768,
    769,
    1023,
    1024,
    1025,
    1535,
    1536,
)
CONCURRENT = (
    (64, 64),
    (90, 90),
    (100, 100),
    (128, 128),
    (60, 130),
    (50, 180),
    (50, 50, 50),
    (40, 70, 120),
    (32, 32, 32, 32),
    (64, 64, 64, 64),
    (100, 100, 100, 100),
    (50, 75, 125, 180),
    (192,),
    (96, 96),
    (64, 64, 64),
    (256,),
    (128, 128),
    (64, 64, 64, 64),
    (64, 192),
    (100, 156),
    (80, 80, 96),
)
PRIMARY = ((64, 1), (68, 4), (128, 1), (128, 4), (128, 8), (133, 4), (192, 1), (256, 1), (512, 1))


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    lower = math.floor(position)
    return values[lower] + (values[math.ceil(position)] - values[lower]) * (position - lower)


def post(url, payload):
    request = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=180)


def request_one(base_url, model, prompt, output, release=None, wait=None, fixed_output=True):
    if wait is not None and not wait.wait(timeout=180):
        raise RuntimeError("Earlier request never produced its first token")
    payload = dict(
        model=model,
        prompt=prompt,
        max_tokens=output,
        min_tokens=output if fixed_output else 0,
        temperature=0,
        seed=1024,
        ignore_eos=fixed_output,
        stream=True,
        logprobs=1,
        stream_options={"include_usage": True},
    )
    started = time.perf_counter()
    first = last = None
    text, tokens, usage = "", [], None
    with post(base_url + "/v1/completions", payload) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                part = choice.get("text", "")
                emitted = (choice.get("logprobs") or {}).get("tokens", [])
                if emitted or part:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                        if release is not None:
                            release.set()
                    last = now
                    text += part
                    tokens.extend(emitted)
    ended = time.perf_counter()
    if first is None or usage is None:
        raise RuntimeError(f"Incomplete generation: expected O{output}, usage={usage}")
    actual_output = usage["completion_tokens"]
    if actual_output <= 0 or actual_output > output or (fixed_output and actual_output != output):
        raise RuntimeError(f"Incorrect output length: requested O{output}, usage={usage}")
    if usage["prompt_tokens"] != len(prompt):
        raise RuntimeError(f"Prompt token length changed: {len(prompt)} -> {usage['prompt_tokens']}")
    return dict(
        status="ok",
        http_status=200,
        prompt_tokens=len(prompt),
        output_tokens=actual_output,
        output_budget=output,
        prompt=prompt,
        text=text,
        tokens=tokens,
        started=started,
        ended=ended,
        ttft_ms=(first - started) * 1000,
        e2e_ms=(ended - started) * 1000,
        tpot_ms=(last - first) * 1000 / (actual_output - 1) if actual_output > 1 else None,
    )


def request_record(base_url, model, prompt, output, *args, **kwargs):
    """Retain failed attempts so a bad c64 run cannot look like a smaller pass."""
    started = time.perf_counter()
    try:
        return request_one(base_url, model, prompt, output, *args, **kwargs)
    except Exception as error:
        ended = time.perf_counter()
        release = kwargs.get("release", args[0] if args else None)
        if release is not None:
            # Let dependent test clients finish after a failure; the run is
            # still marked failed and cannot establish mixed-batch correctness.
            release.set()
        return dict(
            status="error",
            http_status=getattr(error, "code", None),
            error=str(error),
            prompt=prompt,
            prompt_tokens=len(prompt),
            output_budget=output,
            output_tokens=None,
            text="",
            tokens=[],
            started=started,
            ended=ended,
            ttft_ms=None,
            tpot_ms=None,
            e2e_ms=(ended - started) * 1000,
        )


def summary(rows, elapsed):
    successful = [row for row in rows if row.get("status", "ok") == "ok"]
    result = dict(
        requests=len(rows),
        successful_requests=len(successful),
        errors=len(rows) - len(successful),
        error_rate=(len(rows) - len(successful)) / len(rows) if rows else None,
        requests_per_second=len(successful) / elapsed if elapsed > 0 else None,
    )
    for name, percentiles in (("ttft_ms", (50, 95, 99)), ("e2e_ms", (50, 95, 99)), ("tpot_ms", (50, 95))):
        for p in percentiles:
            result[f"{name}_p{p}"] = percentile([row[name] for row in successful if row[name] is not None], p)
    return result


def production_workload(distribution, count, seed, live=False):
    rng = random.Random(seed)
    ragged = (64, 1024, 65, 900, 128, 700, 192, 600)
    result = []
    for index in range(count):
        if distribution == "ragged":
            length = ragged[index % len(ragged)]
        else:
            length = rng.randint(64, 256 if distribution == "short-heavy" else 1024)
        result.append((length, rng.randint(3, 10) if live else 4))
    return result


def controlled_prompt(seed_tokens, length, index):
    if not seed_tokens:
        raise ValueError("Tokenizer returned an empty seed")
    # Rotating a real tokenized seed makes equal-length requests differ while
    # retaining exact token counts. Server usage must independently confirm it.
    offset = index % len(seed_tokens)
    seed = seed_tokens[offset:] + seed_tokens[:offset]
    return (seed * math.ceil(length / len(seed)))[:length]


def continuous_requests(cases, concurrency, request, record):
    """Bound client HTTP concurrency and replace completions immediately.

    These remain individual API requests. vLLM alone packs model-step batches.
    """
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    iterator = iter(enumerate(cases))
    rows = []
    peak = 0
    active = 0
    lock = threading.Lock()

    def run(index, case):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return request(index, case)
        finally:
            with lock:
                active -= 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {}

        def submit():
            entry = next(iterator, None)
            if entry is not None:
                index, case = entry
                pending[pool.submit(run, index, case)] = index

        for _ in range(concurrency):
            submit()
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                row = future.result() | dict(request_id=index)
                record(row)
                rows.append(row)
                submit()
    return rows, peak


def collect_metrics(base_url, output, stop):
    # Preserve the runtime's actual metric names/labels. Missing metrics are
    # explicit errors, not invented zeros for cache, memory or preemption.
    with output.open("w") as file:
        while not stop.is_set():
            sample = dict(timestamp=time.time())
            try:
                with urllib.request.urlopen(base_url + "/metrics", timeout=2) as response:
                    sample["prometheus"] = response.read().decode()
            except Exception as error:
                sample["error"] = str(error)
            file.write(json.dumps(sample) + "\n")
            file.flush()
            stop.wait(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--variant", choices=list("ABCD"), required=True)
    parser.add_argument(
        "--suite",
        choices=["primary", "boundaries", "concurrent", "mixed", "stress", "corpus", "production", "live"],
        default="primary",
    )
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 4, 8, 16, 32, 64], default=1)
    parser.add_argument("--requests", type=int, help="Total production/live requests; defaults to 512/1000")
    parser.add_argument("--distribution", choices=["short-heavy", "wide", "ragged"], default="wide")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=1536)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, help='JSONL: {"prompt": "..." or token IDs, "max_tokens": 8}')
    args = parser.parse_args()
    if args.suite == "corpus" and args.corpus is None:
        parser.error("--suite corpus requires --corpus")
    if args.repetitions <= 0 or (args.requests is not None and args.requests <= 0):
        parser.error("request and repetition counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(args.base_url + "/v1/models") as response:
        json.load(response)
    contents = [
        "Correct the spelling in this query: green apples from the market. ",
        "Return only a corrected query: book a train to Cambridge tomorrow. ",
        "Correct this search: how to bake sourdough bread at home. ",
        "Return a corrected query: rainfall in northern Scotland in spring. ",
    ]
    seeds = []
    for content in contents:
        with post(
            args.base_url + "/tokenize", dict(model=args.model, prompt=content, add_special_tokens=False)
        ) as response:
            seeds.append(json.load(response)["tokens"])

    def prompt(length, index):
        seed = seeds[index % len(seeds)]
        return controlled_prompt(seed, length, index // len(seeds))

    # Warm all relevant paths before timing; compilation time stays in startup logs.
    for index in range(4):
        request_one(args.base_url, args.model, prompt(128, index), 8)
    if args.suite in ("production", "live"):
        count = args.requests or (1000 if args.suite == "live" else 512)
        cases = production_workload(args.distribution, count, args.seed, args.suite == "live")
        if any(p + o > args.max_model_len for p, o in cases):
            parser.error("Production workload exceeds max-model-len")
        stop = threading.Event()
        sampler = threading.Thread(target=collect_metrics, args=(args.base_url, args.output / "metrics.jsonl", stop))
        sampler.start()
        try:
            with (args.output / "requests.jsonl").open("w") as raw:

                def record(row):
                    raw.write(json.dumps(row) + "\n")
                    raw.flush()

                started = time.perf_counter()
                rows, peak = continuous_requests(
                    cases,
                    args.concurrency,
                    lambda index, case: request_record(args.base_url, args.model, prompt(case[0], index), case[1])
                    | dict(variant=args.variant, concurrency=args.concurrency),
                    record,
                )
                report = summary(rows, time.perf_counter() - started) | dict(
                    variant=args.variant,
                    suite=args.suite,
                    distribution=args.distribution,
                    seed=args.seed,
                    concurrency=args.concurrency,
                    max_client_in_flight=peak,
                    mixed_scheduler_evidence="required",
                    server_capacity_evidence="metrics.jsonl and runtime logs",
                )
                (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report), flush=True)
        finally:
            stop.set()
            sampler.join()
        if report["errors"]:
            raise SystemExit("Traffic run had errors; inspect requests.jsonl")
        return
    corpus_batches = []
    if args.suite == "corpus":
        cases = []
        for line in args.corpus.read_text().splitlines():
            case = json.loads(line)
            token_ids = case["prompt"]
            if isinstance(token_ids, str):
                with post(args.base_url + "/tokenize", dict(model=args.model, prompt=token_ids)) as response:
                    token_ids = json.load(response)["tokens"]
            cases.append((token_ids, case["max_tokens"]))
        corpus_batches = [cases[i : i + args.concurrency] for i in range(0, len(cases), args.concurrency)]
        groups = [
            (tuple(len(tokens) for tokens, _ in batch), tuple(output for _, output in batch))
            for batch in corpus_batches
        ]
    elif args.suite == "primary":
        groups = [(tuple([p] * args.concurrency), tuple([o] * args.concurrency)) for p, o in PRIMARY]
    elif args.suite == "boundaries":
        groups = [((p,), (1,)) for p in BOUNDARIES]
    elif args.suite == "concurrent":
        groups = [(p, (4,) * len(p)) for p in CONCURRENT]
    elif args.suite == "mixed":
        groups = [((128, 80), (32, 4)), ((128, 64, 80), (32, 24, 4)), ((128, 64, 96, 80), (32, 24, 16, 4))]
    else:
        rng = random.Random(1024)
        groups = [
            (
                tuple(rng.choice((7, 63, 129, 193, 385, 769)) for _ in range(4)),
                tuple(rng.choice((1, 4, 8)) for _ in range(4)),
            )
            for _ in range(250)
        ]
        args.repetitions = max(1, args.repetitions)
    summaries = []
    with (args.output / "requests.jsonl").open("w") as raw:
        for group_index, (lengths, outputs) in enumerate(groups):
            if any(p + o > args.max_model_len for p, o in zip(lengths, outputs)):
                summaries.append(dict(lengths=lengths, outputs=outputs, status="outside_context_budget"))
                continue
            rows = []
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(lengths)) as pool:
                for repeat in range(args.repetitions):
                    releases = [threading.Event() for _ in lengths]
                    futures = [
                        pool.submit(
                            request_record,
                            args.base_url,
                            args.model,
                            corpus_batches[group_index][i][0] if corpus_batches else prompt(p, i + repeat),
                            o,
                            releases[i],
                            releases[i - 1] if args.suite == "mixed" and i else None,
                            args.suite != "corpus",
                        )
                        for i, (p, o) in enumerate(zip(lengths, outputs))
                    ]
                    for future in futures:
                        row = future.result() | dict(variant=args.variant, concurrency=len(lengths), repeat=repeat)
                        rows.append(row)
                        raw.write(json.dumps(row) + "\n")
                    raw.flush()
            report = dict(
                variant=args.variant,
                concurrency=len(lengths),
                lengths=lengths,
                outputs=outputs,
                mixed_scheduler_evidence="required" if args.suite == "mixed" else None,
            )
            report.update(summary(rows, time.perf_counter() - started))
            summaries.append(report)
            print(json.dumps(report), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    if any(report.get("errors", 0) for report in summaries):
        raise SystemExit("Traffic run had errors; inspect requests.jsonl")


if __name__ == "__main__":
    main()
