"""Day 2: send many concurrent requests and measure what each user experiences.

Usage:
    # Closed loop: 8 users, each sends its next request when the last one finishes
    python loadtest/load.py --mode closed --concurrency 8 --requests 40

    # Open loop: requests arrive at random times, 2 per second on average
    python loadtest/load.py --mode open --rate 2 --requests 40
"""
import argparse
import asyncio
import csv
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from single_request import DEFAULT_MODEL, DEFAULT_URL, build_payload, parse_sse_line

RESULTS_DIR = Path("results")
WORDS = ("cache memory token batch queue server model tensor kernel thread "
         "signal vector stream packet socket buffer").split()


@dataclass
class Result:
    t_scheduled: float
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


def make_prompt(rng: random.Random, n_words: int) -> str:
    # Random words stop vLLM's prefix cache from skipping prefill on repeated prompts.
    words = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return f"Write a long story using these words: {words}"


async def one_request(client: httpx.AsyncClient, url: str, payload: dict,
                      t_scheduled: float | None = None) -> Result:
    """Send one streaming request and time it.

    Latency counts from t_scheduled, the moment the request was meant to go out,
    so time spent waiting on the client side still counts against the server.
    """
    if t_scheduled is None:
        t_scheduled = time.perf_counter()
    t_first = None
    usage = None
    try:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                event = parse_sse_line(line)
                if event is None:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event["choices"]:
                    if choice["delta"].get("content") and t_first is None:
                        t_first = time.perf_counter()
    except httpx.HTTPError as e:
        # httpx messages span several lines; keep one error on one CSV line.
        return Result(t_scheduled, error=f"{type(e).__name__}: {' '.join(str(e).split())}")

    t_end = time.perf_counter()
    if t_first is None or usage is None:
        return Result(t_scheduled, error="stream ended without text or usage")
    output_tokens = usage["completion_tokens"]
    return Result(
        t_scheduled=t_scheduled,
        ttft_ms=(t_first - t_scheduled) * 1000,
        tpot_ms=(t_end - t_first) * 1000 / max(output_tokens - 1, 1),
        e2e_ms=(t_end - t_scheduled) * 1000,
        prompt_tokens=usage["prompt_tokens"],
        output_tokens=output_tokens,
    )


async def run_closed(client: httpx.AsyncClient, url: str, make_payload,
                     concurrency: int, n_requests: int) -> list[Result]:
    """Closed loop: `concurrency` users, n_requests in total."""
    # TODO(you):
    #   1. Write an inner `async def worker()`. It loops: while fewer than n_requests
    #      have been started, count one more as started, then
    #      `result = await one_request(client, url, make_payload())` and save the result.
    #   2. Run `concurrency` workers together with asyncio.gather(...).
    #   3. Return all the results.
    # Think: why is a plain shared counter safe here without a lock? (Where can
    # asyncio switch between tasks?)

    results = []
    started =0

    async def worker():
        nonlocal started
        while started < n_requests:
            started +=1
            result = await one_request(client, url, make_payload())
            results.append(result)

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return results


async def run_open(client: httpx.AsyncClient, url: str, make_payload,
                   rate: float, n_requests: int, rng: random.Random) -> list[Result]:
    """Open loop: Poisson arrivals at `rate` requests/s, n_requests in total."""
    # TODO(you):
    #   1. Start a clock: next_arrival = time.perf_counter().
    #   2. For each request:
    #        - sleep until next_arrival (await asyncio.sleep, never below zero)
    #        - asyncio.create_task(one_request(client, url, make_payload(), next_arrival))
    #          and keep the task. Do NOT await it here. That is the whole point of open loop.
    #        - next_arrival += rng.expovariate(rate)
    #   3. await asyncio.gather(*tasks) and return the results.
    # Think: why add to next_arrival rather than sleeping a random gap each time?
 
    tasks = []
    next_arrival = time.perf_counter()

    for _ in range(n_requests):
        await asyncio.sleep(max(0, next_arrival - time.perf_counter()))
        tasks.append(asyncio.create_task(one_request(client, url, make_payload(), next_arrival)))
        next_arrival += rng.expovariate(rate)

    return await asyncio.gather(*tasks)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least p% of values at or below it."""
    # TODO(you): sort the values, then return sorted_values[rank - 1]
    # where rank = ceil(p / 100 * n). Check by hand: [10, 20, 30, 40], p=50 should give 20.

    sorted_values = sorted(values)
    rank = math.ceil(p / 100 * len(sorted_values))
    return sorted_values[rank - 1]


def summarize(results: list[Result], wall_s: float) -> dict:
    """Turn per-request results into one summary row."""
    ok = [r for r in results if r.error is None]
    # TODO(you): using only `ok`, return a dict with these keys:
    #   n_ok, n_err
    #   req_s          completed requests / wall_s
    #   output_tok_s   all output tokens / wall_s  (the server's total throughput)
    #   ttft_p50, ttft_p95, ttft_p99
    #   tpot_p50, tpot_p95   (what each user feels)
    #   e2e_p50, e2e_p95, e2e_mean
    return {
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "req_s": len(ok)/wall_s,
        "output_tok_s": sum([r.output_tokens for r in ok]) /wall_s,
        "ttft_p50": percentile([r.ttft_ms for r in ok], 50),
        "ttft_p95": percentile([r.ttft_ms for r in ok], 95),
        "ttft_p99": percentile([r.ttft_ms for r in ok], 99),
        "tpot_p50": percentile([r.tpot_ms for r in ok], 50),
        "tpot_p95": percentile([r.tpot_ms for r in ok], 95),
        "e2e_p50": percentile([r.e2e_ms for r in ok], 50),
        "e2e_p95": percentile([r.e2e_ms for r in ok], 95),
        "e2e_mean": sum([r.e2e_ms for r in ok])/ len(ok)
    }


def save(label: str, results: list[Result], summary: dict) -> None:
    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / f"{label}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)

    summary_path = RESULTS_DIR / "summary.csv"
    row = {"label": label, **summary}
    is_new = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def show(summary: dict, wall_s: float) -> None:
    print(f"\n{summary['n_ok']} ok, {summary['n_err']} errors in {wall_s:.1f} s")
    print(f"throughput: {summary['req_s']:.2f} req/s, {summary['output_tok_s']:.1f} output tok/s")
    print(f"TTFT ms: p50 {summary['ttft_p50']:.0f}  p95 {summary['ttft_p95']:.0f}  p99 {summary['ttft_p99']:.0f}")
    print(f"TPOT ms: p50 {summary['tpot_p50']:.1f}  p95 {summary['tpot_p95']:.1f}")
    print(f"E2E  ms: p50 {summary['e2e_p50']:.0f}  p95 {summary['e2e_p95']:.0f}  mean {summary['e2e_mean']:.0f}")
    # Little's Law: average requests in the system = arrival rate × average time in the system.
    print(f"Little's Law: L = {summary['req_s']:.2f} req/s × {summary['e2e_mean'] / 1000:.2f} s "
          f"= {summary['req_s'] * summary['e2e_mean'] / 1000:.1f} requests in the system")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mode", choices=["closed", "open"], required=True)
    ap.add_argument("--concurrency", type=int, default=8, help="closed loop: number of users")
    ap.add_argument("--rate", type=float, default=2.0, help="open loop: requests per second")
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--prompt-words", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=None, help="default: a new seed for every run")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else time.time_ns()
    rng = random.Random(seed)

    def make_payload() -> dict:
        return build_payload(args.model, make_prompt(rng, args.prompt_words), args.max_tokens)

    load = f"c{args.concurrency}" if args.mode == "closed" else f"r{args.rate:g}"
    label = f"{args.mode}_{load}_n{args.requests}_t{args.max_tokens}_{int(time.time())}"

    # No connection limit: httpx's default cap would queue requests inside the client.
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=600, limits=limits) as client:
        # Day 1: the first request after a server start is slow. Keep it out of the results.
        warm = await one_request(client, args.url, make_payload())
        if warm.error:
            raise SystemExit(f"Warmup failed, is the server running? {warm.error}")

        t0 = time.perf_counter()
        if args.mode == "closed":
            results = await run_closed(client, args.url, make_payload, args.concurrency, args.requests)
        else:
            results = await run_open(client, args.url, make_payload, args.rate, args.requests, rng)
        wall_s = time.perf_counter() - t0

    errors = [r.error for r in results if r.error]
    if errors:
        print(f"{len(errors)} errors, first: {errors[0]}")
    if len(errors) == len(results):
        raise SystemExit("Every request failed.")

    summary = summarize(results, wall_s)
    show(summary, wall_s)
    save(label, results, summary)
    print(f"saved results/raw/{label}.csv")


if __name__ == "__main__":
    asyncio.run(main())
