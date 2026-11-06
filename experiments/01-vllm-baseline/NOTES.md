# 01: vLLM baseline

**Question:** how much faster is a real serving engine than a naive server, and why?

## Day 1: First request through vLLM

Setup: Qwen2.5-1.5B-Instruct, bf16, `--max-model-len 4096 --gpu-memory-utilization 0.75`, RTX 4050 Laptop 6 GB.

### Predictions (write these before running)

| # | Question | My prediction | Measured | Why the difference? |
|---|---|---|---|---|
| P1 | GiB of VRAM for model weights | 2.9 GiB (1.54B params × 2 bytes) | 2.98 GiB | Close. The real parameter count is slightly above 1.54B, plus small buffers (e.g. RoPE tables). |
| P2 | KV cache bytes per token (from config.json) | 28 KiB (2 × 28 layers × 2 KV heads × 128 × 2 bytes) | 0.48 GiB ÷ 17,824 tokens ≈ 28 KiB | Matches. Only attention K/V is cached; GQA (2 KV heads for 12 query heads) makes it 6× smaller than full multi-head attention. |
| P3 | Tokens that fit in the KV cache | ~1.2 GiB → ~45k tokens → ~11 requests of 4,096 tokens | 0.48 GiB → 17,824 tokens → 4.35 requests | Underestimated overhead: the log reports non-torch memory on top of the weights (3.24 GiB together), activations 0.79 GiB not 0.4, and CUDA graphs 0.19 GiB, which I forgot. These plus the KV cache total 4.70 GiB, more than the 4.5 GiB budget, so the log's figures do not reconcile by simple subtraction; treat the breakdown as approximate. |
| P4 | Batch-1 decode tok/s (ceiling ~62) | ~37 tok/s (60% of ceiling) | 57–59 tok/s; TPOT 16.9–17.5 ms | Real ceiling uses the real weight size: 3.2 GB ÷ 192 GB/s = 16.7 ms → ~60 tok/s, so vLLM is at ~98% of memory bandwidth. CUDA graphs remove the kernel-launch overhead that limited batch-1 decode in Month 1. |
| P5 | TTFT for a short prompt (ms) | 50 ms | 26 ms (39-token prompt) | A short prefill still reads all weights once, so it costs about one decode step (~17 ms) plus ~9 ms of HTTP, tokenization and scheduling. Prefill only becomes compute-bound with long prompts. |
| P6 | Is the warmup request slower? Why? | Yes: KV cache setup and weight loading | Yes, 2,338 ms, but only the first request after server start (second run's warmup: 26.7 ms) | Right answer, wrong reason: weights and KV cache are set up before the server reports ready. The cost is one-time lazy initialization on the server's first request, which is why production services warm up before taking traffic. |
| P7 | Do stream chunks = output tokens? | No: one word can be several tokens | 256/256; 1,016/1,024 with ignore_eos | Mostly one chunk per token; a multi-token word streams in pieces. The 8 missing chunks are special tokens generated after EOS because of ignore_eos, which have no visible text. Verified on Day 2: the same request with `skip_special_tokens: false` contained exactly 8 (`<\|endoftext\|>`, `<\|im_end\|>`, `<\|im_start\|>`), and 1,016 + 8 = 1,024. |
| P8 | 256 → 1,024 output tokens: does TPOT change? | Goes up | Flat: 17.5 → 16.9 ms | Per step the GPU reads 3.2 GB of weights but only ~28 MB of KV at 1,000 tokens (<1%). KV reads only matter at ~100k tokens of context, or when many sequences are batched together. |

### Startup log lines worth keeping

```
Model loading took 2.98 GiB memory and 38.08 seconds
torch.compile took 13.42 s in total
Graph capturing finished in 7 secs, took 0.19 GiB
Free memory on device (4.95/6.0 GiB) on startup. Desired GPU memory utilization is (0.75, 4.5 GiB).
  Actual usage is 3.24 GiB for consumed memory (weights + non-torch), 0.79 GiB for peak activation,
  and 0.19 GiB for CUDAGraph memory.
Available KV cache memory: 0.48 GiB
GPU KV cache size: 17,824 tokens, Maximum concurrency for 4,096 tokens per request: 4.35x
init engine (profile, create kv cache, warmup model) took 59.30 s (compilation: 13.42 s)
```

### Raw output

```
$ python loadtest/single_request.py
warmup: ttft 2338.3 ms, 59.0 tok/s
prompt_tokens=39 output_tokens=256 chunks=256
      ttft_ms: median     26.2   min     25.7   max    129.5
      tpot_ms: median     17.5   min     16.7   max     19.0
 decode_tok_s: median     57.3   min     52.7   max     60.0
       e2e_ms: median   4487.1   min   4274.3   max   4861.9

$ python loadtest/single_request.py --max-tokens 1024
warmup: ttft 26.7 ms, 58.7 tok/s
prompt_tokens=39 output_tokens=1024 chunks=1016
      ttft_ms: median     25.9   min     24.7   max     29.1
      tpot_ms: median     16.9   min     16.8   max     17.3
 decode_tok_s: median     59.1   min     58.0   max     59.4
       e2e_ms: median  17327.8   min  17233.6   max  17677.1
```

### What I learned

- **Batch-1 decode is memory-bandwidth bound.** Every token reads all 3.2 GB of weights, so the
  ceiling is bandwidth ÷ model size (192 GB/s ÷ 3.2 GB ≈ 60 tok/s). vLLM reaches ~98% of it.
  A faster GPU for this means more bandwidth, not more FLOPs; a smaller or quantized model also helps.
- **On a 6 GB GPU, KV cache memory is the real capacity limit.** After weights, activations and
  CUDA graphs, only 0.48 GiB was left: 17,824 tokens, or 4.35 full 4,096-token conversations.
  Correction from Day 2: this varies between server starts. With the same flags, a later start
  got 0.79 GiB (29,744 tokens) because the profiled peak activation fell from 0.79 to 0.47 GiB.
  The cause of that drop is unknown.
- **TTFT ≈ one decode step for short prompts.** A 39-token prefill costs ~17 ms of GPU time plus
  ~9 ms of HTTP and tokenization.
- **The first request after startup is slow (2.3 s)** because of one-time lazy initialization,
  so a production service should send warmup requests before reporting ready.
- **Setup gotchas on WSL2:** vLLM 0.32 needs Windows driver ≥ 580 (CUDA 13), `python3.12-dev` for
  kernel compilation, and `VLLM_USE_FLASHINFER_SAMPLER=0` when there is no CUDA toolkit (nvcc).

**Next:** Day 2, a load tester that sends many requests at once and reports p50/p95/p99.

## Day 2: Load tester

Requests: 57 prompt tokens + 128 output tokens unless noted. Server start for these runs:
KV cache 0.79 GiB, 29,744 tokens (Day 1's start had 17,824, see the correction above).

### Predictions (written before running)

| # | Question | Prediction | Whose | Measured | Why the difference? |
|---|---|---|---|---|---|
| P1 | Closed, 1 user: TPOT matches Day 1 (~17 ms)? | Yes | Mine | 18.6 ms | About 9% slower, cause not established. TPOT at 128, 256 and 1,024 tokens fits ~16.7 ms per token plus 200–240 ms fixed per request, but this is only a fit across different server starts. |
| P2 | Closed, 8 users: total output tok/s | 472 (8 × 59) | Mine | 399 | Right idea: batching scales almost linearly. Steps are slightly longer and gaps between requests lose some time. |
| P3 | Closed, 8 users: TPOT per user | 17 ms | Mine | 19.1 ms | Only 3% above the 1-user run (18.6 ms). The weight read is shared by the batch. |
| P2b | Same as P2, second estimate | 250–400 tok/s | Claude | 399 | In range, at the top edge. Underestimated how well batching scales. |
| P3b | Same as P3, second estimate | 20–30 ms | Claude | 19.1 ms | Just missed, for the same reason. |
| P4 | Open, 1 req/s: p95 TTFT | 50–100 ms | Claude | 66 ms | In range. |
| P5 | Open, 10 req/s: p95 TTFT | 100–500 ms, with TPOT hit harder than TTFT | Claude | 344 ms; p50 TTFT +56%, p50 TPOT +22% vs 1 req/s | TTFT range right. TPOT claim wrong in percentage terms, right in time lost: of ~630 ms extra p50 E2E, ~530 ms is TPOT and ~32 ms is TTFT. |
| P6 | 32 users, 1,024 output tokens: what fits, what happens | ~16 fit at full length; all start, the cache fills near 550 tokens each, then preemptions | Claude | ~27 fit; all 32 ran, cache 100% near 930 tokens each, 4 preemptions | Pattern right, numbers wrong: they assumed Day 1's 17,824 tokens, but this start had 29,744. |

### Results

| Test | Req/s | Output tok/s | TTFT p50 / p95 (ms) | TPOT p50 / p95 (ms) | E2E p50 (ms) | Preemptions |
|---|---|---|---|---|---|---|
| Closed, 1 user | 0.41 | 53 | 33 / 161 | 18.6 / 18.7 | 2,398 | 0 |
| Closed, 8 users | 3.12 | 399 | 89 / 246 | 19.1 / 19.4 | 2,515 | 0 |
| Open, 1 req/s | 1.13 | 144 | 58 / 66 | 18.9 / 19.3 | 2,464 | 0 |
| Open, 10 req/s | 5.78 | 740 | 90 / 344 | 23.1 / 26.1 | 3,092 | 0 |
| Closed, 32 users, 1,024 tokens | 1.18 | 1,208 | 328 / 328 | 23.7 / 25.8 | 24,576 | 4 |

Sampled during the 32-user run: 32 running with KV usage climbing steadily to 98%, then 100%
with 29 running and 3 waiting. The four slowest requests took 25.2–27.1 s against 24.6 s or
less for the rest, so they are probably the preempted ones, but the data does not link them directly.

### Limits of these numbers

- **Open 10 req/s is not a steady-state test.** Arrivals lasted 4.35 s at 8.96 req/s. The reported
  5.78 req/s and "17.6 in the system" are diluted by the drain after arrivals stopped. Demand
  would be 1,280 tok/s, above the best measured, so sustaining 10 req/s is unproven. Day 4 needs longer runs.
- **Small samples.** With 10 requests p95 is the slowest one (the 161 ms first request); with 40, p99 is the slowest.
- **Cache size is not fixed between starts.** Pin it with `--kv-cache-memory` before the Day 4 tests (flag not yet tested here).

### What I learned

- **Batching is the big win.** Each step reads the weights once and the whole batch shares that read,
  so 8 users got 7.5× the throughput of 1 user at only 3% higher TPOT.
- **TPOT is step time, and every user waits for the same step.** Work per user (KV read, compute,
  streaming) makes the step longer. At 32 users TPOT rose from 16.9 ms to 23.7 ms; the KV read
  explains an estimated 2.7 ms of that (510 MB ÷ 192 GB/s), and the rest is not yet broken down.
- **Closed-loop users fall into lockstep.** With 8 users, TTFT repeats in rounds (`64, 88, 89, 89...`)
  because everyone finishes and resends together. Real traffic does not do this.
- **A closed loop hides overload.** If the server slows down, users send less often, so the arrival
  rate drops and no queue builds up (coordinated omission). An open loop keeps sending on schedule.
- **Little's Law checks the tester.** Closed, 8 users: 3.12 req/s × 2.56 s = 8.0 requests in the system.

**Next:** Day 3.

## Day 3–5

_To be added._
