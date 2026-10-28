# 01: vLLM baseline

**Question:** how much faster is a real serving engine than a naive server, and why?

## Day 1: First request through vLLM

Setup: Qwen2.5-1.5B-Instruct, bf16, `--max-model-len 4096 --gpu-memory-utilization 0.75`, RTX 4050 Laptop 6 GB.

### Predictions (write these before running)

| # | Question | My prediction | Measured | Why the difference? |
|---|---|---|---|---|
| P1 | GiB of VRAM for model weights | 2.9 GiB (1.54B params × 2 bytes) | 2.98 GiB | Close. The real parameter count is slightly above 1.54B, plus small buffers (e.g. RoPE tables). |
| P2 | KV cache bytes per token (from config.json) | 28 KiB (2 × 28 layers × 2 KV heads × 128 × 2 bytes) | 0.48 GiB ÷ 17,824 tokens ≈ 28 KiB | Matches. Only attention K/V is cached; GQA (2 KV heads for 12 query heads) makes it 6× smaller than full multi-head attention. |
| P3 | Tokens that fit in the KV cache | ~1.2 GiB → ~45k tokens → ~11 requests of 4,096 tokens | 0.48 GiB → 17,824 tokens → 4.35 requests | Underestimated overhead: non-torch memory (CUDA context, libraries) 0.26 GiB, activations 0.79 GiB not 0.4 (profiling runs a 2,048-token batch, and logits over a 152k vocab are large), and CUDA graphs 0.19 GiB, which I forgot. |
| P4 | Batch-1 decode tok/s (ceiling ~62) | ~37 tok/s (60% of ceiling) | 57–59 tok/s; TPOT 16.9–17.5 ms | Real ceiling uses the real weight size: 3.2 GB ÷ 192 GB/s = 16.7 ms → ~60 tok/s, so vLLM is at ~98% of memory bandwidth. CUDA graphs remove the kernel-launch overhead that limited batch-1 decode in Month 1. |
| P5 | TTFT for a short prompt (ms) | 50 ms | 26 ms (39-token prompt) | A short prefill still reads all weights once, so it costs about one decode step (~17 ms) plus ~9 ms of HTTP, tokenization and scheduling. Prefill only becomes compute-bound with long prompts. |
| P6 | Is the warmup request slower? Why? | Yes: KV cache setup and weight loading | Yes, 2,338 ms, but only the first request after server start (second run's warmup: 26.7 ms) | Right answer, wrong reason: weights and KV cache are set up before the server reports ready. The cost is one-time lazy initialization on the server's first request, which is why production services warm up before taking traffic. |
| P7 | Do stream chunks = output tokens? | No: one word can be several tokens | 256/256; 1,016/1,024 with ignore_eos | Mostly one chunk per token; a multi-token word streams in pieces. The 8 missing chunks are special tokens (e.g. end-of-turn) generated after EOS because of ignore_eos, which have no visible text. |
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
  CUDA graphs, only 0.48 GiB is left: 17,824 tokens, or 4.35 full 4,096-token conversations.
- **TTFT ≈ one decode step for short prompts.** A 39-token prefill costs ~17 ms of GPU time plus
  ~9 ms of HTTP and tokenization.
- **The first request after startup is slow (2.3 s)** because of one-time lazy initialization,
  so a production service should send warmup requests before reporting ready.
- **Setup gotchas on WSL2:** vLLM 0.32 needs Windows driver ≥ 580 (CUDA 13), `python3.12-dev` for
  kernel compilation, and `VLLM_USE_FLASHINFER_SAMPLER=0` when there is no CUDA toolkit (nvcc).

**Next:** Day 2, a load tester that sends many requests at once and reports p50/p95/p99.

## Day 2–5

_To be added._
