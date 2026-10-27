# 01: vLLM baseline

**Question:** how much faster is a real serving engine than a naive server, and why?

## Day 1: First request through vLLM

Setup: Qwen2.5-1.5B-Instruct, bf16, `--max-model-len 4096 --gpu-memory-utilization 0.75`, RTX 4050 Laptop 6 GB.

### Predictions (write these before running)

| # | Question | My prediction | Measured | Why the difference? |
|---|---|---|---|---|
| P1 | GiB of VRAM for model weights | 2.9 GiB (1.54B params × 2 bytes) | | |
| P2 | KV cache bytes per token (from config.json) | 28 KiB (2 × 28 layers × 2 KV heads × 128 × 2 bytes) | | |
| P3 | Tokens that fit in the KV cache | ~1.2 GiB → ~45k tokens → ~11 requests of 4,096 tokens | | |
| P4 | Batch-1 decode tok/s (ceiling ~62) | ~37 tok/s (60% of ceiling) | | |
| P5 | TTFT for a short prompt (ms) | 50 ms | | |
| P6 | Is the warmup request slower? Why? | Yes: KV cache setup and weight loading | | |
| P7 | Do stream chunks = output tokens? | No: one word can be several tokens | | |
| P8 | 256 → 1,024 output tokens: does TPOT change? | Goes up | | |

### Startup log lines worth keeping

```
(paste: model weights memory, KV cache size / # GPU blocks, maximum concurrency)
```

### What I learned

-

## Day 2–5

_To be added._
