# 01: vLLM baseline

**Question:** how much faster is a real serving engine than a naive server, and why?

## Day 1: First request through vLLM

Setup: Qwen2.5-1.5B-Instruct, bf16, `--max-model-len 4096 --gpu-memory-utilization 0.75`, RTX 4050 Laptop 6 GB.

### Predictions (write these before running)

| # | Question | My prediction | Measured | Why the difference? |
|---|---|---|---|---|
| P1 | GiB of VRAM for model weights | | | |
| P2 | KV cache bytes per token (from config.json) | | | |
| P3 | Tokens that fit in the KV cache | | | |
| P4 | Batch-1 decode tok/s (ceiling ~62) | | | |
| P5 | TTFT for a short prompt (ms) | | | |
| P6 | Is the warmup request slower? Why? | | | |
| P7 | Do stream chunks = output tokens? | | | |

### Startup log lines worth keeping

```
(paste: model weights memory, KV cache size / # GPU blocks, maximum concurrency)
```

### What I learned

-

## Day 2–5

_To be added._
