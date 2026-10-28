# llm-serving

How many users can one GPU serve, at what latency, and what changes that number?

Month 2 of my move into ML infrastructure (Month 1: [ml-fundamentals](https://github.com/lavina98/ml-fundamentals)).
I benchmark LLM serving systems on a laptop GPU, build my own batching server, and measure
how batching, KV cache, quantization and speculative decoding change latency, throughput and capacity.

> Status: in progress. Results are added as each experiment completes.

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop, 6 GB GDDR6, ~192 GB/s, 115 W |
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| Main model | Qwen2.5-1.5B-Instruct |

Month 1 ran on a different GPU (RTX 3050 8 GB), so numbers are not directly comparable.

## Experiments

| # | Experiment | Question | Status |
|---|---|---|---|
| 01 | [vLLM baseline](experiments/01-vllm-baseline/NOTES.md) | How much faster is a real serving engine than a naive server, and why? | in progress |
| 02 | [Batching server](experiments/02-batching-server/NOTES.md) | Build static and continuous batching myself. Where does the win come from? | not started |
| 03 | [Quantization and profiling](experiments/03-quantization-profiling/NOTES.md) | What do int4/fp8 and speculative decoding buy, and what do they cost? | not started |
| 04 | [Triton and hardening](experiments/04-triton-hardening/NOTES.md) | What does a production-ready setup need, and how does it fail? | not started |

## Key results

_Filled in as experiments complete._

## Repo layout

```
loadtest/      load generator and metrics (TTFT, TPOT, p95, tokens/s)
scripts/       server start-up and environment checks
experiments/   one folder per experiment, each with NOTES.md
results/       raw benchmark CSVs and charts
docs/          architecture and failure analysis
```

## Setup (WSL2)

1. Install Ubuntu in WSL from an admin PowerShell window: `wsl --install -d Ubuntu-24.04`
2. Do **not** install an NVIDIA driver inside WSL. The Windows driver provides CUDA to WSL.
3. In Ubuntu (`python3.12-dev` is needed because vLLM compiles GPU kernels at startup;
   vLLM 0.32 also needs Windows NVIDIA driver 580 or newer):
   ```bash
   sudo apt update && sudo apt install -y git build-essential curl python3.12-dev
   curl -LsSf https://astral.sh/uv/install.sh | sh   # then open a new shell
   cd ~/llm-serving
   uv venv --python 3.12 && source .venv/bin/activate
   uv pip install -r requirements.txt
   ./scripts/check_env.sh
   ```
4. Start the server: `./scripts/serve_vllm.sh`
