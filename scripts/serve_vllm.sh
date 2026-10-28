#!/usr/bin/env bash
# Start vLLM's OpenAI-compatible server and save its log.
# Override defaults with env vars, pass extra vLLM flags as arguments:
#   GPU_UTIL=0.8 ./scripts/serve_vllm.sh --max-num-seqs 1
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
# Windows keeps ~1.2 GiB of this 6 GiB GPU busy, and vLLM refuses to start
# if this fraction of total VRAM is more than what is free.
GPU_UTIL="${GPU_UTIL:-0.75}"

# FlashInfer's sampler compiles itself with nvcc on first use, but WSL here has
# only the driver, no CUDA toolkit. Use PyTorch's sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

mkdir -p logs
LOG="logs/vllm_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

vllm serve "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  "$@" 2>&1 | tee "$LOG"
