#!/usr/bin/env bash
# Check that WSL can see the GPU and the Python stack is installed.
set -uo pipefail

echo "== GPU (from the Windows driver) =="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv

echo; echo "== Python =="
python --version

echo; echo "== PyTorch + CUDA =="
python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} | {p.total_memory / 2**30:.2f} GiB | compute capability {p.major}.{p.minor}")
EOF

echo; echo "== vLLM =="
python -c "import vllm; print('vllm', vllm.__version__)"
