# GPTGAT2 P/PP Local Runbook

## Goal
Run the new P/PP training design locally with `uv` on this machine and use the local NVIDIA GeForce RTX 4070 Ti SUPER for smoke testing before any large experiment.

## Environment
- Python: `3.10`
- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- Recommended driver/CUDA path: use the currently working PyTorch CUDA build already validated on the machine.

## UV bootstrap
```bash
uv venv --python 3.10 .venv
. .venv/Scripts/activate
uv pip install -r code/requirements.txt
```

If `torch-geometric` wheels fail, install the exact CUDA-compatible wheel set currently used by the project before retrying the remaining requirements.

## Sanity checks
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m unittest discover -s code/tests
```

## Minimal smoke run
Run a very short experiment before full training.

```bash
python code/train.py --config code/config.yaml
```

For smoke testing, temporarily use:
- fewer epochs
- smaller batch size
- reduced dataset subset if needed

## What to verify
- training starts without schema errors
- P/PP configuration keys load correctly
- stage helpers resolve without runtime errors
- TensorBoard/logging writes normally
- GPU memory usage is stable on the 4070 Ti SUPER

## Experiment ladder
1. Unit tests only
2. Single-epoch smoke run
3. P-only baseline
4. P + PP warmup only
5. Full hybrid schedule
6. Ablations (decoder size, PP ratio, anti-forgetting weight)
