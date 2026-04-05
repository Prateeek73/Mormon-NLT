# Environment Setup Guide

## Hardware Profile

| Item | Spec |
|------|------|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| VRAM | 8 GB |
| CUDA Driver | 595.79 |
| Required CUDA Toolkit | 12.8 (Blackwell) |

## VRAM Constraints and Model Strategy

With 8 GB VRAM the training strategy changes from the default plan:

| Track | Model | Method | Est. VRAM | Feasible? |
|-------|-------|--------|-----------|-----------|
| LoRA  | Qwen2.5-3B-Instruct | 4-bit QLoRA (r=16) | ~6 GB | Yes |
| FFT   | Qwen2.5-1.5B-Instruct | BF16, paged optimizer | ~7 GB | Yes |

**Why these choices?**
- FFT on 3B requires ~18-22 GB — not possible on 8 GB
- Qwen2.5-1.5B in BF16 uses ~3 GB weights + ~4 GB optimizer states = fits
- LoRA on 3B with 4-bit quantization (QLoRA) uses ~3 GB weights + ~2 GB adapters/activations = fits
- This still gives a meaningful LoRA vs FFT comparison at different model scales

---

## Step 1 — Create Conda Environment (Python 3.11)

> Python 3.14 (system) and 3.13 (conda base) are too new for ML packages.
> Most wheels (torch, bitsandbytes, flash-attn) only publish up to Python 3.11/3.12.

```bash
conda create -n advnlp python=3.11 -y
conda activate advnlp
```

## Step 2 — Install PyTorch (CUDA 12.8 for RTX 50 series)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Step 3 — Install Project Dependencies

```bash
pip install -r requirements.txt
```

## Step 4 — Install Jupyter Kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name advnlp --display-name "AdvNLP (Python 3.11)"
```

## Step 5 — (Optional) Flash Attention 2

Gives 2-3x faster training. RTX 5070 supports it.

```bash
pip install flash-attn --no-build-isolation
```

If it fails to build, skip it — `attn_implementation='eager'` works fine.

---

## Running the Project

```bash
conda activate advnlp
cd "c:/Users/pra73/Desktop/Projects/Active_Gits/AdvNLP"
jupyter notebook
```

Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08

Select kernel: **AdvNLP (Python 3.11)**

---

## 8 GB VRAM Settings Reference

These are already set correctly in the notebooks, but for reference:

### LoRA (NB04) — Qwen2.5-3B-Instruct + 4-bit QLoRA
```python
quantization='4bit'              # NF4 quantization
per_device_train_batch_size=2
gradient_accumulation_steps=8    # effective batch = 16
```

### FFT (NB05) — Qwen2.5-1.5B-Instruct + BF16
```python
MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
per_device_train_batch_size=1
gradient_accumulation_steps=16   # effective batch = 16
optim='paged_adamw_32bit'
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `bitsandbytes` CUDA error | Ensure `pip install bitsandbytes>=0.45.0` (Blackwell support) |
| CUDA out of memory | Reduce `per_device_train_batch_size` to 1, set `max_length=256` |
| `flash_attn` build fails | Skip it, use `attn_implementation='eager'` |
| Kernel not showing in VS Code | Reload VS Code window after `ipykernel install` |
| Slow download of Qwen model | First run downloads ~3-6 GB; subsequent runs use cache |
| `OMP Error #15: libiomp5md.dll already initialized` | Duplicate OpenMP DLLs (conda + PyTorch). Fix permanently: `conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE` then `conda deactivate && conda activate advnlp` |
