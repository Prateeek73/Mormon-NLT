# Mormon-NLT: Setup & Running Guide

**Project**: Modern English → Shakespearean Style Transfer using Qwen2.5 + LoRA/FFT

See **REPORT.md** for comprehensive experimental results, metrics, and findings.

---

## Hardware Profile

| Item | Spec |
|------|------|
| **GPU** | NVIDIA GeForce RTX 5070 Laptop GPU |
| **VRAM** | 8.5 GB |
| **CUDA Driver** | 12.8 |
| **PyTorch** | 2.11.0+cu128 |

---

## Environment Setup

### Step 1 — Create Conda Environment (Python 3.11)

```bash
conda create -n mormon-nlt python=3.11 -y
conda activate mormon-nlt
```

### Step 2 — Install PyTorch (CUDA 12.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
PyTorch: 2.11.0+cu128
CUDA: True
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
```

### Step 3 — Clone Repository & Install Dependencies

```bash
cd /home/prasingh/data/Mormon-NLT
pip install -r requirements.txt
```

**Key dependencies**:
- `transformers` ≥ 4.35.0 (Qwen2.5 support)
- `peft` (LoRA)
- `datasets`
- `sacrebleu`, `bert-score` (evaluation metrics)
- `peft`, `accelerate`, `bitsandbytes` (training)

### Step 4 — Install Jupyter Kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name mormon-nlt --display-name "Mormon-NLT"
```

### Step 5 — (Optional) Flash Attention 2

For 2-3x faster training on RTX 5070 (Blackwell, sm_120):

```bash
pip install "flash-attn>=2.8.3" --no-build-isolation
```

**If build fails**: Use `attn_implementation='sdpa'` (PyTorch SDPA) instead—performance difference is minimal with modern PyTorch.

---

## Running the Project

### Quick Start

```bash
conda activate mormon-nlt
cd /home/prasingh/data/Mormon-NLT
jupyter notebook
```

Select kernel: **Mormon-NLT**

### Notebook Execution Order

| Step | Notebook | Purpose | Runtime |
|------|----------|---------|---------|
| 1 | `01_data_download_and_preprocessing.ipynb` | Download data, create train/val/test splits | ~2 min |
| 2 | `02_eda_shakespeare_dataset.ipynb` | Analyze vocabulary, lengths, distributions | ~3 min |
| 3 | `03_model_setup_and_tokenizer.ipynb` | Load Qwen2.5, verify tokenizer, zero-shot test | ~5 min |
| 4a | `04_exp1_lora_training.ipynb` | Train LoRA (Exp1, r=16, bidirectional) | ~4.0 hrs |
| 4b | `04_exp2_lora_training.ipynb` | Train LoRA (Exp2, r=16, early stopping) | ~2.0 hrs |
| 4c | `04_exp3_lora_training.ipynb` | Train LoRA (Exp3, r=32, unidirectional) | ~1.5 hrs |
| 5a | `05_exp1_fft_training.ipynb` | Fine-tune (Exp1, 1.5B, bidirectional) | ~2.0 hrs |
| 5b | `05_exp2_fft_training.ipynb` | Fine-tune (Exp2, 1.5B, higher LR) | ~2.0 hrs |
| 5c | `05_exp3_fft_training.ipynb` | Fine-tune (Exp3, 1.5B, unidirectional) | ~1.0 hrs |
| 6a | `06_exp1_bleu_evaluation.ipynb` | Evaluate Exp1 models | ~2.5 hrs |
| 6b | `06_exp2_bleu_evaluation.ipynb` | Evaluate Exp2 models | ~2.5 hrs |
| 6c | `06_exp3_bleu_evaluation.ipynb` | Evaluate Exp3 models | ~2.5 hrs |
| 7a | `07_exp1_comparison_and_results.ipynb` | Compare LoRA vs FFT (Exp1) | ~1 min |
| 7b | `07_exp2_comparison_and_results.ipynb` | Compare across Exp1-2 | ~1 min |
| 7c | `07_exp3_comparison_and_results.ipynb` | Compare all 3 experiments | ~1 min |
| 8 | `08_overall_comparison.ipynb` | Aggregate all 6 variants | ~1 min |
| 9 | `08_qa_testing.ipynb` | Manual QA validation (10 samples) | ~1 min |

**Total runtime**: ~18-22 hours (sequential on RTX 5070)

You can parallelize steps 4a/4b/4c and 5a/5b/5c if you have multiple GPUs (not applicable here).

---

## Training Configuration

### LoRA Training (Notebooks 04_exp*_lora_training.ipynb)

Configuration for **Qwen2.5-3B-Instruct** with LoRA:

```python
# Exp1: Baseline
lora_r = 16
lora_alpha = 32
epochs = 3
learning_rate = 2e-4
per_device_train_batch_size = 8
bidirectional = True

# Exp2: Early stopping
lora_r = 16
lora_alpha = 32
epochs = 2  # with early stopping (patience=1)
learning_rate = 2e-4
per_device_train_batch_size = 8
bidirectional = True

# Exp3: Unidirectional + higher rank
lora_r = 32
lora_alpha = 64
epochs = 2  # with early stopping
learning_rate = 2e-4
per_device_train_batch_size = 8
bidirectional = False  # Modern→Shakespeare only
```

**VRAM**: 8-9 GB (fits on RTX 5070 laptop)

### FFT Training (Notebooks 05_exp*_fft_training.ipynb)

Configuration for **Qwen2.5-1.5B-Instruct** with full fine-tuning:

```python
# All experiments (Exp1-3 only differ in LR and data direction)
epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 1
learning_rate = 2e-5 (Exp1), 5e-5 (Exp2-3)
max_length = 512
attention_implementation = 'sdpa'  # PyTorch SDPA (or 'flash_attention_2' if available)
bidirectional = False  # Only Exp1 uses bidirectional; Exp2-3 unidirectional
```

**VRAM**: 9-10 GB (fits on RTX 5070 laptop)

---

## Understanding the Experiments

### Exp 1 (Baseline)
- **Data**: Bidirectional (Modern↔Shakespeare)
- **LoRA**: r=16, 3 epochs
- **FFT**: 1.5B, 3 epochs, LR=2e-5
- **Metric**: BLEU, ChrF only
- **Results**: Baseline—establishes reference point

### Exp 2 (Regularization)
- **Data**: Bidirectional
- **LoRA**: r=16, 2 epochs + early stopping (addresses overfitting)
- **FFT**: 1.5B, 3 epochs, LR=5e-5 (addresses underfitting)
- **Metric**: BLEU, ChrF, BERTScore (distilbert-base-uncased)
- **Results**: Modest improvements (~2-3%)

### Exp 3 (Optimal)
- **Data**: Unidirectional Modern→Shakespeare only (focused learning signal)
- **LoRA**: r=32, 2 epochs + early stopping (increased capacity on cleaner data)
- **FFT**: 1.5B, 3 epochs, LR=5e-5, unidirectional
- **Metric**: BLEU, ChrF, BERTScore (roberta-large for better quality)
- **Results**: Major breakthrough—+20% BERTScore gains (0.695→0.8405 LoRA, 0.6831→0.8415 FFT)

See **REPORT.md** for detailed justifications and results.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | Reduce `per_device_train_batch_size` to 2 (LoRA) or 1 (FFT). Set `max_length=256`. |
| **GPU not detected** | Run `python -c "import torch; print(torch.cuda.get_device_name(0))"`. If nothing, reinstall PyTorch with correct CUDA index. |
| **`bitsandbytes` error** | Update: `pip install --upgrade bitsandbytes` |
| **`flash_attn` build fails** | Skip it and use `attn_implementation='sdpa'` in notebooks (already default). |
| **Jupyter kernel not showing** | Reload VS Code/refresh Jupyter hub after `ipykernel install`. |
| **Slow model download first run** | First run downloads ~3-6 GB. Subsequent runs use cache (`.../huggingface_cache/`). |
| **"NotImplementedError: No module named 'flash_attn'"** | Optional—just a performance enhancement. Notebooks default to `sdpa` which works fine. |

---

## Project Structure

```
Mormon-NLT/
├── data/
│   ├── raw/                          # Downloaded datasets (HuggingFace cache)
│   └── processed/                    # train.jsonl, val.jsonl, test.jsonl
├── notebooks/                        # Jupyter notebooks (01-08)
├── outputs/
│   ├── exp1/
│   │   ├── lora/final_adapter/      # LoRA weights
│   │   ├── fft/final_model/         # FFT model
│   │   └── results/                 # Metrics, figures
│   ├── exp2/results/                # Exp2 evaluation
│   └── exp3/results/                # Exp3 evaluation (best results)
├── src/
│   ├── data_utils.py                # Dataset loading/preprocessing
│   ├── model_utils.py               # Model loading, info printing
│   └── evaluation.py                # BLEU, ChrF, BERTScore computation
├── README.md                         # This file (Setup & Running)
├── REPORT.md                         # Comprehensive results & analysis
└── requirements.txt                 # Python dependencies
```

---

## Key Findings (See REPORT.md for Details)

✅ **Unidirectional training** is transformative: +20-23% BERTScore improvement (Exp2→Exp3)

✅ **LoRA matches FFT quality** with **100x fewer parameters**: LoRA 26M (0.84 F1) ≈ FFT 1.54B (0.84 F1)

⚠️ **BERTScore inadequate for validation**: Manual QA shows 49-64pp gap between metric and semantic preservation

📊 **Best deployment option**: Exp3 LoRA (26M params, 0.84 F1, 1.5-2.5 hrs inference)

For full analysis, metrics, and experiment justifications, see **REPORT.md**.

---

## Dataset

**Sources**:
- ayaan04/english-to-shakespeare: 18,395 pairs (HuggingFace)
- Roudranil/shakespearean-and-modern-english-conversational-dataset: 8,787 pairs (5,272 train + 3,515 held-out test)
- cobanov/shakespeare-dataset: 42 raw text files for vocabulary analysis

**Final splits**:
- Train: 40,084 records (20,042 unique pairs × 2 for bidirectional training)
- Validation: 2,234 records
- Test (held-out): 3,515 records

**Format**: Chat-format JSONL (system prompt + user input + assistant target)

---

## Requirements

- Python 3.11+
- PyTorch 2.11+
- CUDA 12.8 (for RTX 5070)
- ~50 GB disk (model cache + data)
- RTX 5070 Laptop GPU (8.5 GB VRAM minimum)

---

## Next Steps

1. Set up environment (Steps 1-5 above)
2. Run notebooks in order (01→08)
3. Check results in `outputs/exp3/results/`
4. Review findings in **REPORT.md**

Questions? See troubleshooting section above, or check notebook comments.
