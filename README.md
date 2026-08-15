# LLM Nondeterminism Analysis

This repository provides an analysis and visualization system for measuring **token-level nondeterminism** in Large Language Model (LLM) inference across different hardware platforms, model architectures, and batch sizes.

**Paper:** [arXiv:2601.06118](https://arxiv.org/pdf/2601.06118) — *currently under review at IEEE Transactions on Computers (TC)*.

## Overview

The system quantifies how model outputs vary across multiple identical inference runs, examining stability across key metrics:

* **Std_Prob_Runs:** Standard deviation of probabilities across runs.
* **Range_Prob_Runs:** Range (min-max spread) of probabilities.
* **Mean_Prob_Runs:** Mean probability values.
* **Range_Logit_Runs:** Range of logit values.

Beyond the core sweep over GPUs and batch sizes, the data also covers three supplementary settings: a multi-GPU run (2×A6000), a closed-source API model (GPT-4o), and a non-MMLU task (Infinity-Chat).

## Repository Structure

```text
llm-nondeterminism/
├── Plot.ipynb                                              # Main visualization pipeline
├── mmlu_10_random_samples.jsonl                            # Target test samples
├── mmlu_1000_random_samples_filler.jsonl                  # Filler samples used for batch padding
├── infinichat_10_samples.jsonl                            # Open-ended Infinity-Chat target prompts
│
├── stability_token_level_report_STD_RANGE_A100.csv        # Standard models on A100
├── stability_token_level_report_STD_RANGE_A6000.csv       # Standard models on A6000
├── stability_token_level_report_STD_RANGE_H200.csv        # Standard models on H200
├── stability_token_level_report_STD_RANGE_Ascend-910.csv  # Standard models on Ascend
├── stability_token_level_report_STD_RANGE_gemma_A100.csv  # Gemma variants on A100
├── stability_token_level_report_STD_RANGE_gemma_A6000.csv # Gemma variants on A6000
├── stability_token_level_report_STD_RANGE_gemma_H200.csv  # Gemma variants on H200
├── stability_token_level_report_STD_RANGE_gemma_Ascend-910.csv  # Gemma variants on Ascend
├── stability_token_level_report_STD_RANGE_2xA6000.csv     # Gemma3-12B sharded across two A6000s
├── stability_token_level_report_STD_RANGE_gpt-4o_api.csv  # Closed-source GPT-4o (repeated API calls)
├── stability_token_level_report_STD_RANGE_infinichat.csv  # Task-independence (Infinity-Chat)
├── adversarial_stability_token_level_report_STD_RANGE_Huawei.csv  # Adversarial prompts on Ascend
├── branched_trace_data.csv                                # Divergence trace data
└── Figures_final/                                         # Output directory
```

## Running the Analysis

1. Open `Plot.ipynb` in Jupyter Notebook or JupyterLab.
2. Execute the cells sequentially to generate all figures.
3. Results will be saved to the `Figures_final/` directory.

Each figure generation cell in the notebook follows this pattern:

```python
# Configuration
GPU_name = 'A100'  # Change to target GPU
CSV_FILE_PATH = f"stability_token_level_report_STD_RANGE_{GPU_name}.csv"

# Data loading and processing
df_raw = pd.read_csv(CSV_FILE_PATH)
# ... processing logic ...

# Visualization and export
plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

## Revision Additions (R2)

Files added during the second revision round, supporting the new experiments and statistical analyses referenced in the paper:

```text
├── stability_token_level_report_STD_RANGE_precision_A100.csv  # Precision study: DeepSeek-Qwen3-8B under BF16/FP16/FP32 (A100, B=16 vs. B=1)
├── ks_test_results.csv                    # Pairwise Kolmogorov-Smirnov tests of sigma distributions across GPU platforms (806 comparisons: KS D, p-value, normalized Wasserstein)
├── Figures_final/Fig_Precision/           # Precision-study figures (sigma and range vs. probability, three formats)
└── scripts/
    ├── precision_gen.py                   # Precision study: generation (parameterized model/dtype/batch; TF32 disabled for FP32)
    ├── precision_eval.py                  # Precision study: token-level evaluation -> CSV
    ├── precision_plot.py                  # Precision study: comparison figures
    └── ks_test_gpus.py                    # Cross-GPU KS tests -> ks_test_results.csv
```
