# LLM Nondeterminism Analysis

This repository provides an analysis and visualization system for measuring **token-level nondeterminism** in Large Language Model (LLM) inference across different hardware platforms, model architectures, and batch sizes.

## Overview

The system quantifies how model outputs vary across multiple identical inference runs, examining stability across four key metrics:

* **Std_Prob_Runs:** Standard deviation of probabilities across runs.
* **Range_Prob_Runs:** Range (min-max spread) of probabilities.
* **Mean_Prob_Runs:** Mean probability values.
* **Range_Logit_Runs:** Range of logit values.

## Repository Structure

```text
llm-nondeterminism/  
├── Plot.ipynb                                             # Main visualization pipeline
├── mmlu_10_random_samples.jsonl                           # Target test samples
├── mmlu_1000_random_samples_filler.jsonl                  # Filler samples used for batch padding
├── stability_token_level_report_STD_RANGE_A100.csv        # Standard models on A100  
├── stability_token_level_report_STD_RANGE_A6000.csv       # Standard models on A6000  
├── stability_token_level_report_STD_RANGE_H200.csv        # Standard models on H200  
├── stability_token_level_report_STD_RANGE_Ascend-910.csv  # Standard models on Ascend  
├── stability_token_level_report_STD_RANGE_gemma_A100.csv  # Gemma variants on A100  
├── stability_token_level_report_STD_RANGE_gemma_A6000.csv # Gemma variants on A6000  
├── stability_token_level_report_STD_RANGE_gemma_H200.csv  # Gemma variants on H200  
├── stability_token_level_report_STD_RANGE_gemma_Ascend-910.csv  # Gemma variants on Ascend  
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
