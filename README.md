# TelAgentBench-ID: Indonesian Telecommunications BSS Tool-Calling Benchmark & QLoRA Adaptation

This repository contains the localization pipeline, SFT training notebooks, evaluation harnesses, standardized result tables, and figure generation scripts for evaluating tool-calling language models on Indonesian telecommunications Business Support Systems (BSS) workflows (`TelkomNusa`).

## Artifacts

- **Test Benchmark Dataset**: [`Rislantrs/TelAgentBench-ID`](https://huggingface.co/datasets/Rislantrs/TelAgentBench-ID)
- **SFT Training Corpus**: [`Rislantrs/TelkomNusa-SFT-ID`](https://huggingface.co/datasets/Rislantrs/TelkomNusa-SFT-ID)
- **Fine-Tuned Adapters**:
  - [`Rislantrs/telecom-troubleshooter-qwen2.5-3b-balanced-lora`](https://huggingface.co/Rislantrs/telecom-troubleshooter-qwen2.5-3b-balanced-lora)
  - [`Rislantrs/telecom-troubleshooter-qwen2.5-7b-balanced-lora`](https://huggingface.co/Rislantrs/telecom-troubleshooter-qwen2.5-7b-balanced-lora)

## Repository Structure

```text
TelAgentBench-ID/
├── notebooks/
│   ├── 01_SFT_Corpus_Synthetic_Generator.ipynb               # Bitext localization + synthetic 23-API & multi-turn SFT builder
│   ├── 02_TelAgentBench_ID_Localization_Pipeline.ipynb         # English/Korean TelAgentBench to Indonesian translation pipeline
│   ├── 03_QLoRA_Training_TelkomNusa_3B_7B.ipynb                # 4-bit NF4 QLoRA training script (Qwen2.5-3B / Qwen2.5-7B)
│   ├── 04_MultiModel_10Pillar_Benchmark_Evaluator_RegimeA.ipynb # Zero-shot evaluation runner (Regime A)
│   ├── 05_MultiModel_3Shot_CoT_Evaluator_RegimeB.ipynb         # 3-shot prompted evaluation runner (Regime B)
│   ├── 06_TelkomNusa_QLoRA_DualModel_Evaluator.ipynb           # Local GPU evaluation for fine-tuned 3B & 7B LoRA adapters
│   └── 07_Benchmark_Visualization_and_Plots.ipynb              # Summary metric plotting notebook
├── results/
│   ├── Master_Standardized_Benchmark_Summary.csv               # Standardized 12-configuration results matching Table 7
│   ├── Master_Combined_Benchmark_V1_V2_V3_SuccessOnly.csv      # Historical V1/V2/V3 evaluation log
│   ├── RegimeA_and_Main_Benchmark/                             # Per-item predictions and Regime A summary
│   └── RegimeB_FewShot_Ablation/                               # Per-item predictions and Regime B ablation summary
├── scripts/
│   ├── generate_3_publication_figures.py                       # Script to generate Figures 2, 3, and 4
│   ├── upload_to_hf.py                                         # Upload script for TelAgentBench-ID
│   └── upload_sft_to_hf.py                                     # Upload script for TelkomNusa-SFT-ID
├── figures/                                                    # Publication figures
├── .env.example
├── .gitignore
└── LICENSE
```

## Setup and Usage

1. Configure credentials in a local `.env` file (never commit `.env`):
   ```bash
   cp .env.example .env
   ```
   Populate `HF_TOKEN`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, and/or `OPENAI_API_KEY` as required by the evaluator notebooks.

2. Reproduce Figures 2-4 from the standardized benchmark table:
   ```bash
   python scripts/generate_3_publication_figures.py
   ```

## Evaluation Regimes

- **Regime A (Zero-Shot)**: Model receives the system prompt and full 23-tool schema catalog with `temperature=0.0` (used for QLoRA-tuned SLMs, untuned Qwen2.5 backbones, and Open-Weight 20B-120B baselines).
- **Regime B (3-Shot Prompting)**: Adds three Indonesian BSS tool-calling exemplars (`GET__rate_unpaid-bills`, `GET__rate_current-data-limits`, `GET__rate_family_bills`) and formatting constraints to elicit structured tool calls from models that otherwise fail to emit `<tool_call>` tags zero-shot (Claude-Opus-5, MiniMax-M3, Llama-3.2-3B-Instruct, Phi-4-mini).

## License

- Code and evaluation scripts: [Apache 2.0](LICENSE)
- Localized datasets: CC BY-SA 4.0
