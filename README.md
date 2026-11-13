# TelAgentBench-ID: Indonesian Telecommunications BSS Tool-Calling Benchmark & QLoRA Adaptation

This repository provides the localization pipeline, QLoRA fine-tuning workflow, unified evaluation harness, standardized benchmark results, and publication figure generator for evaluating tool-calling language models on Indonesian telecommunications Business Support Systems (BSS) workflows (`TelkomNusa`).

## Published Artifacts

- **Test Benchmark Dataset (156 held-out items + extended subsets)**: [`Rislantrs/TelAgentBench-ID`](https://huggingface.co/datasets/Rislantrs/TelAgentBench-ID)
- **SFT Training Corpus (3,129 train / 381 validation)**: [`Rislantrs/TelkomNusa-SFT-ID`](https://huggingface.co/datasets/Rislantrs/TelkomNusa-SFT-ID)
- **Fine-Tuned QLoRA Adapters**:
  - [`Rislantrs/telecom-troubleshooter-qwen2.5-3b-balanced-lora`](https://huggingface.co/Rislantrs/telecom-troubleshooter-qwen2.5-3b-balanced-lora)
  - [`Rislantrs/telecom-troubleshooter-qwen2.5-7b-balanced-lora`](https://huggingface.co/Rislantrs/telecom-troubleshooter-qwen2.5-7b-balanced-lora)

## Repository Structure

```text
TelAgentBench-ID/
├── notebooks/
│   ├── 01_Data_Localization_and_SFT_Builder.ipynb  # Localization pipeline + 3,510-dialogue SFT builder & 4-gram Jaccard audit
│   ├── 02_QLoRA_FineTuning_TelkomNusa.ipynb        # 4-bit NF4 QLoRA training (Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct)
│   └── 03_Benchmark_Evaluation_and_Figures.ipynb   # Unified 156-item evaluator (Regime A & B) + Figures 2-4
├── results/
│   ├── Master_Standardized_Benchmark_Summary.csv   # Final 12-configuration results matching Table 7 in the manuscript
│   └── TelkomNusa_DualModel_Final/                 # Per-item 156-sample predictions & summary for TelkomNusa-3B and 7B
├── scripts/
│   ├── generate_3_publication_figures.py           # Reproduces Figures 2, 3, and 4 from standardized metrics
│   ├── upload_to_hf.py                             # Uploads TelAgentBench-ID to Hugging Face Hub
│   └── upload_sft_to_hf.py                         # Uploads TelkomNusa-SFT-ID to Hugging Face Hub
├── figures/                                        # Manuscript figures (Figures 1-4)
├── .env.example
├── .gitignore
└── LICENSE
```

## Setup and Usage

1. Copy `.env.example` to `.env` (never commit `.env`) and populate only the credentials needed for the models you plan to run (`HF_TOKEN`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`):
   ```bash
   cp .env.example .env
   ```

2. Run the three-stage notebook workflow (designed for Google Colab or Kaggle T4/L4 GPU runtimes):
   - **Stage 1 (`notebooks/01_Data_Localization_and_SFT_Builder.ipynb`)**: Localizes upstream TelAgentBench files into Indonesian and builds the 23-tool `TelkomNusa-SFT-ID` corpus with exact-string and word 4-gram Jaccard ($\tau = 0.30$) decontamination.
   - **Stage 2 (`notebooks/02_QLoRA_FineTuning_TelkomNusa.ipynb`)**: Trains the 4-bit NF4 Unsloth QLoRA adapters (`r=16`, `alpha=32`, all 7 linear modules) on `Rislantrs/TelkomNusa-SFT-ID`.
   - **Stage 3 (`notebooks/03_Benchmark_Evaluation_and_Figures.ipynb`)**: Runs the 156-item evaluation suite (55 Single-Turn execution, 55 Ambiguous/Abstention, 46 Multi-Turn stateless dialogues) and generates publication figures.

3. Reproduce Figures 2-4 directly from the standardized benchmark table:
   ```bash
   python scripts/generate_3_publication_figures.py
   ```

## Evaluation Regimes

- **Regime A (Zero-Shot)**: Evaluates models with the system prompt and full 23-tool OpenAPI schema catalog (`temperature=0.0`), used for the QLoRA-adapted SLMs, untuned `Qwen2.5-3B/7B-Instruct`, `Gemma4-31B`, `GPT-OSS-20B/120B`, and `Nemotron-30B`.
- **Regime B (3-Shot Prompting)**: Supplies explicit formatting instructions and three Indonesian BSS tool-calling exemplars (`GET__rate_unpaid-bills`, `GET__rate_current-data-limits`, `GET__rate_family_bills`) for baselines that fail to emit valid structured tool calls under zero-shot prompting (`Claude-Opus-5`, `MiniMax-M3`, `Llama-3.2-3B-Instruct`, `Phi-4-mini`).

## Citation

If you use this repository or dataset in your research, please cite our manuscript and the upstream TelAgentBench benchmark:

```bibtex
@article{tristansyah2026telagentbenchid,
  title   = {QLoRA-Adapted Small Language Models for On-Premise Telecom BSS Tool-Calling: A Benchmark on Indonesian TelAgentBench},
  author  = {Tristansyah, M Rislan and Ichsan, Ichwan Nul},
  journal = {JUTI: Jurnal Ilmiah Teknologi Informasi},
  year    = {2026},
  note    = {Universitas Pendidikan Indonesia. Code and datasets available at https://github.com/Rislantrs/TelAgentBench-ID}
}

@inproceedings{lee2025telagentbench,
  title     = {TelAgentBench: A Multi-faceted Benchmark for Evaluating LLM-based Agents in Telecommunications},
  author    = {Lee, Sunwoo and Jang, Daseong and Arya, Dhammiko and Han, Gyoung-eun and Song, Injee and Kim, SaeRom and Kim, Sangjin and Lee, Seojin and Hong, Seokyoung and Sek, Sereimony and Cho, Seung-Mo and Park, Sohee and Yoon, Sungbin and Jang, Wonbeom and Davis, Eric},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  pages     = {1173--1211},
  year      = {2025},
  doi       = {10.18653/v1/2025.emnlp-industry.83}
}
```

## License

- Code and evaluation scripts: [Apache 2.0](LICENSE)
- Localized datasets: CC BY-SA 4.0
