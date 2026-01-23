# Inference Scripts

Unified inference runner for all models and datasets.

## Quick Start

```bash
# Run from project root
./scripts/inference/run.sh --model <model> --dataset <dataset>
```

## Usage Examples

```bash
# Single model, single dataset
./scripts/inference/run.sh --model deepseek8b --dataset aime2025

# Single model, all datasets
./scripts/inference/run.sh --model qwen32b --dataset all

# GPT-OSS models
./scripts/inference/run.sh --model gpt-oss-120b --dataset brumo2025
```

## Available Models

| Model | Config Prefix | Description |
|-------|---------------|-------------|
| `deepseek8b` | DEEPSEEK8B_ | DeepSeek-R1-0528-Qwen3-8B |
| `deepseek-r1-llama-70b` | DEEPSEEK_R1_LLAMA_70B_ | DeepSeek-R1-Distill-Llama-70B |
| `qwen32b` | QWEN32B_ | Qwen3-32B (with thinking) |
| `qwen25-32b` | QWEN25_32B_ | Qwen2.5-32B base |
| `qwq32b` | QWQ32B_ | QwQ-32B reasoning |
| `gemma3-27b` | GEMMA3_27B_ | Gemma 3-27B-IT (non-reasoning baseline) |
| `gpt-oss-20b` | GPTOSS20B_ | GPT-OSS-20B |
| `gpt-oss-120b` | GPTOSS120B_ | GPT-OSS-120B |

## Available Datasets

| Dataset | Config Variable |
|---------|-----------------|
| `aime2025` | DATASET_AIME_2025 |
| `aime2024` | DATASET_AIME_2024 |
| `hmmt2025` | DATASET_HMMT_2025 |
| `brumo2025` | DATASET_BRUMO_2025 |
| `all` | Runs all 4 datasets |

## Configuration

All model/dataset settings are in `scripts/config.sh`.

## GPT-OSS Requirements

For GPT-OSS models, install the special vLLM version:

```bash
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

uv pip install openai-harmony
```

## Output

Results are saved to `${OUTPUT_BASE}/<rid>/`:
- `qid{N}_{timestamp}.pkl` - Raw inference results per question
- `qid{N}_{timestamp}.json` - Human-readable summary with traces
