# GPT-OSS-120B Inference Scripts

Run GPT-OSS-120B on math competition datasets with 512 traces per question.

## Requirements

GPT-OSS requires a **special vLLM version** and the **openai-harmony** package. This is different from the standard vLLM used for Qwen/DeepSeek models.

```bash
# Install special vLLM for GPT-OSS
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

# Install harmony package (for prompt formatting)
uv pip install openai-harmony
```

## Model Settings

Per DeepConf paper Table 11:

| Parameter | Value |
|-----------|-------|
| Model | `openai/gpt-oss-120b` |
| Temperature | 1.0 |
| Top-p | 1.0 |
| Top-k | 40 |
| Max tokens | 130,000 |
| Reasoning effort | high |

## Usage

### Run all datasets
```bash
screen -S gptoss120b_inference
cd /path/to/sampling_credit
./scripts/gpt-oss-120b/run_all.sh
# Detach: Ctrl+A, then D
```

### Run individual datasets
```bash
./scripts/gpt-oss-120b/run_aime2025.sh
./scripts/gpt-oss-120b/run_bruno2025.sh
./scripts/gpt-oss-120b/run_hmmt2025.sh
./scripts/gpt-oss-120b/run_aime2024.sh
```

## Dataset Priority Order

1. AIME 2025
2. Bruno 2025
3. HMMT Feb 2025
4. AIME 2024

## Key Differences from Qwen/DeepSeek

1. **Special vLLM version**: GPT-OSS requires `vllm==0.10.1+gptoss` instead of standard vLLM
2. **Harmony format**: Uses `openai-harmony` for prompt formatting instead of HuggingFace tokenizers
3. **Different inference script**: Uses `run_gpt_oss_batch.py` instead of `run_vllm_batch.py`
4. **Higher temperature**: Uses temp=1.0 (vs 0.6 for Qwen/DeepSeek)

## Hardware

- GPT-OSS-120B fits on a single 80GB GPU (H100/MI300X) thanks to MXFP4 quantization
- For batch inference with 512 traces, recommend using tensor parallelism across multiple GPUs

## Output

Results are saved to `${OUTPUT_BASE}/<run_id>/`:
- `qid<N>_<timestamp>.pkl` - Full trace data
- `qid<N>_<timestamp>.json` - Human-readable summary (if `--save_json` flag used)
