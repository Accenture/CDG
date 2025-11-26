"""
Test script to verify attention capture works with VLLM.

Run this first to check if attention capture is feasible with your model.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_attention_capture.py
"""
import os
import sys

# Set HuggingFace cache
os.environ["HF_HOME"] = "/mnt/dev/model_ckpt/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/dev/model_ckpt/hf_cache"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepconf'))

import torch
import numpy as np
from vllm import LLM, SamplingParams


def find_attention_layers(model):
    """Find all attention-related layers in the model."""
    attention_layers = []
    for name, module in model.named_modules():
        if any(pattern in name.lower() for pattern in ['attn', 'attention']):
            attention_layers.append((name, type(module).__name__))
    return attention_layers


def test_hook_approach(llm):
    """Test if PyTorch hooks can capture attention."""
    print("\n" + "="*60)
    print("Testing PyTorch Hook Approach")
    print("="*60)

    try:
        # Access the model
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        print(f"✓ Model accessed successfully")
        print(f"  Model type: {type(model).__name__}")

        # Find attention layers
        attn_layers = find_attention_layers(model)
        print(f"\n✓ Found {len(attn_layers)} attention-related layers:")
        for name, layer_type in attn_layers[-5:]:  # Show last 5
            print(f"    {name}: {layer_type}")

        # Try to register a hook on the last attention layer
        captured = []

        def capture_hook(module, args, kwargs, output):
            captured.append({
                'output_type': type(output).__name__,
                'output_len': len(output) if isinstance(output, tuple) else 1,
            })
            if isinstance(output, tuple):
                for i, o in enumerate(output):
                    if o is not None and hasattr(o, 'shape'):
                        captured[-1][f'shape_{i}'] = str(o.shape)

        # Find last self_attn layer
        last_attn = None
        last_attn_name = None
        for name, module in model.named_modules():
            if 'self_attn' in name and not any(x in name for x in ['proj', 'norm']):
                last_attn = module
                last_attn_name = name

        if last_attn is None:
            print("✗ Could not find self_attn layer")
            return False

        print(f"\n✓ Registering hook on: {last_attn_name}")
        handle = last_attn.register_forward_hook(capture_hook, with_kwargs=True)

        # Run a short generation
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=10,
        )
        output = llm.generate(["What is 2+2?"], sampling_params)

        handle.remove()

        if captured:
            print(f"\n✓ Hook captured {len(captured)} forward passes")
            print(f"  Last capture: {captured[-1]}")
            return True
        else:
            print("\n✗ Hook did not capture any forward passes")
            print("  This may be due to CUDA graphs. Try enforce_eager=True")
            return False

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_attentions_config(model_name):
    """Test if the model config supports output_attentions."""
    print("\n" + "="*60)
    print("Testing output_attentions Config")
    print("="*60)

    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        if hasattr(config, 'output_attentions'):
            print(f"✓ Config has output_attentions attribute")
            print(f"  Current value: {config.output_attentions}")
        else:
            print("✗ Config does not have output_attentions")

        # Check if model can be loaded with output_attentions=True
        print("\nNote: Even if config supports it, VLLM may not pass attention through")

    except Exception as e:
        print(f"✗ Error loading config: {e}")


def test_logprobs_as_proxy():
    """Show that logprobs can serve as attention proxy."""
    print("\n" + "="*60)
    print("Logprobs as Attention Proxy (Current Fallback)")
    print("="*60)

    print("""
Your current implementation uses confidence variance as an attention proxy:

  high variance in confidence ≈ forking points (high attention)
  low variance in confidence ≈ stable tokens (low attention)

This is already implemented in:
  - entropy_based_weight(): uses variance directly
  - token_level_smooth_blend(): per-token classification

If hook-based capture doesn't work, this proxy approach is still valid!
""")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument('--enforce-eager', action='store_true', help='Disable CUDA graphs')
    args = parser.parse_args()

    print("="*60)
    print("ATTENTION CAPTURE TEST")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Enforce eager: {args.enforce_eager}")

    # Test config first (doesn't need GPU)
    test_output_attentions_config(args.model)

    # Load model
    print("\n" + "="*60)
    print("Loading Model")
    print("="*60)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,  # Critical for hooks!
        max_model_len=4096,  # Small for testing
    )
    print("✓ Model loaded")

    # Test hook approach
    hook_works = test_hook_approach(llm)

    # Show fallback
    test_logprobs_as_proxy()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if hook_works:
        print("✓ PyTorch hooks work! You can capture real attention.")
        print("  Next: Integrate into run_offline_single.py")
    else:
        print("✗ PyTorch hooks don't capture attention with current setup.")
        print("\nOptions:")
        print("  1. Try with --enforce-eager flag")
        print("  2. Use logprobs/confidence as attention proxy (already implemented)")
        print("  3. Patch VLLM source to expose attention weights")

    if not args.enforce_eager:
        print("\n⚠ Recommendation: Re-run with --enforce-eager")
        print("  python scripts/test_attention_capture.py --enforce-eager")


if __name__ == "__main__":
    main()
