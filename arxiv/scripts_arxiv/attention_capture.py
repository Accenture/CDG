"""
Attention capture utilities for VLLM.

Captures attention weights from the last transformer layer during inference.
Requires enforce_eager=True in VLLM to disable CUDA graphs.

Usage:
    from attention_capture import AttentionCapture, DeepThinkLLMWithAttention

    llm = DeepThinkLLMWithAttention(model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    result = llm.deepthink(prompt, ...)
    # result now contains 'attention_received' per trace
"""
import torch
import numpy as np
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class AttentionCapture:
    """
    Captures attention weights from the last transformer layer.

    We compute and store O(seq_len) metrics per generation (not O(seq_len^2)):
    - attention_received[i]: how much later tokens attended to token i (forking signal)
    - self_attention[i]: how much token i attends to itself (self-reliance)
    - attention_entropy[i]: entropy of token i's attention distribution (decision spread)

    These identify "forking points" - tokens that subsequent generation looked back at.
    """

    def __init__(self, model, layer_idx: int = -1):
        """
        Args:
            model: The VLLM model (llm.llm_engine.model_executor.driver_worker.model_runner.model)
            layer_idx: Which layer to capture (-1 = last layer)
        """
        self.model = model
        self.layer_idx = layer_idx
        self.hook_handle = None
        self.captured_attention: Optional[torch.Tensor] = None
        # Aggregated metrics (all O(seq_len))
        self.attention_received: Optional[np.ndarray] = None
        self.self_attention: Optional[np.ndarray] = None
        self.attention_entropy: Optional[np.ndarray] = None

    def _find_last_attention_layer(self):
        """Find the last self-attention layer in the model."""
        attention_layers = []

        for name, module in self.model.named_modules():
            # Common patterns for attention layers
            if any(pattern in name.lower() for pattern in ['self_attn', 'attention', 'attn']):
                # Skip output projections, only want the attention module itself
                if not any(skip in name.lower() for skip in ['out_proj', 'o_proj', 'dense']):
                    attention_layers.append((name, module))

        if not attention_layers:
            raise ValueError("No attention layers found in model")

        # Return the last one (or specified index)
        idx = self.layer_idx if self.layer_idx >= 0 else len(attention_layers) + self.layer_idx
        return attention_layers[idx]

    def _attention_hook(self, module, args, kwargs, output):
        """
        Hook to capture attention weights.

        Note: The exact structure depends on the model architecture.
        For most models, we need to intercept Q, K before the attention computation.
        """
        # Try to get attention weights from output if available
        if isinstance(output, tuple) and len(output) > 1:
            # Some models return (attn_output, attn_weights)
            if output[1] is not None:
                self.captured_attention = output[1].detach()
                return

        # If not in output, try to compute from Q, K stored in module
        if hasattr(module, 'last_attention_weights'):
            self.captured_attention = module.last_attention_weights.detach()

    def _qkv_hook(self, module, args, kwargs, output):
        """
        Alternative hook that captures Q, K, V and computes attention manually.
        This is more reliable across different model architectures.
        """
        # output is typically (query, key, value) or the projected tensors
        # We'll store it and compute attention in post-processing
        if isinstance(output, tuple) and len(output) >= 2:
            q, k = output[0], output[1]
            # Compute attention: softmax(Q @ K^T / sqrt(d_k))
            d_k = q.shape[-1]
            # Handle multi-head: q shape is typically [batch, heads, seq, d_k]
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            self.captured_attention = attn_weights.detach()

    def register_hook(self):
        """Register the attention capture hook."""
        layer_name, layer_module = self._find_last_attention_layer()
        print(f"Registering attention hook on: {layer_name}")

        # Use forward hook with return value capture
        self.hook_handle = layer_module.register_forward_hook(
            self._attention_hook,
            with_kwargs=True
        )
        return self

    def remove_hook(self):
        """Remove the hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def compute_attention_metrics(self) -> Dict[str, np.ndarray]:
        """
        Compute all attention metrics for each token.

        Returns dict with:
        - attention_received[i]: sum of attention from tokens j>i to token i (forking signal)
        - self_attention[i]: attention[i,i] - how much token attends to itself
        - attention_entropy[i]: entropy of row i's attention distribution (decision spread)

        All metrics are O(seq_len).
        """
        if self.captured_attention is None:
            return {}

        # Shape: [batch, heads, seq_len, seq_len] or [heads, seq_len, seq_len]
        attn = self.captured_attention

        # Average over heads
        if attn.dim() == 4:
            attn = attn.mean(dim=1)  # [batch, seq, seq]
            attn = attn[0]  # Take first batch item [seq, seq]
        elif attn.dim() == 3:
            attn = attn.mean(dim=0)  # [seq, seq]

        # attn[i, j] = attention from token i to token j (row i sums to 1)
        seq_len = attn.shape[0]
        attn_np = attn.cpu().numpy()

        # 1. Attention received: how much later tokens look back at token i
        #    = sum over j>i of attn[j, i] (column sum from below diagonal)
        attention_received = np.zeros(seq_len, dtype=np.float32)
        for i in range(seq_len - 1):
            attention_received[i] = attn_np[i+1:, i].sum()

        # 2. Self attention: diagonal elements attn[i, i]
        self_attention = np.diag(attn_np).astype(np.float32)

        # 3. Attention entropy: -sum(p * log(p)) for each row
        #    High entropy = spread attention, Low entropy = focused attention
        attention_entropy = np.zeros(seq_len, dtype=np.float32)
        for i in range(seq_len):
            row = attn_np[i, :i+1]  # Only attend to tokens 0..i (causal)
            # Avoid log(0) by adding small epsilon
            row_safe = np.clip(row, 1e-10, 1.0)
            attention_entropy[i] = -np.sum(row * np.log(row_safe))

        self.attention_received = attention_received
        self.self_attention = self_attention
        self.attention_entropy = attention_entropy

        return {
            'attention_received': attention_received,
            'self_attention': self_attention,
            'attention_entropy': attention_entropy,
        }

    def compute_attention_received(self) -> Optional[np.ndarray]:
        """Legacy method - computes all metrics and returns attention_received."""
        metrics = self.compute_attention_metrics()
        return metrics.get('attention_received')

    def reset(self):
        """Reset captured attention for next generation."""
        self.captured_attention = None
        self.attention_received = None
        self.self_attention = None
        self.attention_entropy = None

    @contextmanager
    def capture(self):
        """Context manager for capturing attention."""
        self.reset()
        self.register_hook()
        try:
            yield self
        finally:
            self.compute_attention_metrics()
            self.remove_hook()

    def get_all_metrics(self) -> Dict[str, Optional[np.ndarray]]:
        """Get all computed attention metrics."""
        return {
            'attention_received': self.attention_received,
            'self_attention': self.self_attention,
            'attention_entropy': self.attention_entropy,
        }


class SimpleAttentionCapture:
    """
    Simpler approach: Inject a custom attention layer that stores weights.

    This modifies the model's forward pass to save attention weights.
    Works better with VLLM's architecture.
    """

    def __init__(self):
        self.attention_weights: List[torch.Tensor] = []
        self.is_capturing = False

    def start_capture(self):
        """Start capturing attention weights."""
        self.attention_weights = []
        self.is_capturing = True

    def stop_capture(self):
        """Stop capturing and return aggregated attention."""
        self.is_capturing = False
        return self.attention_weights

    def record(self, attn_weights: torch.Tensor):
        """Record attention weights (called from modified attention layer)."""
        if self.is_capturing:
            # Only keep attention_received to save memory
            # attn_weights shape: [batch, heads, seq, seq]
            if attn_weights is not None:
                # Average over heads, take last row (attention from newest token)
                avg_attn = attn_weights.mean(dim=1)  # [batch, seq, seq]
                last_token_attn = avg_attn[0, -1, :].detach().cpu().numpy()  # [seq]
                self.attention_weights.append(last_token_attn)

    def get_attention_received(self) -> np.ndarray:
        """
        Aggregate captured attention into attention_received per token.

        For autoregressive generation, we capture attention from each new token.
        attention_received[i] = sum of attention new tokens paid to token i
        """
        if not self.attention_weights:
            return np.array([])

        # Stack all captured attention vectors
        # Each vector is attention from token t to all previous tokens
        seq_len = len(self.attention_weights) + 1  # +1 for prompt tokens
        attention_received = np.zeros(seq_len, dtype=np.float32)

        for t, attn in enumerate(self.attention_weights):
            # attn[i] = attention from token t to token i
            attention_received[:len(attn)] += attn

        return attention_received

    def get_all_metrics(self) -> Dict[str, np.ndarray]:
        """
        Get all attention metrics from captured data.

        Note: SimpleAttentionCapture only captures per-token attention from new tokens,
        so we can only compute attention_received. Self-attention and entropy require
        full attention matrix access (use AttentionCapture instead).
        """
        return {
            'attention_received': self.get_attention_received(),
            'self_attention': None,  # Not available in simple capture
            'attention_entropy': None,  # Not available in simple capture
        }


def patch_attention_layer_for_capture(model, capture: SimpleAttentionCapture):
    """
    Patch the model's attention layers to capture weights.

    This is a more invasive but reliable approach.
    """
    import types

    def make_capturing_forward(original_forward, capture):
        def capturing_forward(self, *args, **kwargs):
            # Call original
            output = original_forward(*args, **kwargs)

            # Try to extract attention weights
            if hasattr(self, '_attn_weights'):
                capture.record(self._attn_weights)

            return output
        return capturing_forward

    # Find and patch attention layers
    for name, module in model.named_modules():
        if 'self_attn' in name.lower() and not any(x in name for x in ['proj', 'dense']):
            original_forward = module.forward
            module.forward = types.MethodType(
                make_capturing_forward(original_forward, capture),
                module
            )
            print(f"Patched: {name}")


# ============= Integration with DeepConf =============

def create_attention_aware_llm(model_name: str, **vllm_kwargs):
    """
    Create a DeepThinkLLM with attention capture enabled.

    IMPORTANT: Sets enforce_eager=True to allow PyTorch hooks to work.
    This has a ~20-30% performance penalty.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepconf'))

    from deepconf import DeepThinkLLM

    # Force eager mode to allow hooks
    vllm_kwargs['enforce_eager'] = True

    llm = DeepThinkLLM(model=model_name, **vllm_kwargs)

    return llm


def extract_attention_from_generation(
    llm,
    prompt: str,
    sampling_params,
    capture_last_n_layers: int = 1
) -> Dict[str, Any]:
    """
    Run generation with attention capture.

    Returns dict with:
        - 'output': The generation output
        - 'attention_received': Array of attention received per token (forking signal)
        - 'self_attention': Array of self-attention per token (self-reliance)
        - 'attention_entropy': Array of attention entropy per token (decision spread)

    Note: This is a simplified version. For production, integrate into DeepThinkLLM.
    """
    # Get the underlying model
    try:
        model = llm.llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        print("Warning: Could not access model for attention capture")
        print("Make sure enforce_eager=True is set")
        return {
            'output': None,
            'attention_received': None,
            'self_attention': None,
            'attention_entropy': None,
        }

    # Set up capture
    capture = AttentionCapture(model, layer_idx=-1)

    with capture.capture():
        output = llm.llm.generate([prompt], [sampling_params])

    return {
        'output': output,
        'attention_received': capture.attention_received,
        'self_attention': capture.self_attention,
        'attention_entropy': capture.attention_entropy,
        'raw_attention': capture.captured_attention,
    }


if __name__ == "__main__":
    # Test the attention capture
    print("Attention capture module loaded successfully")
    print("\nTo use:")
    print("  1. Create LLM with enforce_eager=True")
    print("  2. Use AttentionCapture.capture() context manager")
    print("  3. Access attention_received after generation")
