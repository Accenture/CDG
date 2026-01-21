# Attention-Based Token Weighting for Majority Voting

## Core Hypothesis

Tokens that receive high attention from later tokens are "forking points" - critical decision moments where the model chose a reasoning path. These tokens should be weighted differently when evaluating trace quality for majority voting.

## Two-Step Approach

### Step 1: Attention-Based Token Credit

Use **attention received** to assign importance to each token:

```
attention_received[i] = Σ attention[j → i]  for all j > i
```

This measures how much later tokens "looked back" at token i during generation.

**Windowed variant**: Instead of summing all future attention, use a sliding window:
```
attention_received_window[i] = Σ attention[j → i]  for j in [i+1, i+window_size]
```

This captures local importance rather than global, which may be more relevant for identifying immediate decision points.

**Intuition**:
- High attention_received → token was a "forking point" → model made important decision here
- Low attention_received → routine continuation → not critical

### Step 2: Dynamic Weighting with Tanh Function

Apply a smooth, tunable function to convert attention scores to weights:

```
weight = f(attention_received)
```

**Novel tanh-based formula**:
```python
y = x^{20 * ((-tanh(0.8 * x)) + 1.05)}
```

**Behavior**:
- Small x (low attention) → large exponent → moderate weight
- Large x (high attention) → exponent ≈ 1 → weight ≈ x

**Why tanh?**
- Smooth transition between regimes (no hard thresholds)
- Tunable parameters (0.8 controls steepness, 1.05 controls offset)
- Bounded behavior (no exploding weights)

**Alternative: Sigmoid blend**
```python
def smooth_blend(x, midpoint=3.0, steepness=2.0):
    blend = 1 / (1 + exp(-steepness * (x - midpoint)))
    return (1 - blend) * x + blend * (1/x)
```

This smoothly transitions from y=x (reward) to y=1/x (penalize) based on attention value.

## Implementation Levels

### Level 1: Trace-Level Weighting (Current)
Aggregate attention metrics per trace, then weight traces for voting:
```python
trace_weight = f(mean(attention_received))
voted_answer = weighted_majority_vote(answers, trace_weights)
```

### Level 2: Token-Level Weighting (Advanced)
Weight individual tokens, then aggregate to trace score:
```python
token_weights = [f(attention_received[i]) for i in range(seq_len)]
trace_score = aggregate(token_weights)  # mean, sum, or custom
```

This allows finer-grained control over which parts of reasoning matter.

### Level 3: Answer-Region Focused (Future)
Focus on tokens near the answer extraction (e.g., last 2048 tokens):
```python
answer_region_attention = attention_received[-2048:]
trace_weight = f(mean(answer_region_attention))
```

Intuition: The reasoning near the final answer matters most.

## Metrics to Capture

| Metric | Formula | Meaning |
|--------|---------|---------|
| `attention_received[i]` | Σ attn[j,i] for j>i | Forking signal |
| `self_attention[i]` | attn[i,i] | Self-reliance |
| `attention_entropy[i]` | -Σ p log(p) for row i | Decision spread |

All O(seq_len) storage - feasible for long contexts.

## Experiments to Run

1. **Baseline**: Uniform weighting (all traces equal)
2. **Confidence only**: weight = 1/mean(conf)
3. **Attention received**: weight = 1/(1 + mean(attn_received))
4. **Tanh formula on attention**: weight = tanh_formula(mean(attn_received))
5. **Combined**: geometric_mean(conf_weight, attn_weight)
6. **Token-level tanh**: per-token weighting then aggregate

## Technical Requirements

- `enforce_eager=True` in vLLM (disables CUDA graphs)
- ~20-30% slower inference
- May have memory issues at 130K context (attention matrix is large)
- Fallback: Use confidence variance as proxy for forking (already implemented as `entropy_based_weight`)

## Questions to Answer

1. Does attention_received correlate with confidence?
2. Which windowing strategy works best?
3. What's the optimal midpoint for the tanh/sigmoid function?
4. Does token-level weighting outperform trace-level?
5. How much accuracy improvement over uniform voting?

## Code Locations

- Attention capture: `scripts/attention_capture.py`
- Weighting strategies: `analysis/custom_weighting.py`
- Evaluation: `python analysis/custom_weighting.py --results_dir <dir>`
