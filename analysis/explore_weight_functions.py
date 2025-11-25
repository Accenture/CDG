"""
Explore different interpretations of the weight function.

Hypothesis:
- Forking tokens: HIGH attention, LOW confidence → weight = 1/x (downweight)
- Non-forking tokens: LOW attention, HIGH confidence → weight = x (upweight)

Since we don't have attention yet, we work with confidence only.
"""
import math

def print_table(title, func, x_range):
    """Print comparison table for a weight function."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"{'x (conf)':<10} {'weight':<15} {'1/x':<10} {'x':<10}")
    print("-" * 50)
    for x in x_range:
        try:
            w = func(x)
            print(f"{x:<10.2f} {w:<15.4f} {1/x:<10.4f} {x:<10.2f}")
        except:
            print(f"{x:<10.2f} {'ERROR':<15}")


# Original teammate function
def teammate_original(x):
    """y = x^{20*((-tanh(0.8x))+1.05)}"""
    exp = 20 * ((-math.tanh(0.8 * x)) + 1.05)
    return x ** exp


# Alternative interpretation 1: Maybe she meant (1/x) raised to the power
def teammate_alt1(x):
    """y = (1/x)^{20*((-tanh(0.8x))+1.05)}"""
    exp = 20 * ((-math.tanh(0.8 * x)) + 1.05)
    return (1/x) ** exp


# Alternative interpretation 2: Piecewise-like behavior
def piecewise_intuition(x, threshold=3.0):
    """
    Simple piecewise:
    - x < threshold (high conf): weight = x (reward confidence)
    - x >= threshold (low conf): weight = 1/x (penalize uncertainty)
    """
    if x < threshold:
        return x
    else:
        return 1/x


# Alternative 3: Smooth transition using sigmoid
def smooth_transition(x, midpoint=3.0, steepness=2.0):
    """
    Smooth blend between y=x and y=1/x
    Uses sigmoid to transition based on confidence level.
    """
    # sigmoid centered at midpoint
    blend = 1 / (1 + math.exp(-steepness * (x - midpoint)))
    # blend=0 → y=x, blend=1 → y=1/x
    return (1 - blend) * x + blend * (1/x)


# Alternative 4: What if x represents 1/conf (inverted already)?
def teammate_if_x_is_inverted(inv_conf):
    """
    If x = 1/conf (so high x = high confidence):
    Then the original formula makes more sense!
    """
    exp = 20 * ((-math.tanh(0.8 * inv_conf)) + 1.05)
    return inv_conf ** exp


x_range = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]

print_table("Original: y = x^{20*((-tanh(0.8x))+1.05)}", teammate_original, x_range)
print_table("Alt 1: y = (1/x)^{20*((-tanh(0.8x))+1.05)}", teammate_alt1, x_range)
print_table("Alt 2: Piecewise (x if conf>3 else 1/x)", piecewise_intuition, x_range)
print_table("Alt 3: Smooth sigmoid blend between x and 1/x", smooth_transition, x_range)

# If x represents 1/mean_conf (inverted)
print(f"\n{'='*70}")
print("If x = 1/mean_conf (INVERTED confidence):")
print("  High x (e.g., 10) = high confidence (mean_conf=0.1)")
print("  Low x (e.g., 1.5) = low confidence (mean_conf=0.67)")
print(f"{'='*70}")
print_table("Original formula with x=1/conf", teammate_if_x_is_inverted, x_range)


print("""
===============================================================
RECOMMENDATION:
===============================================================

For your hypothesis (without attention data), I suggest starting with:

1. **Simple inverse**: weight = 1 / (mean_conf + ε)
   - Already in your code as 'inverse_conf'
   - High conf → high weight

2. **Variance-aware**: Consider confidence variance
   - High variance suggests forking happened
   - weight = 1 / (mean_conf + α * std_conf)

3. **Ask teammate**: What is x exactly?
   - Raw confidence (neg log prob)?
   - Inverted confidence (1/conf)?
   - Something normalized?

Once you have attention data from VLLM, you can properly implement:
   weight_token = 1/x if attention > threshold else x
""")
