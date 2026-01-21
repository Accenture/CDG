"""Analyze the teammate's proposed weight function."""
import math

def teammate_weight(x):
    """y = x^{20*((-tanh(0.8x))+1.05)}"""
    exponent = 20 * ((-math.tanh(0.8 * x)) + 1.05)
    return x ** exponent

print("=" * 60)
print("Analysis of: y = x^{20*((-tanh(0.8x))+1.05)}")
print("=" * 60)

print("\nExponent behavior:")
print(f"  tanh(0.8x) ranges from 0 (at x=0) to ~1 (at x→∞)")
print(f"  So exponent = 20*(-tanh(0.8x) + 1.05)")
print(f"    - At x→0: exponent ≈ 20*(0 + 1.05) = 21")
print(f"    - At x→∞: exponent ≈ 20*(-1 + 1.05) = 1")
print(f"  Therefore: small x → exponent~21, large x → exponent~1")

print("\n" + "-" * 60)
print(f"{'x (conf)':<10} {'exponent':<12} {'y (weight)':<15} {'1/x':<10} {'x':<10}")
print("-" * 60)

for x in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]:
    exp = 20 * ((-math.tanh(0.8 * x)) + 1.05)
    y = x ** exp
    print(f"{x:<10.1f} {exp:<12.2f} {y:<15.4f} {1/x:<10.4f} {x:<10.1f}")

print("\n" + "=" * 60)
print("INTERPRETATION:")
print("=" * 60)
print("""
⚠️  This function does NOT behave like 1/x for small x!

When x is small (1.5-4, high confidence):
  - exponent is LARGE (~5-15)
  - y = x^(large) → y gets VERY LARGE (e.g., 2^10 = 1024)

When x is large (>6, low confidence):
  - exponent approaches 1
  - y ≈ x (linear)

This seems INVERTED from what you want!
  - High confidence (low x) → HUGE weight (exponential blow-up)
  - Low confidence (high x) → weight ≈ x

Maybe she meant: y = x^{-20*((-tanh(0.8x))+1.05)} (negative exponent)?
Or perhaps the x here represents something different?
""")

# Try with negative exponent
print("\n" + "=" * 60)
print("Alternative: y = x^{-20*((-tanh(0.8x))+1.05)} (NEGATIVE exponent)")
print("=" * 60)
print(f"{'x (conf)':<10} {'exponent':<12} {'y (weight)':<15} {'1/x':<10}")
print("-" * 60)

for x in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]:
    exp = -20 * ((-math.tanh(0.8 * x)) + 1.05)  # NEGATIVE
    y = x ** exp
    print(f"{x:<10.1f} {exp:<12.2f} {y:<15.6f} {1/x:<10.4f}")

print("""
With NEGATIVE exponent:
  - Small x (high conf) → large negative exponent → very small y
  - Hmm, that's also not 1/x behavior...
""")
