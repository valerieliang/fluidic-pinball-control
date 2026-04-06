#!/usr/bin/env python3
"""
Compare simulation outputs between two runs.

Compares:
- Full state vector (q)
- Velocity (u)
- Pressure (p)

Outputs:
- Error metrics (L2, max)
- Visualization plots
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# File paths
# -------------------------
new_dir = "output"
old_dir = "output_original"

files = {
    "fields": (
        f"{new_dir}/pinball_Re80_fields.npz",
        f"{old_dir}/pinball_Re80_fields.npz",
    ),
    "steady": (
        f"{new_dir}/pinball_Re80_steady.npz",
        f"{old_dir}/pinball_Re80_steady.npz",
    ),
}

# -------------------------
# Helper functions
# -------------------------
def compute_error(a, b):
    diff = a - b
    l2 = np.linalg.norm(diff)
    max_err = np.max(np.abs(diff))
    rel = l2 / (np.linalg.norm(b) + 1e-12)
    return l2, max_err, rel, diff


def plot_comparison(a, b, diff, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(a)
    axes[0].set_title("New")

    axes[1].plot(b)
    axes[1].set_title("Original")

    axes[2].plot(diff)
    axes[2].set_title("Difference")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# -------------------------
# Compare steady state (q)
# -------------------------
print("\n=== Comparing steady state (q) ===")

new_q = np.load(files["steady"][0])["q"]
old_q = np.load(files["steady"][1])["q"]

l2, max_err, rel, diff = compute_error(new_q, old_q)

print(f"L2 error:        {l2:.6e}")
print(f"Max error:       {max_err:.6e}")
print(f"Relative error:  {rel:.6e}")

plot_comparison(
    new_q,
    old_q,
    diff,
    "Steady State Vector (q)",
    "compare_q.png",
)

# -------------------------
# Compare fields (u, p)
# -------------------------
print("\n=== Comparing fields (u, p) ===")

new_fields = np.load(files["fields"][0])
old_fields = np.load(files["fields"][1])

for key in ["u", "p"]:
    print(f"\n--- Field: {key} ---")

    new = new_fields[key]
    old = old_fields[key]

    l2, max_err, rel, diff = compute_error(new, old)

    print(f"L2 error:        {l2:.6e}")
    print(f"Max error:       {max_err:.6e}")
    print(f"Relative error:  {rel:.6e}")

    plot_comparison(
        new.flatten(),
        old.flatten(),
        diff.flatten(),
        f"{key} comparison",
        f"compare_{key}.png",
    )

print("\nDone. Plots saved as:")
print("  compare_q.png, compare_u.png, compare_p.png")