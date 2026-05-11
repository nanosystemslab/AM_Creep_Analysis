#!/usr/bin/env python3
"""Plot force (N) vs time (min) for every bulk TSSP run."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from plot_style import apply_journal_style, get_stress_style

apply_journal_style()

TSSP_DIR = Path(__file__).resolve().parents[2] / "bulk" / "out" / "tssp"
OUT_DIR = Path(__file__).resolve().parents[2] / "out"

# Collect full-timeline CSVs (exclude _creep / _relaxation)
csvs = sorted(
    [f for f in TSSP_DIR.glob("tssp_*_run*.csv")
     if "_creep" not in f.name and "_relaxation" not in f.name],
    key=lambda p: (float(p.stem.split("_")[1]), int(p.stem.split("run")[1])),
)

stresses = sorted({float(f.stem.split("_")[1]) for f in csvs})
n_stress = len(stresses)
n_runs = max(sum(1 for f in csvs if f.stem.split("_")[1] == str(s) or f.stem.split("_")[1] == str(int(s))) for s in stresses)

fig, axes = plt.subplots(n_stress, n_runs, figsize=(4.5 * n_runs, 2.8 * n_stress),
                         sharex=False, sharey=False, squeeze=False)

for f in csvs:
    parts = f.stem.split("_")
    stress_val = float(parts[1])
    run_idx = int(parts[2].replace("run", "")) - 1
    row = stresses.index(stress_val)

    df = pd.read_csv(f)
    color, _ = get_stress_style(stress_val)
    ax = axes[row][run_idx]
    ax.plot(df["time_sec"] / 60, df["force_N"], color=color, linewidth=0.6)
    target_force = df["force_N"].iloc[len(df) // 4 : len(df) // 2].median()
    ax.axhline(target_force, color="red", ls="--", lw=0.5, alpha=0.5)
    ax.set_title(f"{int(stress_val) if stress_val == int(stress_val) else stress_val} MPa — run {run_idx + 1}",
                 fontsize=9)
    if run_idx == 0:
        ax.set_ylabel("Force (N)")
    if row == n_stress - 1:
        ax.set_xlabel("Time (min)")

# Hide unused subplots
for row in range(n_stress):
    for col in range(n_runs):
        if axes[row][col].lines == []:
            axes[row][col].set_visible(False)

fig.tight_layout()
out_path = OUT_DIR / "bulk_force_vs_time.png"
fig.savefig(out_path, dpi=200)
print(f"Saved → {out_path}")
plt.close()
