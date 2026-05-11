#!/usr/bin/env python3
"""
Plot bulk compliance vs time for FULL 24-hour dataset.

This script reads the preprocessed bulk compliance CSV files and creates
publication-quality plots showing the complete time-dependent behavior.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_DIR = _PROJECT_ROOT / "out"
OUTPUT_DIR.mkdir(exist_ok=True)

# Data files (processed compliance data)
DATA_FILES = {
    17.5: str(_PROJECT_ROOT / "out/bulk_17.5mpa.csv"),
    20.0: str(_PROJECT_ROOT / "out/bulk_20mpa.csv"),
    22.5: str(_PROJECT_ROOT / "out/bulk_22.5mpa.csv"),
    25.0: str(_PROJECT_ROOT / "out/bulk_25mpa.csv"),
    30.0: str(_PROJECT_ROOT / "out/bulk_30mpa.csv"),
}

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================
def load_compliance_data(file_path):
    """Load preprocessed compliance data from CSV."""
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Extract time and compliance
    time_s = df['Time (s)'].values
    compliance = df['Creep Compliance (1/GPa)'].values

    # Remove any invalid data
    valid = np.isfinite(time_s) & np.isfinite(compliance) & (compliance > 0)
    time_s = time_s[valid]
    compliance = compliance[valid]

    return time_s, compliance


def plot_all_stress_levels_linear():
    """Plot all stress levels on linear scale."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color scheme - viridis colormap
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DATA_FILES)))

    for i, (stress_mpa, file_path) in enumerate(sorted(DATA_FILES.items())):
        if not Path(file_path).exists():
            print(f"⚠ File not found: {file_path}")
            continue

        # Load data
        time_s, compliance = load_compliance_data(file_path)
        time_hr = time_s / 3600  # Convert to hours

        # Plot
        ax.plot(time_hr, compliance, '-', color=colors[i], linewidth=2.5,
                label=f'{stress_mpa:.1f} MPa', alpha=0.8)

        print(f"✓ Loaded {stress_mpa:.1f} MPa: {len(time_s)} points, "
              f"max time = {time_hr[-1]:.1f} hr")

    # Formatting
    ax.set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Creep Compliance J(t) (1/GPa)', fontsize=14, fontweight='bold')
    ax.set_title('Bulk Creep Compliance - Full 24 Hour Test\nASTM D2990 Compression',
                 fontsize=16, fontweight='bold')

    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', fontsize=12, title='Applied Stress', framealpha=0.9)
    ax.set_xlim(left=0)

    # Make tick labels larger
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()

    return fig


def plot_all_stress_levels_log():
    """Plot all stress levels on log-log scale."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color scheme - viridis colormap
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DATA_FILES)))

    for i, (stress_mpa, file_path) in enumerate(sorted(DATA_FILES.items())):
        if not Path(file_path).exists():
            print(f"⚠ File not found: {file_path}")
            continue

        # Load data
        time_s, compliance = load_compliance_data(file_path)

        # Remove t=0 for log scale
        mask = time_s > 0
        time_s = time_s[mask]
        compliance = compliance[mask]

        # Plot
        ax.loglog(time_s, compliance, '-', color=colors[i], linewidth=2.5,
                  label=f'{stress_mpa:.1f} MPa', alpha=0.8)

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Creep Compliance J(t) (1/GPa)', fontsize=14, fontweight='bold')
    ax.set_title('Bulk Creep Compliance - Full 24 Hour Test (Log-Log)\nASTM D2990 Compression',
                 fontsize=16, fontweight='bold')

    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', fontsize=12, title='Applied Stress', framealpha=0.9)

    # Make tick labels larger
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()

    return fig


def plot_individual_stress_level(stress_mpa, file_path):
    """Plot a single stress level on both linear and log scales."""
    if not Path(file_path).exists():
        print(f"⚠ File not found: {file_path}")
        return None, None

    # Load data
    time_s, compliance = load_compliance_data(file_path)
    time_hr = time_s / 3600

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Linear scale
    ax1.plot(time_hr, compliance, '-', color='#440154', linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Creep Compliance J(t) (1/GPa)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Bulk Creep - {stress_mpa:.1f} MPa (Linear Scale)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    ax1.tick_params(axis='both', labelsize=12)

    # Log scale
    mask = time_s > 0
    ax2.loglog(time_s[mask], compliance[mask], '-', color='#440154',
               linewidth=2.5, alpha=0.8)
    ax2.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Creep Compliance J(t) (1/GPa)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Bulk Creep - {stress_mpa:.1f} MPa (Log-Log Scale)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linestyle='--')
    ax2.tick_params(axis='both', labelsize=12)

    plt.tight_layout()

    return fig, (ax1, ax2)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("="*80)
    print("BULK COMPLIANCE PLOTTING - FULL 24 HOUR DATA")
    print("="*80)

    # Check which files exist
    available_files = {k: v for k, v in DATA_FILES.items() if Path(v).exists()}

    if not available_files:
        print("\n❌ No data files found!")
        print("\nExpected files:")
        for stress, path in DATA_FILES.items():
            print(f"  - {path}")
        return

    print(f"\n✓ Found {len(available_files)} stress levels:")
    for stress in sorted(available_files.keys()):
        print(f"  - {stress:.1f} MPa")

    # Plot 1: All stress levels - Linear scale
    print("\n⏳ Creating overlay plot (linear scale)...")
    fig_linear = plot_all_stress_levels_linear()
    output_linear = OUTPUT_DIR / "bulk_24hr_overlay_linear.png"
    fig_linear.savefig(output_linear, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_linear}")

    # Plot 2: All stress levels - Log scale
    print("\n⏳ Creating overlay plot (log scale)...")
    fig_log = plot_all_stress_levels_log()
    output_log = OUTPUT_DIR / "bulk_24hr_overlay_log.png"
    fig_log.savefig(output_log, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_log}")

    # Plot 3: Individual plots for each stress level
    print("\n⏳ Creating individual stress level plots...")
    for stress_mpa, file_path in sorted(available_files.items()):
        fig_indiv, _ = plot_individual_stress_level(stress_mpa, file_path)
        if fig_indiv is not None:
            output_indiv = OUTPUT_DIR / f"bulk_24hr_{stress_mpa:.1f}mpa.png"
            fig_indiv.savefig(output_indiv, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_indiv}")
            plt.close(fig_indiv)

    print("\n" + "="*80)
    print("✅ COMPLETE - All plots saved to out/")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - bulk_24hr_overlay_linear.png")
    print(f"  - bulk_24hr_overlay_log.png")
    for stress in sorted(available_files.keys()):
        print(f"  - bulk_24hr_{stress:.1f}mpa.png")

    # Close all figures to free memory
    plt.close('all')


if __name__ == "__main__":
    main()
