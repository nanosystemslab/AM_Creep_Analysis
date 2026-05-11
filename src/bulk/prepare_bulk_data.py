#!/usr/bin/env python3
"""
Prepare bulk creep data for Peng-Peng stress-lock model fitting.

Converts raw ASTM D2990 CSV files to simple time-compliance format.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Specimen configuration (CYLINDRICAL - from your plot)
SPECIMEN_CONFIG = {
    'diameter': 12.7,   # mm
    'length': 25.4,     # mm
    'area': np.pi * (12.7/2)**2  # mm² = 126.68 mm²
}

# Material properties
POISSON_RATIO = 0.37  # Tough1500 (typical thermoset range 0.3-0.4)


def process_bulk_file(input_file, output_file, target_stress_mpa=None):
    """
    Process bulk creep CSV file to extract time and compliance.

    Args:
        input_file: Path to raw CSV file
        output_file: Path for output CSV
        target_stress_mpa: Expected stress level (MPa) for verification

    Returns:
        Average stress (MPa), compliance at final time (1/GPa)
    """
    print(f"\nProcessing: {input_file}")

    # Read CSV (skip header rows)
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Find header row with column names
    header_row = next((i for i, line in enumerate(lines)
                      if 'Time' in line or 'time' in line.lower()), 0)

    # Read data starting from header
    df = pd.read_csv(input_file, skiprows=header_row)

    # Clean column names (remove quotes)
    df.columns = df.columns.str.strip('"')

    # Get columns
    time_col = 'Time'
    force_col = 'Force'
    stroke_col = 'Stroke'

    # Convert to numeric (coerce errors to NaN)
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df[force_col] = pd.to_numeric(df[force_col], errors='coerce')
    df[stroke_col] = pd.to_numeric(df[stroke_col], errors='coerce')

    # Drop rows with NaN
    df = df.dropna(subset=[time_col, force_col, stroke_col])

    # Calculate stress and strain
    df['Stress (MPa)'] = df[force_col] / SPECIMEN_CONFIG['area']
    df['Strain'] = df[stroke_col] / SPECIMEN_CONFIG['length']

    # Detect hold period start: when stress reaches 95% of maximum and stabilizes
    max_stress = df['Stress (MPa)'].max()
    threshold_stress = 0.95 * max_stress

    # Find first point where stress exceeds threshold
    hold_start_idx = df[df['Stress (MPa)'] >= threshold_stress].index[0]

    # Extract hold period only
    df_hold = df.loc[hold_start_idx:].copy()

    # Reset time to start from 0 at hold period
    hold_start_time = df_hold[time_col].iloc[0]
    df_hold['Time (s)'] = (df_hold[time_col] - hold_start_time) * 3600.0

    # Calculate shear creep compliance using Equation 1 from Peng et al.:
    # J(t) = 2(1 + ν) × ε(t) / σ₀
    # This converts tensile compliance to shear compliance
    nu = POISSON_RATIO
    df_hold['Tensile Compliance (1/MPa)'] = df_hold['Strain'] / df_hold['Stress (MPa)']
    df_hold['Shear Compliance (1/MPa)'] = 2 * (1 + nu) * df_hold['Tensile Compliance (1/MPa)']

    # Convert to 1/GPa (multiply by 1000)
    df_hold['Creep Compliance (1/GPa)'] = df_hold['Shear Compliance (1/MPa)'] * 1000.0

    # Remove NaN and inf values
    df_hold = df_hold.replace([np.inf, -np.inf], np.nan).dropna()

    # Use hold period data
    df = df_hold

    # Calculate statistics
    avg_stress = df['Stress (MPa)'].mean()
    stress_std = df['Stress (MPa)'].std()
    final_compliance_shear = df['Creep Compliance (1/GPa)'].iloc[-1]
    final_compliance_tensile = df['Tensile Compliance (1/MPa)'].iloc[-1] * 1000
    hold_duration_hrs = df['Time (s)'].max() / 3600.0
    n_points = len(df)

    print(f"  Hold period detected: stress ≥ {threshold_stress:.2f} MPa (95% of max)")
    print(f"  Hold data points: {n_points}")
    print(f"  Hold duration: {hold_duration_hrs:.2f} hours ({df['Time (s)'].max():.1f} s)")
    print(f"  Hold stress: {avg_stress:.2f} ± {stress_std:.2f} MPa")
    print(f"  Poisson's ratio: ν = {POISSON_RATIO}")
    print(f"  Final tensile compliance: {final_compliance_tensile:.4f} 1/GPa")
    print(f"  Final shear compliance (Eq 1): {final_compliance_shear:.4f} 1/GPa")
    print(f"  Conversion factor: 2(1+ν) = {2*(1+POISSON_RATIO):.3f}")

    # Verify stress level
    if target_stress_mpa:
        stress_error = abs(avg_stress - target_stress_mpa) / target_stress_mpa * 100
        print(f"  Target stress: {target_stress_mpa:.1f} MPa (error: {stress_error:.1f}%)")
        if stress_error > 10:
            print(f"  WARNING: Stress differs from target by >{stress_error:.1f}%!")

    # Save processed data (only time and compliance)
    output_df = df[['Time (s)', 'Creep Compliance (1/GPa)']].copy()
    output_df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    return avg_stress, final_compliance_shear


def main():
    if len(sys.argv) < 3:
        print("Usage: python prepare_bulk_data.py <input_csv> <output_csv> [target_stress_mpa]")
        print("\nExample:")
        print("  python prepare_bulk_data.py bulk/bulkdata/astm/20MPa-1.csv output/bulk_20mpa.csv 20")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_stress = float(sys.argv[3]) if len(sys.argv) > 3 else None

    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Process file
    avg_stress, final_comp = process_bulk_file(input_file, output_file, target_stress)

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Specimen: Cylindrical Ø{SPECIMEN_CONFIG['diameter']}×{SPECIMEN_CONFIG['length']} mm")
    print(f"Average stress: {avg_stress:.2f} MPa = {avg_stress/1000:.4f} GPa")
    print(f"Final shear compliance (Eq 1): {final_comp:.4f} 1/GPa")
    print(f"Output file: {output_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
