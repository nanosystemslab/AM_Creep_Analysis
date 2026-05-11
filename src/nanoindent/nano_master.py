#!/usr/bin/env python3
"""
Nanoindentation TSSP Master Curve Construction

Builds a time-stress superposition master curve from nanoindentation creep
compliance data at multiple stress levels (flat-end conical punch, method 103).
Reuses shifting/fitting/plotting functions from tssp_master.py.

Usage:
    python src/nanoindent/nano_master.py \
        --data-dir /Users/ethan/data/t1500/peng \
        --reference-stress 17.5

    python src/nanoindent/nano_master.py \
        --data-dir /Users/ethan/data/t1500/peng \
        --reference-stress 17.5 \
        --probe-type conical --half-angle 59.8 \
        --shift-method rmse
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import calculate_compliance
from src.plot_style import apply_journal_style
from src.tssp_master import (
    plot_unshifted,
    compute_shift_factors,
    compute_shift_factors_gp,
    fit_shift_equation,
    build_master_curve,
    plot_shift_factors,
    export_results,
)


# ───────────────────────────────────────────────────────────────────────────
# DYN file I/O
# ───────────────────────────────────────────────────────────────────────────
def read_dyn_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Read a nanoDMA DYN text file.

    Returns (time_s, disp_nm, load_uN, area_nm2) or None on failure.
    """
    try:
        df = pd.read_csv(path, sep='\t', skiprows=2, encoding='latin1')
        df.columns = df.columns.str.strip().str.replace('Âµ', 'µ')

        time = pd.to_numeric(df['Test Time (s)'], errors='coerce').values
        disp = pd.to_numeric(df['Indent Disp. (nm)'], errors='coerce').values
        load = pd.to_numeric(df['Indent Load (µN)'], errors='coerce').values
        area = pd.to_numeric(df['Contact Area (nm^2)'], errors='coerce').values

        # Drop rows with NaN in critical columns
        valid = np.isfinite(time) & np.isfinite(disp) & np.isfinite(load)
        if valid.sum() < 100:
            return None
        return time[valid], disp[valid], load[valid], area[valid]
    except Exception as e:
        print(f"  Warning: Could not read {path.name}: {e}")
        return None


def extract_hold_period(
    time: np.ndarray,
    disp: np.ndarray,
    load: np.ndarray,
    hold_threshold: float = 0.98,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Extract the constant-load hold period from a nanoindentation test.

    Detects hold as the region where load >= hold_threshold * max_load.
    Resets time to start at 0.

    Returns (time_hold, disp_hold, load_hold) or None if too short.
    """
    max_load = np.nanmax(load)
    threshold = hold_threshold * max_load
    mask = load >= threshold

    if mask.sum() < 100:
        return None

    t_hold = time[mask] - time[mask][0]
    d_hold = disp[mask]  # total indentation depth (J includes elastic component)
    l_hold = load[mask]

    return t_hold, d_hold, l_hold


# ───────────────────────────────────────────────────────────────────────────
# Load and compute compliance across all stress levels
# ───────────────────────────────────────────────────────────────────────────
def load_nano_compliance(
    data_dir: Path,
    method: str = '103',
    probe_type: str = 'flat_punch',
    poisson_ratio: float = 0.43,
    area_m2: float | None = None,
    half_angle_deg: float | None = None,
    tip_radius_nm: float | None = None,
    n_avg_grid: int = 500,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Load nanoindentation DYN files and compute averaged compliance per stress.

    Discovers 103-* subdirectories, parses stress from folder name, reads all
    DYN files per folder, computes compliance via calculate_compliance(),
    then averages across tests on a common log-spaced time grid.

    Returns {stress_MPa: (time_sec, J_shear_1_per_GPa)}.
    """
    stress_dirs = sorted(data_dir.glob(f"{method}-*"))
    if not stress_dirs:
        print(f"Error: No {method}-* directories found in {data_dir}")
        sys.exit(1)

    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for sd in stress_dirs:
        # Parse stress from directory name like "103-17.5" or "103-t1500-17.5-fp"
        # Strategy: try each '-'-separated token as a float, take the first numeric one
        # that isn't the leading method number
        parts = sd.name.split('-')
        stress = None
        for p in parts[1:]:                       # skip "103"
            try:
                stress = float(p)
                break
            except ValueError:
                continue
        if stress is None:
            print(f"  Skipping {sd.name}: cannot parse stress")
            continue

        dyn_files = sorted(sd.glob("*LC_DYN.txt"))
        if not dyn_files:
            print(f"  {stress:.1f} MPa: no DYN files found")
            continue

        print(f"\n  {stress:.1f} MPa: {len(dyn_files)} DYN files")
        run_curves: list[tuple[np.ndarray, np.ndarray]] = []

        for f in dyn_files:
            result = read_dyn_file(f)
            if result is None:
                continue
            time, disp, load, area_arr = result

            hold = extract_hold_period(time, disp, load)
            if hold is None:
                print(f"    {f.name}: hold period too short, skipping")
                continue
            t_hold, d_hold, l_hold = hold

            # Compute shear creep compliance (calculate_compliance already
            # returns J_s = 1/G in 1/GPa for all probe types)
            try:
                J = calculate_compliance(
                    probe_type,
                    d_hold, l_hold,
                    poisson_ratio=poisson_ratio,
                    area_m2=area_m2,
                    half_angle_deg=half_angle_deg,
                    tip_radius_nm=tip_radius_nm,
                )
            except Exception as e:
                print(f"    {f.name}: compliance error: {e}")
                continue

            # Filter invalid
            valid = np.isfinite(J) & (J > 0) & (t_hold > 0)
            if valid.sum() < 50:
                continue

            run_curves.append((t_hold[valid], J[valid]))

        if not run_curves:
            print(f"    No valid runs for {stress:.1f} MPa")
            continue

        # Average across runs: interpolate onto common log-spaced grid
        t_min = max(rc[0][rc[0] > 0][0] for rc in run_curves)
        t_max = min(rc[0][-1] for rc in run_curves)
        if t_max <= t_min:
            print(f"    {stress:.1f} MPa: no overlapping time range")
            continue

        t_grid = np.logspace(np.log10(t_min), np.log10(t_max), n_avg_grid)
        J_stack = []
        for t_r, J_r in run_curves:
            interp_fn = interp1d(t_r, J_r, kind='linear',
                                 bounds_error=False, fill_value=np.nan)
            J_stack.append(interp_fn(t_grid))

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            J_avg = np.nanmean(J_stack, axis=0)
        valid = ~np.isnan(J_avg)
        t_avg = t_grid[valid]
        J_avg = J_avg[valid]

        n_good = len(run_curves)
        print(f"    Averaged {n_good} runs → {len(t_avg)} points, "
              f"J_shear range: {J_avg.min():.4f}–{J_avg.max():.4f} 1/GPa")
        curves[stress] = (t_avg, J_avg)

    print(f"\nLoaded nano compliance for {len(curves)} stress levels: "
          f"{sorted(curves.keys())} MPa")
    return curves


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Nanoindentation TSSP master curve construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/nanoindent/nano_master.py \\
      --data-dir /Users/ethan/data/t1500/peng \\
      --reference-stress 17.5

  python src/nanoindent/nano_master.py \\
      --data-dir /Users/ethan/data/t1500/peng \\
      --reference-stress 17.5 \\
      --probe-type conical --half-angle 59.8
        """,
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing <method>-* stress subdirs")
    parser.add_argument("--method", type=str, default="103",
                        help="Test method prefix for folder discovery (default: 103)")
    parser.add_argument("--reference-stress", type=float, default=17.5,
                        help="Reference stress sigma0 in MPa (default: 17.5)")
    parser.add_argument("--probe-type",
                        choices=["flat_punch", "berkovich", "conical", "frustum"],
                        default="flat_punch",
                        help="Probe geometry (default: flat_punch)")
    parser.add_argument("--poisson-ratio", type=float, default=0.43,
                        help="Poisson ratio (default: 0.43)")
    parser.add_argument("--area", type=float, default=3.62e-10,
                        help="Flat punch contact area in m² (default: 3.62e-10)")
    parser.add_argument("--half-angle", type=float, default=59.8,
                        help="Cone half-angle in degrees (default: 59.8)")
    parser.add_argument("--tip-radius", type=float, default=10728.0,
                        help="Tip radius in nm for frustum (default: 10728)")
    parser.add_argument("--shift-method", choices=["loglevel", "rmse"],
                        default="loglevel",
                        help="Shift estimation method (default: loglevel)")
    parser.add_argument("--use-mastercurves", action="store_true",
                        help="Use GP-based shifting (mastercurves library)")
    parser.add_argument("--vertical-shift", action="store_true",
                        help="Enable vertical shift factor b_sigma")
    parser.add_argument("--min-overlap-frac", type=float, default=0.2,
                        help="Minimum overlap fraction (default: 0.2)")
    parser.add_argument("--min-overlap-points", type=int, default=50,
                        help="Minimum overlap points (default: 50)")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom title for all plots")
    parser.add_argument("--thesis-figures", action="store_true",
                        help="Save figures to out/figures/ch3/ for thesis integration")
    parser.add_argument("--out-dir", type=Path, default=Path("out/nano_master"),
                        help="Output directory (default: out/nano_master)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.thesis_figures:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from plot_style import THESIS_FIGURES_DIR
        out_dir = THESIS_FIGURES_DIR
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma0 = args.reference_stress

    apply_journal_style()

    # --- Step 1: Load nanoindentation compliance ---
    print("\n[Step 1] Loading nanoindentation compliance data...")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Probe: {args.probe_type}, nu={args.poisson_ratio}")

    # Set probe-specific params
    area_m2 = args.area if args.probe_type == 'flat_punch' else None
    half_angle = args.half_angle if args.probe_type in ('conical', 'frustum') else None
    tip_radius = args.tip_radius if args.probe_type == 'frustum' else None

    curves = load_nano_compliance(
        args.data_dir,
        method=args.method,
        probe_type=args.probe_type,
        poisson_ratio=args.poisson_ratio,
        area_m2=area_m2,
        half_angle_deg=half_angle,
        tip_radius_nm=tip_radius,
    )
    if not curves:
        print("Error: No valid compliance curves loaded")
        sys.exit(1)

    # --- Step 2: Plot unshifted compliance ---
    print("\n[Step 2] Plotting unshifted nano compliance...")
    plot_unshifted(
        curves, out_dir,
        ylabel=r'Shear Creep Compliance $J(t)$ (1/GPa)',
        title_suffix=args.title or r'Flat-End Conical Unshifted Shear Creep Compliance $J(t)$ (1/GPa) vs.\ Creep Time (min) (log-log)',
        filename='nano_unshifted_compliance.png',
    )

    # --- Step 3: Compute shift factors ---
    if args.use_mastercurves:
        print(f"\n[Step 3] Computing shift factors via mastercurves GP "
              f"(ref = {sigma0} MPa)...")
        sf_df = compute_shift_factors_gp(
            curves, sigma0, vertical_shift=args.vertical_shift,
        )
    else:
        print(f"\n[Step 3] Computing shift factors (ref = {sigma0} MPa, "
              f"method = {args.shift_method})...")
        sf_df = compute_shift_factors(
            curves, sigma0,
            min_overlap_frac=args.min_overlap_frac,
            min_overlap_points=args.min_overlap_points,
            vertical_shift=args.vertical_shift,
            shift_method=args.shift_method,
        )

    # --- Step 4: Fit shift factor model ---
    print("\n[Step 4] Fitting shift factor model...")
    fit = fit_shift_equation(sf_df)

    # --- Step 5: Build master curve ---
    print("\n[Step 5] Constructing nano master curve...")
    master_df = build_master_curve(
        curves, sf_df, out_dir,
        ylabel=r'Shear Creep Compliance $J(t)$ (1/GPa)',
        title=args.title or (r'Nano TSSP Mastercurve Shear Creep Compliance $J(t)$ (1/GPa)' '\n' r'vs.\ Creep Time (s)'),
        filename='nano_master_curve.png',
        value_col='compliance_1_per_GPa',
    )

    # --- Step 6: Plot shift factors ---
    print("\n[Step 6] Plotting shift factors...")
    plot_shift_factors(
        sf_df, fit, sigma0, out_dir,
        filename='nano_shift_factors.png',
        suptitle=args.title or r"Nano TSSP Horizontal and Vertical Shift Factors vs.\ Stress (MPa)",
    )

    # --- Step 7: Export results ---
    print("\n[Step 7] Exporting results...")
    export_results(
        master_df, sf_df, fit, sigma0, out_dir,
        prefix='nano',
        description='Nanoindentation shear creep compliance J=2(1+nu)/E',
    )

    # --- Summary ---
    print(f"\n{'='*55}")
    print("Nano master curve construction complete!")
    print(f"{'='*55}")
    print(f"  Stress levels: {sorted(curves.keys())}")
    print(f"  Reference: {sigma0} MPa")
    print(f"  Probe: {args.probe_type}")
    if fit['model'] == 'wlf':
        print(f"  Model: WLF — C1 = {fit['C1']:.4f}, C3 = {fit['C3']:.4f}")
    else:
        print(f"  Model: Linear — m = {fit['m']:.4f} MPa^-1")
    print(f"  Output: {out_dir}/")


if __name__ == "__main__":
    main()
