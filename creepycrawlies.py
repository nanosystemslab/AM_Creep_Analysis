#!/usr/bin/env python3
"""
Creepy Crawlies - Main Nanoindentation Creep Analysis Script
=============================================================

This is the main entry point for nanoindentation creep data analysis.
Uses modular imports from the consolidated codebase.

Key Features:
    - Load and process nanoindentation data files
    - Calculate creep compliance, strain, and rates
    - Average data across multiple tests
    - Generate publication-quality plots
    - Support for multiple probe geometries
    - Automatic method detection from folder names
    - Caching for fast repeated analysis

Quick Usage:
    # Plot displacement vs time for Berkovich tests
    python creepycrawlies.py -d nano/103/ -i out/ -x 0 -y 1

    # Plot creep compliance (calculated on-the-fly)
    python creepycrawlies.py -d nano/103/ -i out/ -x 0 -y compliance --calculate_compliance

    # Overlay multiple stress levels
    python creepycrawlies.py -d nano/103-17.5 nano/103-20 nano/103-22.5 -i out/ -x 0 -y 1 --overlay

Author: Creep Analysis Framework
Date: 2024-2025
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import shlex

# Add src/ to path so intra-src bare imports (e.g. data_utils → analysis_functions) resolve
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Import our modular functions from src/
try:
    from src.data_utils import (
        find_files, load_and_clean_data, resolve_columns,
        average_data_with_bins, average_data_no_bins, group_files_by_max_load
    )
    from src.analysis_functions import compute_strain_recovery, fit_linear_sections
    from src.plotting_functions import plot_data_with_stdev, plot_individual_files
    from src.plot_style import apply_journal_style, get_display_label, get_folder_style, get_style_by_index
except ImportError as e:
    print(f"[ERROR] Failed to import required modules: {e}")
    print("[INFO] Make sure you're running from the AM_Creep_Analysis directory")
    print("[INFO] and that src/ directory contains all required modules.")
    sys.exit(1)

__version__ = "2.0.0"


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        format="%(levelname)s - %(name)s - %(funcName)s @%(lineno)d: %(message)s",
        level=logging.INFO,
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process and plot nanoindentation creep data.",
        epilog="""
Examples:
  # Basic displacement plot
  python creepycrawlies.py -d nano/103/ -i out/ -x 0 -y 1

  # Calculate and plot compliance
  python creepycrawlies.py -d nano/103/ -i out/ -x 0 -y compliance --calculate_compliance

  # Overlay multiple folders
  python creepycrawlies.py -d nano/103-17.5 nano/103-20 -i out/ -x 0 -y 1 --overlay

  # Plot individual files (no averaging)
  python creepycrawlies.py -d nano/103/ -i out/ -x 0 -y 1 --plot_individual
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('-d', '--directories', nargs='+', required=True,
                       help='Input folder(s) containing data files')
    parser.add_argument('-i', '--output', required=True,
                       help='Output folder for plots and results')
    parser.add_argument('--filename', type=str, default=None,
                       help='Override output filename (stem, no extension). '
                            'Saves directly to --output dir without subdirectories.')
    parser.add_argument('--thesis-figures', action='store_true', dest='thesis_figures',
                       help='Save figures to out/figures/ch3/ for thesis integration')
    parser.add_argument('-x', '--x_column', required=True,
                       help='X-axis column title or index (e.g., "0" for Test Time)')
    parser.add_argument('-y', '--y_columns', nargs='+', required=True,
                       help='Y-axis column titles or indices (e.g., "1" for displacement)')

    # Optional columns
    parser.add_argument('-r', '--r_columns', nargs='*', default=[],
                       help='Right Y-axis column titles or indices')

    # File type selection
    parser.add_argument('--type', choices=["DYN", "AVG", "QS", "DC"], default="DYN",
                       help="File type to load (DYN/AVG/QS for load-controlled, DC for displacement-controlled)")
    parser.add_argument('--type-y', choices=["DYN", "AVG", "QS", "DC"], default=None,
                       help="File type for Y-axis (overrides --type)")
    parser.add_argument('--type-r', choices=["DYN", "AVG", "QS", "DC"], default=None,
                       help="File type for R-axis (overrides --type)")

    # Data processing options
    parser.add_argument('--bins', type=int, default=20,
                       help='Number of bins for averaging (default: 20)')
    parser.add_argument('--adaptive-bins', action='store_true',
                       help='Use adaptive binning (equal points per bin) instead of equal intervals')
    parser.add_argument('--nobin', action='store_true',
                       help="Disable binning; interpolate each run onto the first run's X grid and average across runs")
    parser.add_argument('--nosem', action='store_true',
                       help='Use standard deviation for error bars instead of SEM')
    parser.add_argument('--sem-by-run', action='store_true',
                       help='Compute SEM across runs by binning each run before averaging')

    # Plotting options
    parser.add_argument('--log_x', action='store_true', help='Log scale X-axis')
    parser.add_argument('--log_y', action='store_true', help='Log scale Y-axis')
    parser.add_argument('--log_r', action='store_true', help='Log scale right Y-axis')
    parser.add_argument('--plot_show', action='store_true', help='Show plots interactively')
    parser.add_argument('--journal_style', action='store_true',
                       help='Use journal-quality plot styling')
    parser.add_argument('--title', type=str, default=None,
                       help='Custom plot title (overrides auto-generated title)')
    parser.add_argument('--title-pad', type=float, default=None,
                       help='Title padding above plot (default: 10)')
    parser.add_argument('--legend-y', type=float, default=None,
                       help='Legend vertical position below plot (default: -0.15, more negative = lower)')

    # Analysis options
    parser.add_argument('--calculate_elastic', action='store_true',
                       help='Calculate elastic modulus')
    parser.add_argument('--calculate_compliance', action='store_true',
                       help='Calculate creep compliance J(t) = strain/stress')
    parser.add_argument('--calculate_stress', action='store_true',
                       help='Calculate stress σ = Load/Contact_Area (GPa)')
    parser.add_argument('--calculate_creep_strain', action='store_true',
                       help='Calculate creep strain during hold period')
    parser.add_argument('--correct_drift', type=float,
                       help='Drift rate (nm/s) to subtract from displacement')
    parser.add_argument('--probe_type',
                       choices=['berkovich', 'conical_60', 'conical_90', 'flat_punch'],
                       default=None,
                       help='Probe type for compliance calculation (overrides method_config). '
                            'Default: from method_config.json or berkovich')
    parser.add_argument('--poisson_ratio', type=float, default=0.43,
                       help='Poisson ratio for compliance calculation (default: 0.43)')
    parser.add_argument('--flat_punch_area', type=float, default=3.6157e-10,
                       help='Flat punch contact area in m² (default: 3.6157e-10 for 21.456 µm diameter punch)')
    parser.add_argument('--folder_probe', nargs='+', action='append',
                       metavar='PROBE_OR_FOLDER',
                       help='Assign probe type to folders: --folder_probe conical_60 folder1 folder2 '
                            '(repeatable for different probes). Auto-detects from folder name if not set.')

    # Fitting and analysis
    parser.add_argument('--fit_sections', action='store_true',
                       help='Fit linear sections of the plot')
    parser.add_argument('--fit_manual', nargs='*', type=float,
                       help='Manual fit ranges: xmin1 xmax1 xmin2 xmax2 ...')
    parser.add_argument('--strain_recovery', nargs=2, type=float,
                       metavar=('T_START', 'T_END'),
                       help='Compute strain recovery between T_START and T_END seconds')
    parser.add_argument('--fit_model', type=str, default=None,
                       metavar='MODEL',
                       help='Fit a creep model to averaged data and overlay. '
                            'Models: logarithmic, gen-kelvin-{1,3,5,7,9}, burgers, '
                            'power-law, prony-{3,5}, peng-{2,3,vp,3vp}')
    parser.add_argument('--fit_stats', action='store_true', default=False,
                       help='Show R² and model parameters on the plot (requires --fit_model)')

    # Rate calculations
    parser.add_argument('--compute_creep', nargs=2, type=float,
                       metavar=('T_START', 'T_END'),
                       help='Calculate creep rate (1/h * dh/dt) between T_START and T_END seconds')
    parser.add_argument('--compute_strain_rate', nargs=2, type=float,
                       metavar=('T_START', 'T_END'),
                       help='Calculate strain rate (dε/dt) between T_START and T_END seconds')

    # Time range filtering
    parser.add_argument('--plot_range', nargs=2, type=float,
                       metavar=('T_START', 'T_END'),
                       help='Only plot data between T_START and T_END seconds')
    parser.add_argument('--folder_range', nargs=3, action='append',
                       metavar=('FOLDER', 'T_START', 'T_END'),
                       help='Per-folder time range: --folder_range nano/103/ 5 30 '
                            '(can be repeated for multiple folders)')

    # Plotting modes
    parser.add_argument('--plot_individual', action='store_true',
                       help='Plot each test file separately (no averaging)')
    parser.add_argument('--plot_individual_y', action='store_true',
                       help='Plot individual runs for Y-axis only (R-axis remains averaged)')
    parser.add_argument('--plot_individual_r', action='store_true',
                       help='Plot individual runs for R-axis only (Y-axis remains averaged)')
    parser.add_argument('--overlay', action='store_true',
                       help='Overlay multiple folders on one plot (default: separate plots)')

    # Frequency sweep mode
    parser.add_argument('--fs', action='store_true',
                       help='Frequency sweep mode: plot discrete points without lines')

    # Grouping options
    parser.add_argument('--group_by_load', nargs='?', type=float, const=100,
                       metavar='TOLERANCE',
                       help='Group files by maximum load with tolerance in µN (default: 100)')

    # Hold period configuration
    parser.add_argument('--hold_start', type=float, default=None,
                       help='Time (s) when hold period starts (overrides auto-detection)')
    parser.add_argument('--hold_end', type=float, default=None,
                       help='Time (s) when hold period ends (overrides auto-detection)')
    parser.add_argument('--no-method-config', action='store_true',
                       help='Disable automatic method detection from folder names')

    # Reference lines
    parser.add_argument('--datasheet', type=float, default=None,
                       metavar='VALUE',
                       help='Draw a red dashed reference line at VALUE on the Y-axis '
                            '(e.g. --datasheet 0.18 for datasheet hardness)')
    parser.add_argument('--shallow', type=float, nargs='?', const=500.0, default=None,
                       metavar='DEPTH_NM',
                       help='Draw a vertical red dashed line on the X-axis marking the '
                            'shallow-depth cutoff in nm (default: 500 nm)')

    # Display options
    parser.add_argument('--no-watermark', action='store_true',
                       help='Hide CLI watermark text at bottom of plots')
    parser.add_argument('--png', action='store_true',
                       help='Also save PNG at 300 DPI alongside PDF')

    # Cache management
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching (reprocess all files)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached data before processing')

    return parser.parse_args()


def _get_folder_range(args, group_key):
    """Return per-folder range (t_start, t_end) or None.

    Priority:
      1. Explicit --folder_range from CLI
      2. Auto-trim from method_config hold periods (unless --no-method-config)
    """
    if group_key is not None:
        name = Path(group_key).name.rstrip('/')
        if name in args._folder_ranges:
            return args._folder_ranges[name]
        # Auto-trim to method_config hold period if available
        if not args.no_method_config and hasattr(args, '_method_config'):
            config = args._method_config
            if config:
                from src.data_utils import extract_method_from_path
                # Build a fake file path so extract_method_from_path can parse the folder name
                fake_path = Path(name) / "dummy.txt"
                method = extract_method_from_path(fake_path, config)
                if method and method in config.get("methods", {}):
                    mc = config["methods"][method]
                    hs, he = mc.get("hold_start"), mc.get("hold_end")
                    if hs is not None and he is not None:
                        return (hs, he)
    return None


def _apply_range(df, x_col, time_col, args, group_key):
    """Filter df by range.

    --folder_range always filters on time_col (column 0).
    --plot_range filters on x_col.
    folder_range takes priority.
    """
    fr = _get_folder_range(args, group_key)
    if fr is not None and time_col in df.columns:
        t_start, t_end = fr
        return df[(df[time_col] >= t_start) & (df[time_col] <= t_end)]
    if args.plot_range:
        t_start, t_end = args.plot_range
        return df[(df[x_col] >= t_start) & (df[x_col] <= t_end)]
    return df


def _get_plot_range_for_bins(args, group_key):
    """Return the effective plot_range tuple for binning functions.

    For folder_range, returns the range (binning functions always use x_col internally).
    For global plot_range, returns it directly. Otherwise None.
    Note: folder_range is time-based, so this is only correct when x_col == time_col.
    For folder_range with non-time x_col, the df is pre-filtered by _apply_range before binning.
    """
    fr = _get_folder_range(args, group_key)
    if fr is not None:
        return None  # pre-filter handles it; don't double-filter in binning
    return args.plot_range if args.plot_range else None


def _resolve_probe_for_folder(args, group_key):
    """Resolve probe type for a folder.

    Priority: folder_probe > probe_type > auto-detect from name > None.
    """
    if group_key is None:
        return args.probe_type

    name = Path(group_key).name.rstrip('/')
    name_lower = name.lower()

    # 1. Explicit per-folder probe
    if hasattr(args, '_folder_probes') and args._folder_probes:
        for key, probe in args._folder_probes.items():
            if key.lower() == name_lower:
                return probe

    # 2. Global CLI override
    if args.probe_type:
        return args.probe_type

    # 3. Auto-detect from folder name
    segments = name_lower.replace('-', '_').split('_')
    if 'flat' in segments or 'flatpunch' in segments or 'fp' in segments:
        return 'flat_punch'
    if 'berk' in segments or 'berkovich' in segments:
        return 'berkovich'
    if 'con' in segments or 'conical' in segments:
        return 'conical_60'

    # 4. Let load_and_clean_data handle via method_config
    return None


def _fit_model_to_data(model_name, x_data, y_data, folder_label, color):
    """Fit a creep model to averaged data and return fit curve info."""
    from src.models import get_model

    model = get_model(model_name)

    # Filter valid data
    valid = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0)
    t = x_data[valid]
    J = y_data[valid]

    if len(t) < 5:
        print(f"[WARN] Not enough data points for fitting {folder_label} ({len(t)} points)")
        return None

    try:
        result = model.fit(t, J)
        stats = model.get_statistics()
        params = model.get_parameters()

        # Generate smooth curve
        t_fine = np.linspace(t.min(), t.max(), 200)
        J_fine = model.predict(t_fine)

        print(f"[FIT] {folder_label}: {model.name}")
        print(f"       R² = {stats['R_squared']:.6f}, RMSE = {stats['RMSE']:.4e}")
        for pname, pinfo in params.items():
            stderr_str = f" ± {pinfo['stderr']:.2e}" if pinfo['stderr'] else ""
            print(f"       {pname} = {pinfo['value']:.4e}{stderr_str} {pinfo['units']}")

        return {
            't': t_fine,
            'J': J_fine,
            'label': f"$R^2$={stats['R_squared']:.4f}",
            'color': color,
            'R2': stats['R_squared'],
            'RMSE': stats['RMSE'],
            'params': params,
            'model_name': model.name,
        }
    except Exception as e:
        print(f"[WARN] Fit failed for {folder_label}: {e}")
        return None


def main():
    """Main function."""
    setup_logging()
    args = parse_arguments()
    args.cli_command = " ".join(shlex.quote(arg) for arg in sys.argv)

    # Build per-folder range lookup: normalized folder name → (t_start, t_end)
    # Accepts any order: --folder_range FOLDER T0 T1  or  --folder_range T0 T1 FOLDER
    _folder_ranges = {}
    if args.folder_range:
        for triplet in args.folder_range:
            nums = []
            folder = None
            for val in triplet:
                try:
                    nums.append(float(val))
                except ValueError:
                    folder = val
            if folder is None or len(nums) != 2:
                print(f"[ERROR] --folder_range expects one folder path and two numbers, got: {triplet}")
                sys.exit(1)
            key = Path(folder).name.rstrip('/')
            _folder_ranges[key] = (min(nums), max(nums))
    args._folder_ranges = _folder_ranges

    # Build per-folder probe lookup: normalized folder name → probe type
    _folder_probes = {}
    if args.folder_probe:
        valid_probes = ['berkovich', 'conical_60', 'conical_90', 'flat_punch']
        for group in args.folder_probe:
            if len(group) < 2:
                print(f"[ERROR] --folder_probe requires PROBE_TYPE and at least one FOLDER, got: {group}")
                sys.exit(1)
            probe = group[0]
            if probe not in valid_probes:
                print(f"[ERROR] Unknown probe type '{probe}'. Valid: {valid_probes}")
                sys.exit(1)
            for folder in group[1:]:
                key = Path(folder).name.rstrip('/')
                _folder_probes[key] = probe
    args._folder_probes = _folder_probes

    # Load method config for auto-trim in _get_folder_range
    if not args.no_method_config:
        from src.data_utils import load_method_config
        args._method_config = load_method_config()
    else:
        args._method_config = None

    print(f"[INFO] Creepy Crawlies v{__version__}")
    print(f"[INFO] Nanoindentation Creep Analysis Framework")
    print("="*60)
    if _folder_ranges:
        for k, (s, e) in _folder_ranges.items():
            print(f"[INFO] Folder range: {k} → [{s}, {e}] s")
    if _folder_probes:
        for k, p in _folder_probes.items():
            print(f"[INFO] Folder probe: {k} → {p}")

    # Handle cache clearing
    if args.clear_cache:
        from src.data_utils import clear_old_caches
        print("[INFO] Clearing all caches...")
        clear_old_caches(keep_recent=0)
        if not any([args.directories]):  # If only clearing cache
            return

    # Redirect to thesis figures directory if requested
    if args.thesis_figures:
        from src.plot_style import THESIS_FIGURES_DIR
        args.output = str(THESIS_FIGURES_DIR)

    # Create base output directory
    base_output = Path(args.output)

    # Determine cache usage
    use_cache = not args.no_cache

    # Determine file types for each axis
    type_y = args.type_y if args.type_y else args.type
    type_r = args.type_r if args.type_r else args.type

    # Check if we need to load two different file types
    dual_file_types = (type_y != type_r) and args.r_columns

    if dual_file_types:
        print(f"[INFO] Dual file type mode: Y-axis from {type_y} files, R-axis from {type_r} files")

    # Find files for Y-axis
    files_y = find_files(args.directories, type_y)
    if not files_y:
        logging.error(f"[ERROR] No {type_y} files found in: {args.directories}")
        return
    print(f"[INFO] Found {len(files_y)} {type_y} files for Y-axis")

    # Find files for R-axis if different type specified
    if dual_file_types:
        files_r = find_files(args.directories, type_r)
        if not files_r:
            logging.error(f"[ERROR] No {type_r} files found for R-axis.")
            return
        print(f"[INFO] Found {len(files_r)} {type_r} files for R-axis")
    else:
        files_r = files_y  # Use same files for both axes

    # Initialize rate lists
    load_and_clean_data.avg_creep_rates = []
    load_and_clean_data.avg_strain_rates = []

    # Load a sample to get column names for resolving indices
    sample_files = [files_y[0]] if files_y else []
    if sample_files:
        sample_dfs = load_and_clean_data(
            sample_files,
            correct_drift=args.correct_drift,
            calculate_elastic=args.calculate_elastic,
            compute_creep=args.compute_creep,
            compute_strain_rate=args.compute_strain_rate,
            calculate_compliance=args.calculate_compliance,
            calculate_stress=getattr(args, 'calculate_stress', False),
            calculate_creep_strain=args.calculate_creep_strain,
            creep_hold_start=getattr(args, 'hold_start', None),
            creep_hold_end=getattr(args, 'hold_end', None),
            use_method_config=not args.no_method_config,
            use_cache=use_cache,
            probe_type=args.probe_type,
            poisson_ratio=args.poisson_ratio,
            flat_punch_area_m2=args.flat_punch_area
        )
        if sample_dfs:
            sample_df = sample_dfs[0]
        else:
            logging.error("[ERROR] Cannot load sample data for column resolution.")
            return
    else:
        logging.error("[ERROR] No files to process.")
        return

    # Resolve column references (convert indices to names)
    x_col = resolve_columns([args.x_column], sample_df.columns)[0]
    y_cols = resolve_columns(args.y_columns, sample_df.columns)
    r_cols = resolve_columns(args.r_columns, sample_df.columns)
    time_col = resolve_columns(["0"], sample_df.columns)[0]  # column 0 = Test Time

    print(f"[INFO] X column: {x_col}")
    print(f"[INFO] Y columns: {y_cols}")
    if r_cols:
        print(f"[INFO] R columns: {r_cols}")

    # Check if any columns were not found
    for cols, name in [(y_cols, "Y"), (r_cols, "R")]:
        for col in cols:
            if col not in sample_df.columns:
                logging.warning(f"[WARN] {name} column '{col}' not found in data")

    # Get all files for processing
    all_files = find_files(args.directories, args.type)

    # Determine grouping strategy
    if args.group_by_load is not None:
        load_tolerance = args.group_by_load  # Use the provided tolerance value
        print(f"[INFO] Grouping {len(all_files)} files by maximum load (tolerance: ±{load_tolerance:.0f} µN)")
        files_by_group = group_files_by_max_load(all_files, load_tolerance=load_tolerance)
        group_labels = [f"{load:.0f}µN" for load in files_by_group.keys()]
    else:
        # Group by folder (original behavior)
        files_by_group = {}
        for file_path in all_files:
            folder_key = str(file_path.parent)
            if folder_key not in files_by_group:
                files_by_group[folder_key] = []
            files_by_group[folder_key].append(file_path)
        group_labels = [Path(folder_path).name for folder_path in files_by_group.keys()]

    # Process files based on plotting mode
    if args.plot_individual:
        print(f"[INFO] Individual plotting mode: processing {len(all_files)} files separately")

        if args.overlay:
            print("[INFO] Overlay mode: combining individual files across groups")

            if args.filename:
                output_dir = base_output
            else:
                folder_names = [Path(key).name if isinstance(key, str) else f"{key:.0f}µN"
                              for key in files_by_group.keys()]
                overlay_name = "overlay_" + "_".join(folder_names[:3])
                if len(folder_names) > 3:
                    overlay_name += f"_and_{len(folder_names)-3}_more"
                output_dir = base_output / overlay_name
            output_dir.mkdir(parents=True, exist_ok=True)

            individual_datasets = []
            individual_labels = []

            for group_key, group_files in zip(files_by_group.keys(), files_by_group.values()):
                if args.group_by_load is not None:
                    group_name = f"{group_key:.0f}µN"
                else:
                    group_name = get_display_label(Path(group_key).name)

                for file_path in group_files:
                    file_name = file_path.stem
                    print(f"[INFO] Processing individual file: {file_name}")

                    file_dfs = load_and_clean_data(
                        [file_path],
                        correct_drift=args.correct_drift,
                        calculate_elastic=args.calculate_elastic,
                        compute_creep=args.compute_creep,
                        compute_strain_rate=args.compute_strain_rate,
                        calculate_compliance=args.calculate_compliance,
                        calculate_creep_strain=args.calculate_creep_strain,
                        creep_hold_start=getattr(args, 'hold_start', None),
                        creep_hold_end=getattr(args, 'hold_end', None),
                        use_method_config=not args.no_method_config,
                        use_cache=use_cache,
                        probe_type=_resolve_probe_for_folder(args, group_key),
                        poisson_ratio=args.poisson_ratio,
                        flat_punch_area_m2=args.flat_punch_area
                    )

                    if file_dfs and len(file_dfs) > 0:
                        df = _apply_range(file_dfs[0], x_col, time_col, args, group_key)

                        if not df.empty:
                            individual_datasets.append(df)
                            individual_labels.append(f"{group_name}_{file_name}")
                            print(f"[INFO] Successfully processed: {file_name}")
                        else:
                            print(f"[WARN] No data after filtering for: {file_name}")

            if not individual_datasets:
                logging.error("[ERROR] No individual files processed successfully.")
                return

            print(f"[INFO] Plotting {len(individual_datasets)} individual files (overlay)")

            original_output = args.output
            args.output = str(output_dir)

            plot_individual_files(
                individual_datasets,
                x_col,
                y_cols,
                r_cols,
                args,
                individual_labels
            )

            args.output = original_output

        else:
            print("[INFO] Separate mode: creating individual plots per group")

            for group_key, group_files in zip(files_by_group.keys(), files_by_group.values()):
                if args.group_by_load is not None:
                    group_name = f"{group_key:.0f}µN"
                else:
                    group_name = get_display_label(Path(group_key).name)

                output_dir = base_output / group_name
                output_dir.mkdir(parents=True, exist_ok=True)

                print(f"[INFO] Processing group: {group_name} ({len(group_files)} files)")
                print(f"[INFO] Output directory: {output_dir}")

                individual_datasets = []
                individual_labels = []

                for file_path in group_files:
                    file_name = file_path.stem
                    print(f"[INFO] Processing individual file: {file_name}")

                    file_dfs = load_and_clean_data(
                        [file_path],
                        correct_drift=args.correct_drift,
                        calculate_elastic=args.calculate_elastic,
                        compute_creep=args.compute_creep,
                        compute_strain_rate=args.compute_strain_rate,
                        calculate_compliance=args.calculate_compliance,
                        calculate_creep_strain=args.calculate_creep_strain,
                        creep_hold_start=getattr(args, 'hold_start', None),
                        creep_hold_end=getattr(args, 'hold_end', None),
                        use_method_config=not args.no_method_config,
                        use_cache=use_cache,
                        probe_type=_resolve_probe_for_folder(args, group_key),
                        poisson_ratio=args.poisson_ratio,
                        flat_punch_area_m2=args.flat_punch_area
                    )

                    if file_dfs and len(file_dfs) > 0:
                        df = _apply_range(file_dfs[0], x_col, time_col, args, group_key)

                        if not df.empty:
                            individual_datasets.append(df)
                            individual_labels.append(file_name)
                            print(f"[INFO] Successfully processed: {file_name}")
                        else:
                            print(f"[WARN] No data after filtering for: {file_name}")

                if not individual_datasets:
                    print(f"[WARN] No individual files processed for group: {group_name}")
                    continue

                print(f"[INFO] Plotting {len(individual_datasets)} individual files for {group_name}")

                original_output = args.output
                args.output = str(output_dir)

                plot_individual_files(
                    individual_datasets,
                    x_col,
                    y_cols,
                    r_cols,
                    args,
                    individual_labels
                )

                args.output = original_output

    else:
        print(f"[INFO] Averaging mode: processing {len(all_files)} files by group")

        # Determine if we're doing overlay or separate outputs
        if args.overlay:
            # Overlay mode: all folders on one plot
            print(f"[INFO] Overlay mode: combining all folders into one plot")

            # Create overlay output directory
            if args.filename:
                output_dir = base_output
            else:
                folder_names = [Path(key).name if isinstance(key, str) else f"{key:.0f}µN"
                              for key in files_by_group.keys()]
                overlay_name = "overlay_" + "_".join(folder_names[:3])  # Limit to first 3
                if len(folder_names) > 3:
                    overlay_name += f"_and_{len(folder_names)-3}_more"
                output_dir = base_output / overlay_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process all groups and collect data
            folder_datasets = []  # For averaged data
            folder_labels = []    # Display labels for legend
            folder_keys = []      # Raw folder names for style lookup

            # For individual data per axis
            y_individual_datasets = [] if args.plot_individual_y else None
            y_individual_labels = [] if args.plot_individual_y else None
            r_individual_datasets = [] if args.plot_individual_r else None
            r_individual_labels = [] if args.plot_individual_r else None

            for group_key, group_files in zip(files_by_group.keys(), files_by_group.values()):
                if args.group_by_load is not None:
                    raw_name = f"{group_key:.0f}µN"
                    group_name = raw_name
                else:
                    raw_name = Path(group_key).name
                    group_name = get_display_label(raw_name)

                print(f"[INFO] Processing group: {group_name} ({len(group_files)} files)")

                folder_dfs = load_and_clean_data(
                    group_files,
                    correct_drift=args.correct_drift,
                    calculate_elastic=args.calculate_elastic,
                    compute_creep=args.compute_creep,
                    compute_strain_rate=args.compute_strain_rate,
                    calculate_compliance=args.calculate_compliance,
                    calculate_creep_strain=args.calculate_creep_strain,
                    creep_hold_start=getattr(args, 'hold_start', None),
                    creep_hold_end=getattr(args, 'hold_end', None),
                    use_method_config=not args.no_method_config,
                    use_cache=use_cache,
                    probe_type=_resolve_probe_for_folder(args, group_key),
                    poisson_ratio=args.poisson_ratio,
                    flat_punch_area_m2=args.flat_punch_area
                )

                if folder_dfs:
                    # Pre-filter by folder_range (time-based) if set
                    _fr = _get_folder_range(args, group_key)
                    if _fr:
                        folder_dfs = [
                            d[(d[time_col] >= _fr[0]) & (d[time_col] <= _fr[1])]
                            for d in folder_dfs
                        ]
                        folder_dfs = [d for d in folder_dfs if not d.empty]

                    # Handle individual Y-axis data
                    if args.plot_individual_y:
                        for file_idx, df in enumerate(folder_dfs):
                            if not _fr and args.plot_range:
                                t_start, t_end = args.plot_range
                                df = df[(df[x_col] >= t_start) & (df[x_col] <= t_end)]
                            if not df.empty:
                                y_individual_datasets.append(df)
                                y_individual_labels.append(f"{group_name}_{file_idx}")

                    # Handle individual R-axis data
                    if args.plot_individual_r:
                        for file_idx, df in enumerate(folder_dfs):
                            if not _fr and args.plot_range:
                                t_start, t_end = args.plot_range
                                df = df[(df[x_col] >= t_start) & (df[x_col] <= t_end)]
                            if not df.empty:
                                r_individual_datasets.append(df)
                                r_individual_labels.append(f"{group_name}_{file_idx}")

                    _bin_range = _get_plot_range_for_bins(args, group_key)

                    # Handle averaged data (for axes not in individual mode)
                    if not args.plot_individual_y or not args.plot_individual_r:
                        if args.nobin:
                            averaged_folder_df = average_data_no_bins(
                                folder_dfs,
                                x_col,
                                y_cols,
                                r_cols,
                                plot_range=_bin_range
                            )
                        else:
                            data_for_bins = folder_dfs if args.sem_by_run else pd.concat(folder_dfs, ignore_index=True)
                            averaged_folder_df = average_data_with_bins(
                                data_for_bins,
                                x_col,
                                y_cols,
                                r_cols,
                                bins=args.bins,
                                plot_range=_bin_range,
                                adaptive=args.adaptive_bins,
                                sem_by_run=args.sem_by_run,
                                log_x=getattr(args, 'log_x', False)
                            )

                        if not averaged_folder_df.empty:
                            folder_datasets.append(averaged_folder_df)
                            folder_labels.append(group_name)
                            folder_keys.append(raw_name)
                            print(f"[INFO] Successfully processed group: {group_name}")
                        else:
                            print(f"[WARN] No data after binning for group: {group_name}")

            # Check if we have any data to plot
            has_data = False
            if not args.plot_individual_y and not args.plot_individual_r:
                has_data = len(folder_datasets) > 0
            else:
                has_data = (y_individual_datasets and len(y_individual_datasets) > 0) or \
                          (r_individual_datasets and len(r_individual_datasets) > 0) or \
                          len(folder_datasets) > 0

            if not has_data:
                logging.error("[ERROR] No data after processing all groups.")
                return

            print(f"[INFO] Plotting groups as overlay")
            if args.plot_individual_y:
                print(f"[INFO] Y-axis: {len(y_individual_datasets)} individual traces")
            if args.plot_individual_r:
                print(f"[INFO] R-axis: {len(r_individual_datasets)} individual traces")

            # Fit models if requested
            if args.fit_model and folder_datasets:
                args._fit_curves = []
                total_cols = len(y_cols) + len(r_cols)
                for folder_index, (ds, flabel, fkey) in enumerate(zip(folder_datasets, folder_labels, folder_keys)):
                    for col_index, y_col in enumerate(y_cols):
                        mean_col = y_col + '_mean'
                        if mean_col not in ds.columns:
                            continue
                        x_data = ds[x_col].values
                        y_data = ds[mean_col].values
                        style_index = folder_index * total_cols + col_index
                        f_color, _ = get_style_by_index(style_index)
                        fit_info = _fit_model_to_data(args.fit_model, x_data, y_data, flabel, f_color)
                        if fit_info:
                            args._fit_curves.append(fit_info)

            # Temporarily override output directory for plotting
            original_output = args.output
            args.output = str(output_dir)

            plot_data_with_stdev(
                folder_datasets if folder_datasets else [pd.DataFrame()],
                x_col,
                y_cols,
                r_cols,
                args,
                folder_labels if folder_labels else [],
                y_individual_data=y_individual_datasets,
                r_individual_data=r_individual_datasets,
                y_individual_labels=y_individual_labels,
                r_individual_labels=r_individual_labels,
                folder_keys=folder_keys if folder_keys else None
            )

            args.output = original_output

        else:
            # Separate mode: each folder gets its own output directory
            print(f"[INFO] Separate mode: creating individual plots per folder")

            for group_key, group_files in zip(files_by_group.keys(), files_by_group.values()):
                if args.group_by_load is not None:
                    raw_name = f"{group_key:.0f}µN"
                    group_name = raw_name
                else:
                    raw_name = Path(group_key).name
                    group_name = get_display_label(raw_name)

                # Create output directory using raw name (filesystem-safe)
                output_dir = base_output / raw_name
                output_dir.mkdir(parents=True, exist_ok=True)

                print(f"[INFO] Processing group: {group_name} ({len(group_files)} files)")
                print(f"[INFO] Output directory: {output_dir}")

                folder_dfs = load_and_clean_data(
                    group_files,
                    correct_drift=args.correct_drift,
                    calculate_elastic=args.calculate_elastic,
                    compute_creep=args.compute_creep,
                    compute_strain_rate=args.compute_strain_rate,
                    calculate_compliance=args.calculate_compliance,
                    calculate_creep_strain=args.calculate_creep_strain,
                    creep_hold_start=getattr(args, 'hold_start', None),
                    creep_hold_end=getattr(args, 'hold_end', None),
                    use_method_config=not args.no_method_config,
                    use_cache=use_cache,
                    probe_type=_resolve_probe_for_folder(args, group_key),
                    poisson_ratio=args.poisson_ratio,
                    flat_punch_area_m2=args.flat_punch_area
                )

                if folder_dfs:
                    # Pre-filter by folder_range (time-based) if set
                    _fr = _get_folder_range(args, group_key)
                    if _fr:
                        folder_dfs = [
                            d[(d[time_col] >= _fr[0]) & (d[time_col] <= _fr[1])]
                            for d in folder_dfs
                        ]
                        folder_dfs = [d for d in folder_dfs if not d.empty]

                    # Prepare data for individual axes if requested
                    y_individual_datasets = [] if args.plot_individual_y else None
                    y_individual_labels = [] if args.plot_individual_y else None
                    r_individual_datasets = [] if args.plot_individual_r else None
                    r_individual_labels = [] if args.plot_individual_r else None

                    # Handle individual Y-axis data
                    if args.plot_individual_y:
                        for file_idx, df in enumerate(folder_dfs):
                            if not _fr and args.plot_range:
                                t_start, t_end = args.plot_range
                                df = df[(df[x_col] >= t_start) & (df[x_col] <= t_end)]
                            if not df.empty:
                                y_individual_datasets.append(df)
                                y_individual_labels.append(f"{group_name}_{file_idx}")

                    # Handle individual R-axis data
                    if args.plot_individual_r:
                        for file_idx, df in enumerate(folder_dfs):
                            if not _fr and args.plot_range:
                                t_start, t_end = args.plot_range
                                df = df[(df[x_col] >= t_start) & (df[x_col] <= t_end)]
                            if not df.empty:
                                r_individual_datasets.append(df)
                                r_individual_labels.append(f"{group_name}_{file_idx}")

                    _bin_range = _get_plot_range_for_bins(args, group_key)

                    # Handle averaged data (for axes not in individual mode)
                    averaged_folder_df = pd.DataFrame()
                    if not args.plot_individual_y or not args.plot_individual_r:
                        if args.nobin:
                            averaged_folder_df = average_data_no_bins(
                                folder_dfs,
                                x_col,
                                y_cols,
                                r_cols,
                                plot_range=_bin_range
                            )
                        else:
                            data_for_bins = folder_dfs if args.sem_by_run else pd.concat(folder_dfs, ignore_index=True)
                            averaged_folder_df = average_data_with_bins(
                                data_for_bins,
                                x_col,
                                y_cols,
                                r_cols,
                                bins=args.bins,
                                plot_range=_bin_range,
                                adaptive=args.adaptive_bins,
                                sem_by_run=args.sem_by_run,
                                log_x=getattr(args, 'log_x', False)
                            )

                    if not averaged_folder_df.empty or y_individual_datasets or r_individual_datasets:
                        print(f"[INFO] Successfully processed group: {group_name}")

                        # Fit models if requested
                        if args.fit_model and not averaged_folder_df.empty:
                            args._fit_curves = []
                            for y_col in y_cols:
                                mean_col = y_col + '_mean'
                                if mean_col not in averaged_folder_df.columns:
                                    continue
                                x_data = averaged_folder_df[x_col].values
                                y_data = averaged_folder_df[mean_col].values
                                f_color, _ = get_folder_style(raw_name)
                                fit_info = _fit_model_to_data(args.fit_model, x_data, y_data, group_name, f_color)
                                if fit_info:
                                    args._fit_curves.append(fit_info)

                        # Plot this single folder
                        # Temporarily override output directory for plotting
                        original_output = args.output
                        args.output = str(output_dir)

                        plot_data_with_stdev(
                            [averaged_folder_df] if not averaged_folder_df.empty else [pd.DataFrame()],
                            x_col,
                            y_cols,
                            r_cols,
                            args,
                            [group_name],
                            y_individual_data=y_individual_datasets,
                            r_individual_data=r_individual_datasets,
                            y_individual_labels=y_individual_labels,
                            r_individual_labels=r_individual_labels,
                            folder_keys=[raw_name]
                        )

                        args.output = original_output
                    else:
                        print(f"[WARN] No data after processing for group: {group_name}")

    print("="*60)
    print("[INFO] Processing complete!")
    print(f"[INFO] Output saved to: {base_output}")


if __name__ == "__main__":
    main()
