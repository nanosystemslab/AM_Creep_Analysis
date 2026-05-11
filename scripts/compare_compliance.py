#!/usr/bin/env python3
"""Compare bulk and nano shear creep compliance on the same plot.

Supports:
  - TSSP averaged CSVs (--bulk)
  - ASTM raw CSVs (--astm)
  - Nano master curve CSVs (--nano)
  - Raw nano DYN directories (--nano-dir), auto-computes compliance per stress
  - Time range filtering (--time-range)
  - Peng stress-lock fitting with custom thresholds (--peng-locks, --yield-stress)
  - Generic model fitting (--fit-model)
  - Running average smoothing (--smooth-window)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textwrap
from src.plot_style import apply_journal_style, get_style_by_index


def _wrap_title(text, fig, ax, fontsize=None):
    """Wrap title text so it never exceeds the axes width."""
    if fontsize is None:
        fontsize = plt.rcParams.get('axes.titlesize', 12)
        if isinstance(fontsize, str):
            fontsize = 12
    char_width_in = 0.6 * fontsize / 72.0
    bbox = ax.get_position()
    axes_width_in = fig.get_size_inches()[0] * bbox.width
    max_chars = max(20, int(axes_width_in / char_width_in))
    lines = textwrap.wrap(text, width=max_chars)
    return "\n".join(r"\textbf{" + line + "}" for line in lines)


# ── Constants ─────────────────────────────────────────────────────────────
SPECIMEN_AREA = np.pi / 4 * 12.7**2  # mm² (12.7 mm dia cylinder = 126.68 mm²)
SPECIMEN_LENGTH = 25.4          # mm
POISSON_RATIO = 0.43


# ── ASTM raw CSV loading ─────────────────────────────────────────────────
def load_astm(csv_path):
    """Load a raw ASTM D2990 CSV → (time_s, J_shear 1/GPa, stress_MPa)."""
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.iloc[1:].reset_index(drop=True)
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Force'] = pd.to_numeric(df['Force'], errors='coerce')
    df['Stroke'] = pd.to_numeric(df['Stroke'], errors='coerce')
    df = df.dropna()

    time_s = df['Time'].values * 3600
    stress_MPa = df['Force'].values / SPECIMEN_AREA
    strain = df['Stroke'].values / SPECIMEN_LENGTH

    max_stress = stress_MPa.max()
    hold_idx = np.where(stress_MPa >= 0.95 * max_stress)[0]
    if len(hold_idx) == 0:
        return None
    i0 = hold_idx[0]

    t = time_s[i0:] - time_s[i0]
    s = stress_MPa[i0:]
    eps = strain[i0:]
    stress_GPa = s / 1000
    J = np.where(stress_GPa > 0, 2 * (1 + POISSON_RATIO) * eps / stress_GPa, np.nan)

    # Average stress only over the hold period (>= 95% max), not relaxation
    hold_stress = stress_MPa[hold_idx]

    valid = np.isfinite(J) & (J > 0)
    return t[valid], J[valid], np.nanmean(hold_stress)


def load_astm_folder(folder, time_range=None):
    """Load and average all CSVs in an ASTM stress-level folder."""
    csvs = sorted(Path(folder).glob("*.csv"))
    if not csvs:
        print(f"  [WARN] No CSVs in {folder}")
        return None

    all_t, all_J, stresses = [], [], []
    for p in csvs:
        result = load_astm(p)
        if result is None:
            continue
        t, J, sigma = result
        if time_range:
            mask = (t >= time_range[0]) & (t <= time_range[1])
            t, J = t[mask], J[mask]
        if len(t) == 0:
            continue
        all_t.append(t)
        all_J.append(J)
        stresses.append(sigma)

    if not all_t:
        return None

    t_min = max(arr[arr > 0][0] for arr in all_t)
    t_max = min(arr[-1] for arr in all_t)
    if t_max <= t_min:
        return None
    t_grid = np.logspace(np.log10(max(t_min, 0.01)), np.log10(t_max), 300)

    J_stack = []
    for t_r, J_r in zip(all_t, all_J):
        fn = interp1d(t_r, J_r, bounds_error=False, fill_value=np.nan)
        J_stack.append(fn(t_grid))
    J_avg = np.nanmean(J_stack, axis=0)
    valid = ~np.isnan(J_avg)
    return t_grid[valid], J_avg[valid], np.mean(stresses)


# ── TSSP / nano CSV loading ──────────────────────────────────────────────
def load_bulk(csv_path, time_range=None):
    """Load bulk TSSP averaged creep compliance CSV."""
    df = pd.read_csv(csv_path)
    time_col = next((c for c in df.columns if 'time' in c.lower() and 'creep' in c.lower()), None)
    if time_col is None:
        time_col = next((c for c in df.columns if 'time' in c.lower()), df.columns[0])
    j_col = next((c for c in df.columns if 'J_mean' in c or 'compliance' in c.lower()), None)
    if j_col is None:
        j_col = df.columns[1]

    t = df[time_col].values
    J = df[j_col].values
    if np.nanmax(J) < 0.1:
        J = J * 1000

    # Prefer target_stress_MPa (nominal) over instantaneous stress_MPa
    if 'target_stress_MPa' in df.columns:
        stress = df['target_stress_MPa'].iloc[0]
    else:
        stress_col = next((c for c in df.columns if 'stress' in c.lower() or 'sigma' in c.lower()), None)
        stress = df[stress_col].iloc[0] if stress_col else None

    if time_range:
        mask = (t >= time_range[0]) & (t <= time_range[1])
        t, J = t[mask], J[mask]
    return t, J, stress


def load_nano(csv_path, time_range=None):
    """Load nano master curve CSV."""
    df = pd.read_csv(csv_path)
    time_col = next((c for c in df.columns if 'time' in c.lower()), df.columns[0])
    j_col = next((c for c in df.columns if 'compliance' in c.lower()), df.columns[1])
    t = df[time_col].values
    J = df[j_col].values
    if time_range:
        mask = (t >= time_range[0]) & (t <= time_range[1])
        t, J = t[mask], J[mask]
    return t, J


# ── Raw nano DYN loading (reuses nano_master.py logic) ───────────────────
def load_nano_dir(data_dir, time_range=None, probe_type='flat_punch',
                  poisson_ratio=0.43, area_m2=3.62e-10):
    """Load raw nano DYN files per stress subfolder → {stress_MPa: (t, J)}.

    Discovers 103-* subdirectories, parses stress from folder name,
    reads DYN files, computes hold-period shear compliance, averages runs.
    """
    from src.utils import calculate_compliance

    data_dir = Path(data_dir)
    stress_dirs = sorted(data_dir.glob("103-*"))
    if not stress_dirs:
        print(f"  [ERROR] No 103-* directories found in {data_dir}")
        return {}

    curves = {}
    for sd in stress_dirs:
        # Parse stress from folder name (handles "103-17.5" and "103-t1500-17.5-fp")
        parts = sd.name.split('-')
        stress = None
        for p in parts[1:]:
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
            continue

        print(f"  {stress:.1f} MPa: {len(dyn_files)} DYN files")
        run_curves = []

        for f in dyn_files:
            try:
                df = pd.read_csv(f, sep='\t', skiprows=2, encoding='latin1')
                df.columns = df.columns.str.strip().str.replace('Âµ', 'µ')
                time = pd.to_numeric(df['Test Time (s)'], errors='coerce').values
                disp = pd.to_numeric(df['Indent Disp. (nm)'], errors='coerce').values
                load = pd.to_numeric(df['Indent Load (µN)'], errors='coerce').values
            except Exception:
                continue

            valid = np.isfinite(time) & np.isfinite(disp) & np.isfinite(load)
            if valid.sum() < 100:
                continue
            time, disp, load = time[valid], disp[valid], load[valid]

            # Extract hold period (load >= 98% max)
            max_load = np.nanmax(load)
            mask = load >= 0.98 * max_load
            if mask.sum() < 100:
                continue
            t_hold = time[mask] - time[mask][0]
            d_hold = disp[mask]
            l_hold = load[mask]

            # Compute shear compliance (calculate_compliance already returns
            # shear compliance J_s = 1/G in 1/GPa for all probe types)
            try:
                J = calculate_compliance(
                    probe_type, d_hold, l_hold,
                    poisson_ratio=poisson_ratio, area_m2=area_m2,
                )
            except Exception:
                continue

            ok = np.isfinite(J) & (J > 0) & (t_hold > 0)
            if ok.sum() < 50:
                continue
            run_curves.append((t_hold[ok], J[ok]))

        if not run_curves:
            print(f"    No valid runs for {stress:.1f} MPa")
            continue

        # Average on log-spaced grid
        t_min = max(rc[0][rc[0] > 0][0] for rc in run_curves)
        t_max = min(rc[0][-1] for rc in run_curves)
        if t_max <= t_min:
            continue
        t_grid = np.logspace(np.log10(t_min), np.log10(t_max), 500)

        import warnings
        J_stack = []
        for t_r, J_r in run_curves:
            fn = interp1d(t_r, J_r, bounds_error=False, fill_value=np.nan)
            J_stack.append(fn(t_grid))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            J_avg = np.nanmean(J_stack, axis=0)
        ok = ~np.isnan(J_avg)
        t_avg, J_avg = t_grid[ok], J_avg[ok]

        # Apply time range
        if time_range:
            mask = (t_avg >= time_range[0]) & (t_avg <= time_range[1])
            t_avg, J_avg = t_avg[mask], J_avg[mask]

        if len(t_avg) > 0:
            print(f"    Averaged {len(run_curves)} runs → {len(t_avg)} pts, "
                  f"J range: {J_avg.min():.4f}–{J_avg.max():.4f} 1/GPa")
            curves[stress] = (t_avg, J_avg)

    return curves


def load_nano_single(folder, time_range=None, probe_type='flat_punch',
                     poisson_ratio=0.43, area_m2=3.62e-10, label=None):
    """Load DYN files from a single folder → (t, J, label_str).

    Stress is derived from the actual hold-period load data.
    *label* defaults to the folder name if not given.
    """
    from src.utils import calculate_compliance

    folder = Path(folder)
    dyn_files = sorted(folder.glob("*LC_DYN.txt"))
    if not dyn_files:
        print(f"  [WARN] No DYN files in {folder}")
        return None

    run_curves = []
    hold_loads_uN = []
    hold_depths_nm = []

    for f in dyn_files:
        try:
            df = pd.read_csv(f, sep='\t', skiprows=2, encoding='latin1')
            df.columns = df.columns.str.strip().str.replace('Âµ', 'µ')
            time = pd.to_numeric(df['Test Time (s)'], errors='coerce').values
            disp = pd.to_numeric(df['Indent Disp. (nm)'], errors='coerce').values
            load = pd.to_numeric(df['Indent Load (µN)'], errors='coerce').values
        except Exception:
            continue

        valid = np.isfinite(time) & np.isfinite(disp) & np.isfinite(load)
        if valid.sum() < 100:
            continue
        time, disp, load = time[valid], disp[valid], load[valid]

        max_load = np.nanmax(load)
        mask = load >= 0.98 * max_load
        if mask.sum() < 100:
            continue
        t_hold = time[mask] - time[mask][0]
        d_hold = disp[mask]
        l_hold = load[mask]
        hold_loads_uN.append(np.nanmean(l_hold))
        hold_depths_nm.append(np.nanmean(d_hold))

        try:
            J = calculate_compliance(
                probe_type, d_hold, l_hold,
                poisson_ratio=poisson_ratio, area_m2=area_m2,
            )
        except Exception:
            continue

        ok = np.isfinite(J) & (J > 0) & (t_hold > 0)
        if ok.sum() < 50:
            continue
        run_curves.append((t_hold[ok], J[ok]))

    if not run_curves:
        print(f"  [WARN] No valid runs in {folder}")
        return None

    # Average on log-spaced grid
    t_min = max(rc[0][rc[0] > 0][0] for rc in run_curves)
    t_max = min(rc[0][-1] for rc in run_curves)
    if t_max <= t_min:
        return None
    t_grid = np.logspace(np.log10(t_min), np.log10(t_max), 500)

    import warnings
    J_stack = []
    for t_r, J_r in run_curves:
        fn = interp1d(t_r, J_r, bounds_error=False, fill_value=np.nan)
        J_stack.append(fn(t_grid))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        J_avg = np.nanmean(J_stack, axis=0)
    ok = ~np.isnan(J_avg)
    t_avg, J_avg = t_grid[ok], J_avg[ok]

    if time_range:
        mask = (t_avg >= time_range[0]) & (t_avg <= time_range[1])
        t_avg, J_avg = t_avg[mask], J_avg[mask]

    if len(t_avg) == 0:
        return None

    # Derive stress from mean hold load
    mean_load_uN = np.mean(hold_loads_uN) if hold_loads_uN else 0
    mean_depth_nm = np.mean(hold_depths_nm) if hold_depths_nm else 0
    if probe_type == 'flat_punch' and area_m2:
        stress_MPa = (mean_load_uN * 1e-6) / area_m2 / 1e6  # µN→N, Pa→MPa
    elif mean_depth_nm > 0:
        # Estimate mean contact pressure (hardness) from tip geometry
        h_m = mean_depth_nm * 1e-9  # nm → m
        if 'berkovich' in probe_type.lower():
            A_proj = 24.5 * h_m ** 2  # ideal Berkovich area function
        elif 'conical' in probe_type.lower():
            A_proj = np.pi * (np.tan(np.radians(60)) * h_m) ** 2  # 60° semi-angle
        else:
            A_proj = None
        if A_proj and A_proj > 0:
            stress_MPa = (mean_load_uN * 1e-6) / A_proj / 1e6  # H in MPa
        else:
            stress_MPa = None
    else:
        stress_MPa = None
    lbl = label or folder.name
    stress_str = f"{stress_MPa:.1f} MPa" if stress_MPa is not None else f"{mean_load_uN:.0f} µN"
    print(f"  {lbl}: {len(run_curves)} runs, {len(t_avg)} pts, "
          f"mean load {mean_load_uN:.0f} µN ({stress_str}), "
          f"J range: {J_avg.min():.4f}–{J_avg.max():.4f} 1/GPa")
    return t_avg, J_avg, lbl, stress_MPa


def _shade_group(ax, curves, color, group_label, linestyle='-', marker='o'):
    """Plot mean ± min/max envelope for a list of (t, J) curves.

    Interpolates all curves onto a common log-spaced time grid,
    then draws the mean as a line with markers and fills between min and max.
    """
    if not curves:
        return
    # Use the full range across all curves (not just the shortest)
    t_starts = [tc[tc > 0][0] for tc, _ in curves]
    t_ends = [tc[-1] for tc, _ in curves]
    t_min = min(t_starts)
    t_max = max(t_ends)
    if t_max <= t_min:
        return
    t_grid = np.logspace(np.log10(t_min), np.log10(t_max), 500)

    J_stack = []
    for tc, Jc in curves:
        fn = interp1d(tc, Jc, bounds_error=False, fill_value=np.nan)
        J_stack.append(fn(t_grid))
    J_stack = np.array(J_stack)

    # Require at least 2 curves contributing (or 1 if only 1 curve total)
    min_count = min(2, len(curves))
    count = np.sum(np.isfinite(J_stack), axis=0)
    J_mean = np.nanmean(J_stack, axis=0)
    J_min = np.nanmin(J_stack, axis=0)
    J_max = np.nanmax(J_stack, axis=0)
    ok = np.isfinite(J_mean) & (count >= min_count)

    ax.plot(t_grid[ok], J_mean[ok], linestyle, color=color, linewidth=1.5,
            marker=marker, markevery=max(1, int(ok.sum()) // 12), markersize=6,
            label=group_label)
    ax.fill_between(t_grid[ok], J_min[ok], J_max[ok],
                    color=color, alpha=0.2, rasterized=True)


def smooth_curve(t, J, window):
    """Running average in log-time space for smoother curves."""
    log_t = np.log10(t)
    n_pts = 500
    log_grid = np.linspace(log_t.min(), log_t.max(), n_pts)
    J_smooth = np.empty(n_pts)
    for i, lg in enumerate(log_grid):
        mask = np.abs(log_t - lg) <= window / 2
        J_smooth[i] = np.nanmean(J[mask]) if mask.any() else np.nan
    valid = ~np.isnan(J_smooth)
    return 10 ** log_grid[valid], J_smooth[valid]


# ── Model fitting ────────────────────────────────────────────────────────
def _print_params(params):
    """Print model parameters."""
    for k, v in params.items():
        if isinstance(v, dict):
            val = v.get('value', v)
            err = v.get('stderr', 0)
            units = v.get('units', '')
            print(f"    {k}: {val:.6g} ± {err:.6g} {units}")
        elif isinstance(v, (int, float)):
            print(f"    {k}: {v:.6g}")
        else:
            print(f"    {k}: {v}")


def fit_peng_model(t, J, stress_MPa, locks_MPa, yield_MPa):
    """Fit the PengStressLockModel at a given stress level.

    Uses the same model code as creepycrawlies --fit_model peng-vp.
    set_stress() gates the Heaviside-locked elements based on the applied
    stress, and stress thresholds are passed to initial_guess().

    Returns (t_fit, J_fit, label) or None.
    """
    try:
        from src.models import PengStressLockModel
    except ImportError:
        print("[WARN] Cannot import PengStressLockModel")
        return None

    n_locks = len(locks_MPa)
    include_vp = yield_MPa is not None
    stress_GPa = stress_MPa / 1000.0

    model = PengStressLockModel(
        n_locked_elements=n_locks,
        include_viscoplastic=include_vp,
        applied_stress=stress_GPa,
    )

    # Convert thresholds to GPa for initial_guess
    thresholds_GPa = [lk / 1000.0 for lk in locks_MPa]
    yield_GPa = yield_MPa / 1000.0 if yield_MPa else None

    model.fit(t, J,
              stress_thresholds=thresholds_GPa,
              yield_stress=yield_GPa)

    stats = model.get_statistics()
    r2 = stats.get('R_squared', stats.get('R²', stats.get('r_squared', 0)))

    # Build Peng label based on active elements
    n_active = sum(1 for lk in locks_MPa if stress_MPa > lk)
    above_yield = yield_MPa is not None and stress_MPa > yield_MPa
    n_vk = 1 + n_active

    # If fit blew up (R² < 0), retry with fewer locked elements
    while r2 < 0 and n_active > 0:
        n_active -= 1
        n_vk = 1 + n_active
        print(f"    R²={r2:.1f} — reducing to {n_vk} VK elements")
        model = PengStressLockModel(
            n_locked_elements=max(n_active, 0),
            include_viscoplastic=False,
            applied_stress=stress_GPa,
        )
        # Only use thresholds below current stress
        active_thresholds = thresholds_GPa[:n_active] if n_active > 0 else []
        model.fit(t, J,
                  stress_thresholds=active_thresholds,
                  yield_stress=None)
        stats = model.get_statistics()
        r2 = stats.get('R_squared', stats.get('R²', stats.get('r_squared', 0)))

    # Final fallback to plain Burgers if still bad
    if r2 < 0:
        print(f"    R²={r2:.1f} — falling back to Burgers")
        try:
            from src.models import get_model
            model = get_model('burgers')
            model.fit(t, J)
            stats = model.get_statistics()
            r2 = stats.get('R_squared', stats.get('R²', stats.get('r_squared', 0)))
            n_active = 0
            n_vk = 1
        except Exception:
            return None

    above_yield = yield_MPa is not None and stress_MPa > yield_MPa and n_active == len(locks_MPa)
    if above_yield:
        desc = f"Peng-VP (M+{n_vk}VK+VP)"
    elif n_active == 0:
        desc = f"Peng (M+{n_vk}VK)"
    else:
        desc = f"Peng-{n_active} (M+{n_vk}VK)"

    label = f"{desc} (R²={r2:.4f})"
    print(f"  Fit {desc}: R²={r2:.4f}")
    params = model.get_parameters()
    _print_params(params)

    t_fit = np.logspace(np.log10(t.min()), np.log10(t.max()), 500)
    J_fit = model.predict(t_fit)
    return t_fit, J_fit, label, desc, r2, params


def fit_generic_model(t, J, model_name):
    """Fit a generic creep model (gen-kelvin-N, burgers, etc.)."""
    try:
        from src.models import get_model
    except ImportError:
        print("[WARN] Cannot import models for fitting")
        return None

    model = get_model(model_name)
    model.fit(t, J)
    stats = model.get_statistics()
    t_fit = np.logspace(np.log10(t.min()), np.log10(t.max()), 500)
    J_fit = model.predict(t_fit)
    r2 = stats.get('R_squared', stats.get('R²', stats.get('r_squared', 0)))
    label = f"{model_name} (R²={r2:.4f})"
    print(f"  Fit {model_name}: R²={r2:.4f}")
    _print_params(model.get_parameters())
    return t_fit, J_fit, label


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Compare bulk and nano shear creep compliance.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ASTM 17.5 MPa vs nano raw data, first 300s, Peng stress-lock fits
  python scripts/compare_compliance.py \\
      --astm bulk/bulkdata/astm/17.5MPa \\
      --nano-dir /Users/ethan/data/t1500/peng \\
      --time-range 1 300 --log \\
      --peng-locks 17.5 25 --yield-stress 33

  # Just bulk TSSP vs nano CSV
  python scripts/compare_compliance.py \\
      --bulk bulk/out/tssp/tssp_15_averaged_creep.csv \\
      --nano out/nano_master/nano_master_curve.csv --log

  # Single model fit to everything
  python scripts/compare_compliance.py \\
      --astm bulk/bulkdata/astm/17.5MPa \\
      --nano-dir /Users/ethan/data/t1500/peng \\
      --time-range 1 300 --fit-model gen-kelvin-5 --log
        """,
    )
    # Data sources
    parser.add_argument('--tssp', nargs='+', default=[],
                        help='Bulk TSSP averaged compliance CSV(s)')
    parser.add_argument('--24hour', nargs='+', default=[], dest='hour24',
                        help='ASTM 24-hour raw data folder(s) (each = one stress level)')
    parser.add_argument('--nano', nargs='+', default=[],
                        help='Nano compliance CSV(s) (master curve or per-stress)')
    parser.add_argument('--nano-dir', default=None,
                        help='Directory with 103-* nano DYN subfolders (loads raw data)')
    parser.add_argument('--nano-fp', nargs='+', default=[],
                        help='Individual nano DYN folder(s) (flat punch, one stress each)')
    parser.add_argument('--nano-berk', nargs='+', default=[],
                        help='Individual nano DYN folder(s) (Berkovich)')
    parser.add_argument('--nano-con', nargs='+', default=[],
                        help='Individual nano DYN folder(s) (conical)')

    # Filtering
    parser.add_argument('--time-range', nargs=2, type=float, default=None,
                        metavar=('T_START', 'T_END'),
                        help='Filter time window in seconds (e.g. 1 300)')
    parser.add_argument('--smooth-window', type=float, default=None,
                        metavar='DECADES',
                        help='Running-average window in log10 decades (e.g. 0.3)')

    # Fitting
    parser.add_argument('--fit-model', default=None,
                        help='Fit same model to all curves (e.g. gen-kelvin-5, burgers)')
    parser.add_argument('--peng-locks', nargs='+', type=float, default=None,
                        metavar='STRESS_MPa',
                        help='Peng stress-lock thresholds in MPa (e.g. 17.5 25)')
    parser.add_argument('--yield-stress', type=float, default=33.0,
                        help='Viscoplastic yield stress in MPa (default: 33)')

    # Nano probe settings
    parser.add_argument('--probe-type', default='flat_punch',
                        choices=['flat_punch', 'conical', 'frustum'])
    parser.add_argument('--area', type=float, default=3.62e-10,
                        help='Flat punch area in m² (default: 3.62e-10)')

    # Output
    parser.add_argument('--out', default='out', help='Output directory')
    parser.add_argument('--log', action='store_true', help='Log-log axes')
    parser.add_argument('--log-x', action='store_true', help='Log x-axis only')
    parser.add_argument('--log-y', action='store_true', help='Log y-axis only')
    parser.add_argument('--journal', action='store_true', help='Journal style')
    parser.add_argument('--shade', action='store_true',
                        help='Average curves per source type and show as shaded region')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom title for the plot')
    parser.add_argument('--thesis-figures', action='store_true',
                        help='Save figures to out/figures/ch3/ for thesis integration')
    args = parser.parse_args()

    has_data = args.tssp or args.hour24 or args.nano or args.nano_dir or args.nano_fp or args.nano_berk or args.nano_con
    if not has_data:
        parser.error("Provide at least one data source")

    if args.thesis_figures:
        from src.plot_style import THESIS_FIGURES_DIR
        args.out = str(THESIS_FIGURES_DIR)

    apply_journal_style()

    fig, ax = plt.subplots(figsize=(14, 9))
    # Each entry: (t, J, color, label, stress_MPa_or_None)
    plotted_curves = []
    # For --shade: collect (t, J) per group key
    shade_groups = {}  # key → {'curves': [(t,J),...], 'color': str, 'ls': str}

    # ── Category colors (fixed per source type) ──
    from style import TOL_COLORS, MARKERS
    COLOR_BULK  = '#0077BB'   # blue
    COLOR_ASTM  = '#EE7733'   # orange
    COLOR_NANO  = '#009988'   # teal  (nano-dir / Peng)
    COLOR_BERK  = '#CC3311'   # red
    COLOR_CON   = '#33BBEE'   # cyan
    COLOR_FP    = '#EE3377'   # magenta
    marker_idx = 0  # global marker counter

    def _next_marker():
        nonlocal marker_idx
        m = MARKERS[marker_idx % len(MARKERS)]
        marker_idx += 1
        return m

    # ── Plot TSSP ──
    for path in args.tssp:
        p = Path(path)
        if p.is_dir():
            result = load_astm_folder(p, time_range=args.time_range)
            if result is None:
                print(f"  [WARN] No valid TSSP data from {p}")
                continue
            t, J, stress = result
        else:
            t, J, stress = load_bulk(path, time_range=args.time_range)
        if len(t) == 0:
            continue
        if stress is not None:
            label = f"TSSP {stress:.1f} MPa"
        else:
            label = p.stem
        if args.shade:
            key = 'TSSP'
            shade_groups.setdefault(key, {'curves': [], 'color': COLOR_BULK, 'ls': '-'})
            shade_groups[key]['curves'].append((t, J))
        else:
            marker = _next_marker()
            ax.plot(t, J, '-', label=label, color=COLOR_BULK, marker=marker,
                    markevery=max(1, len(t) // 12), markersize=6, linewidth=1.5)
        plotted_curves.append((t, J, COLOR_BULK, label, stress))

    # ── Plot 24-hour ──
    astm_folders = []
    for folder in args.hour24:
        p = Path(folder)
        csvs = list(p.glob("*.csv"))
        subdirs = sorted([d for d in p.iterdir() if d.is_dir()]) if p.is_dir() else []
        if csvs:
            astm_folders.append(p)
        elif subdirs:
            print(f"  Auto-discovered {len(subdirs)} ASTM stress folders in {p}")
            astm_folders.extend(subdirs)
        else:
            print(f"  [WARN] No CSVs or subdirs in {folder}")

    for folder in astm_folders:
        result = load_astm_folder(folder, time_range=args.time_range)
        if result is None:
            print(f"  [WARN] No valid data from {folder}")
            continue
        t, J, stress = result
        label = f"24hr {stress:.1f} MPa"
        if args.shade:
            key = '24hr'
            shade_groups.setdefault(key, {'curves': [], 'color': COLOR_ASTM, 'ls': '-'})
            shade_groups[key]['curves'].append((t, J))
        else:
            marker = _next_marker()
            ax.plot(t, J, '-', label=label, color=COLOR_ASTM, marker=marker,
                    markevery=max(1, len(t) // 12), markersize=7, linewidth=1.5,
                    markerfacecolor='none', markeredgecolor=COLOR_ASTM, markeredgewidth=1.5)
        plotted_curves.append((t, J, COLOR_ASTM, label, stress))

    # ── Plot nano CSVs ──
    for path in args.nano:
        t, J = load_nano(path, time_range=args.time_range)
        if len(t) == 0:
            continue
        if args.smooth_window:
            t, J = smooth_curve(t, J, args.smooth_window)
        label = Path(path).stem.replace('_', ' ').title()
        if args.shade:
            key = 'Nano'
            shade_groups.setdefault(key, {'curves': [], 'color': COLOR_NANO, 'ls': '--'})
            shade_groups[key]['curves'].append((t, J))
        else:
            marker = _next_marker()
            ax.plot(t, J, '--', label=label, color=COLOR_NANO, marker=marker,
                    markevery=max(1, len(t) // 12), markersize=6, linewidth=1.5)
        plotted_curves.append((t, J, COLOR_NANO, label, None))

    # ── Plot nano raw DYN data (per stress level) ──
    if args.nano_dir:
        print(f"\nLoading nano DYN data from {args.nano_dir}...")
        nano_curves = load_nano_dir(
            args.nano_dir, time_range=args.time_range,
            probe_type=args.probe_type, area_m2=args.area,
        )
        for stress in sorted(nano_curves.keys()):
            t, J = nano_curves[stress]
            if args.smooth_window:
                t, J = smooth_curve(t, J, args.smooth_window)
            label = f"Nano {stress:.1f} MPa"
            if args.shade:
                key = 'Nano'
                shade_groups.setdefault(key, {'curves': [], 'color': COLOR_NANO, 'ls': '--'})
                shade_groups[key]['curves'].append((t, J))
            else:
                marker = _next_marker()
                ax.plot(t, J, '--', label=label, color=COLOR_NANO, marker=marker,
                        markevery=max(1, len(t) // 12), markersize=6, linewidth=1.5)
            plotted_curves.append((t, J, COLOR_NANO, label, stress))

    # ── Plot nano flat-punch folders ──
    for fp_path in args.nano_fp:
        result = load_nano_single(
            fp_path, time_range=args.time_range,
            probe_type=args.probe_type, area_m2=args.area,
        )
        if result is None:
            continue
        t, J, lbl, stress_MPa = result
        if args.smooth_window:
            t, J = smooth_curve(t, J, args.smooth_window)
        label = f"FP {lbl}"
        if args.shade:
            key = 'Flat Punch'
            shade_groups.setdefault(key, {'curves': [], 'color': COLOR_FP, 'ls': '--'})
            shade_groups[key]['curves'].append((t, J))
        else:
            marker = _next_marker()
            ax.plot(t, J, '--', label=label, color=COLOR_FP, marker=marker,
                    markevery=max(1, len(t) // 12), markersize=6, linewidth=1.5)
        plotted_curves.append((t, J, COLOR_FP, label, stress_MPa))

    # ── Plot nano Berkovich folders ──
    for berk_path in args.nano_berk:
        result = load_nano_single(
            berk_path, time_range=args.time_range,
            probe_type='berkovich',
        )
        if result is None:
            continue
        t, J, lbl, stress_MPa = result
        if args.smooth_window:
            t, J = smooth_curve(t, J, args.smooth_window)
        label = f"Berk {lbl}"
        if args.shade:
            key = 'Berkovich'
            shade_groups.setdefault(key, {'curves': [], 'color': COLOR_BERK, 'ls': '--'})
            shade_groups[key]['curves'].append((t, J))
        else:
            marker = _next_marker()
            ax.plot(t, J, '--', label=label, color=COLOR_BERK, marker=marker,
                    markevery=max(1, len(t) // 12), markersize=6, linewidth=1.5)
        plotted_curves.append((t, J, COLOR_BERK, label, stress_MPa))

    # ── Plot nano conical folders ──
    for con_path in args.nano_con:
        result = load_nano_single(
            con_path, time_range=args.time_range,
            probe_type='conical_60',
        )
        if result is None:
            continue
        t, J, lbl, stress_MPa = result
        if args.smooth_window:
            t, J = smooth_curve(t, J, args.smooth_window)
        label = f"Con {lbl}"
        if args.shade:
            key = 'Conical'
            shade_groups.setdefault(key, {'curves': [], 'color': COLOR_CON, 'ls': '--'})
            shade_groups[key]['curves'].append((t, J))
        else:
            marker = _next_marker()
            ax.plot(t, J, '--', label=label, color=COLOR_CON, marker=marker,
                    markevery=max(1, len(t) // 12), markersize=6, linewidth=1.5)
        plotted_curves.append((t, J, COLOR_CON, label, stress_MPa))

    # ── Render shaded regions ──
    if args.shade:
        from style import MARKERS as _SHADE_MARKERS
        for i, (group_label, info) in enumerate(shade_groups.items()):
            n = len(info['curves'])
            _shade_group(ax, info['curves'], info['color'],
                         f"{group_label} (n={n})", linestyle=info['ls'],
                         marker=_SHADE_MARKERS[i % len(_SHADE_MARKERS)])

    # ── Fit models (skip when shading) ──
    fit_rows = []  # collect for CSV table
    if args.peng_locks and plotted_curves and not args.shade:
        locks = sorted(args.peng_locks)
        yield_s = args.yield_stress
        print(f"\nPeng stress-lock fitting (PengStressLockModel): "
              f"locks={locks} MPa, yield={yield_s} MPa")
        for t, J, color, src_label, stress in plotted_curves:
            if stress is None:
                print(f"  Skipping {src_label}: no stress level known")
                continue
            print(f"\n  {src_label} (σ = {stress:.1f} MPa):")
            result = fit_peng_model(t, J, stress, locks, yield_s)
            if result is None:
                continue
            t_fit, J_fit, fit_label, desc, r2, params = result
            ax.plot(t_fit, J_fit, ':', color=color, linewidth=2.5, alpha=0.85)
            row = {'source': src_label, 'stress_MPa': stress,
                   'model': desc, 'R2': r2}
            for pname, pinfo in params.items():
                if isinstance(pinfo, dict):
                    row[pname] = pinfo['value']
                    row[f'{pname}_stderr'] = pinfo.get('stderr') or 0.0
                else:
                    row[pname] = pinfo
            fit_rows.append(row)

    elif args.fit_model and plotted_curves and not args.shade:
        print(f"\nFitting {args.fit_model} to {len(plotted_curves)} curve(s)...")
        for t, J, color, src_label, stress in plotted_curves:
            result = fit_generic_model(t, J, args.fit_model)
            if result is None:
                continue
            t_fit, J_fit, fit_label = result
            ax.plot(t_fit, J_fit, ':', color=color, linewidth=2.5, alpha=0.85)

    # ── Axes ──
    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(r'Shear Creep Compliance $J(t)$ (1/GPa)')
    if args.log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if args.log_x:
        ax.set_xscale('log')
    if args.log_y:
        ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()
    n_entries = len(handles)
    ncol = min(6, max(3, n_entries // 6))
    leg_fontsize = 14 if n_entries > 15 else 18
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              frameon=True, fontsize=leg_fontsize, markerscale=1.2,
              ncol=ncol, handletextpad=0.3, columnspacing=0.8)

    if args.title:
        ax.set_title(_wrap_title(args.title, fig, ax), pad=20)
    else:
        title_parts = ['Bulk and Nano Shear Creep Compliance J(t) (1/GPa) vs. Creep Time (s)']
        if args.log:
            title_parts[0] += ' (log-log)'
        if args.time_range:
            title_parts.append(f": First {int(args.time_range[1])} seconds")
        ax.set_title(_wrap_title(''.join(title_parts), fig, ax), pad=20)

    # ── Save ──
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_log' if args.log else ''
    if args.time_range:
        suffix += f'_{int(args.time_range[0])}-{int(args.time_range[1])}s'
    if args.peng_locks:
        suffix += '_peng'
    if args.shade:
        suffix += '_shaded'
    out_path = out_dir / f'compliance_comparison{suffix}'
    plt.subplots_adjust(bottom=0.35)
    from style import save_figure
    save_figure(fig, str(out_path), formats=['pdf', 'png'], close=False)
    print(f"\nSaved: {out_path}.pdf + .png")
    plt.close()

    # ── Save fit parameters table ──
    if fit_rows:
        csv_path = out_dir / f'fit_params{suffix}.csv'
        fit_df = pd.DataFrame(fit_rows)
        # Reorder columns: source, stress, model, R2 first, then params
        front = ['source', 'stress_MPa', 'model', 'R2']
        rest = [c for c in fit_df.columns if c not in front]
        fit_df = fit_df[front + rest]
        fit_df.to_csv(csv_path, index=False)
        print(f"Saved fit params: {csv_path}")
        # Print table to console
        print(f"\n{'─' * 100}")
        print("Fit Parameters Table")
        print(f"{'─' * 100}")
        print(fit_df.to_string(index=False))
        print(f"{'─' * 100}")


if __name__ == '__main__':
    main()
