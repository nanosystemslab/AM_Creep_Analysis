#!/usr/bin/env python3
"""
TSSP Master Curve Construction (Jazouli et al. 2005)

Builds a time-stress superposition master curve from creep compliance data
at multiple stress levels.  Horizontal shifting on log-time produces a
single master compliance curve and fits the shift-factor equation:

    log phi_sigma = -C1 * (sigma - sigma0) / (C3 + (sigma - sigma0))

Usage:
    cd bulk/
    python ../src/tssp_master.py \
        --compliance out/tssp/tssp_compliance_combined.csv \
        --reference-stress 15

    python ../src/tssp_master.py \
        --creep-dir out/tssp/ \
        --reference-stress 15
"""

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_journal_style, get_stress_style


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


# ───────────────────────────────────────────────────────────────────────────
# Constants (matching prepare_bulk_data.py / tssp_sync.py)
# ───────────────────────────────────────────────────────────────────────────
SPECIMEN_LENGTH_MM = 25.4           # nominal specimen height (1 inch)
SPECIMEN_AREA_MM2 = np.pi * (12.7 / 2) ** 2  # 126.68 mm²


def _thin(x, y, n_markers=10):
    """Subsample arrays for marker plotting so they don't overlap."""
    step = max(1, len(x) // n_markers)
    return x[::step], y[::step]


def _thin_log(log_t, y, n_markers=10):
    """Pick n_markers evenly spaced in log10(t) space."""
    if len(log_t) < n_markers:
        return log_t, y
    targets = np.linspace(log_t.min(), log_t.max(), n_markers)
    idx = np.searchsorted(log_t, targets).clip(0, len(log_t) - 1)
    idx = np.unique(idx)
    return log_t[idx], y[idx]


# ───────────────────────────────────────────────────────────────────────────
# Step 1 — Load compliance data
# ───────────────────────────────────────────────────────────────────────────
def _load_raw_df(compliance_csv: Path | None = None,
                 creep_dir: Path | None = None) -> pd.DataFrame:
    """Load the raw DataFrame from combined CSV or individual creep CSVs."""
    if compliance_csv is not None and compliance_csv.exists():
        return pd.read_csv(compliance_csv)
    if creep_dir is not None and creep_dir.exists():
        parts = []
        for f in sorted(creep_dir.glob("tssp_*_creep.csv")):
            if '_averaged_' in f.name:
                continue
            part = pd.read_csv(f)
            has_data = (
                'J_video' in part.columns or
                'J_stroke' in part.columns or
                'compliance_1_per_MPa' in part.columns or
                'stroke_mm' in part.columns
            )
            if has_data:
                parts.append(part)
        if parts:
            return pd.concat(parts, ignore_index=True)
    print("Error: Provide --compliance or --creep-dir with valid data")
    sys.exit(1)


def load_compliance(compliance_csv: Path | None = None,
                    creep_dir: Path | None = None,
                    strain_source: str = "video",
                    use_averaged: bool = False) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Load bulk creep compliance grouped by stress level.

    The new pipeline (tssp_sync.py) outputs J_video and J_stroke columns
    that already include the 2(1+nu) factor.  This function reads them
    directly — no Poisson multiplication needed here.

    strain_source selects which column to use:
        "video"  -> J_video (preferred)
        "stroke" -> J_stroke
        "any"    -> J_video first, fallback to J_stroke per-group

    Returns {stress_MPa: (time_sec, J_1_per_GPa)} with t > 0 only.
    """
    # --- Averaged data path ---
    if use_averaged:
        avg_csv = None
        if creep_dir is not None:
            avg_csv = creep_dir / "tssp_averaged_compliance.csv"
        if compliance_csv is not None:
            # If compliance_csv points to combined, look for averaged sibling
            avg_csv = compliance_csv.parent / "tssp_averaged_compliance.csv"
        if avg_csv is not None and avg_csv.exists():
            print(f"  Loading averaged compliance from: {avg_csv}")
            df_avg = pd.read_csv(avg_csv)
            curves = {}
            for stress, grp in df_avg.groupby('target_stress_MPa'):
                g = grp.sort_values('time_sec_creep').copy()
                g = g[g['time_sec_creep'] > 0]
                valid = g['J_mean'].notna()
                g = g[valid]
                if len(g) < 5:
                    print(f"  Skipping {stress} MPa — insufficient data (< 5 points)")
                    continue
                n_runs = int(g['n_runs'].iloc[0])
                J_avg = g['J_mean'].values * 1000.0  # 1/MPa → 1/GPa
                # Warn if J(0) is near zero — likely stale CSVs with eps_stroke
                # zeroed to creep start instead of test start
                if len(J_avg) > 0 and J_avg[0] < 0.1:
                    print(f"  WARNING: {stress:.0f} MPa averaged J(0)={J_avg[0]:.4f} 1/GPa "
                          f"≈ 0 — CSVs may need regeneration (run tssp_batch.py)")
                print(f"  {stress:.0f} MPa: {len(g)} points, N={n_runs} runs (averaged)")
                curves[float(stress)] = (g['time_sec_creep'].values, J_avg)
            print(f"Loaded averaged compliance for {len(curves)} stress levels: "
                  f"{sorted(curves.keys())} MPa")
            return curves
        else:
            print("  Warning: --use-averaged set but tssp_averaged_compliance.csv not found")
            print("  Falling back to individual compliance data")

    df = _load_raw_df(compliance_csv, creep_dir)

    if strain_source not in {"video", "stroke", "any"}:
        print(f"Error: Invalid strain_source '{strain_source}'.")
        sys.exit(1)

    # Determine which J column(s) are available
    has_J_video = 'J_video' in df.columns and df['J_video'].notna().any()
    has_J_stroke = 'J_stroke' in df.columns and df['J_stroke'].notna().any()

    if has_J_video or has_J_stroke:
        print("  Using pre-computed bulk compliance (J already includes 2(1+nu) factor)")
    else:
        # Legacy fallback: old-format data with compliance_1_per_MPa
        if 'compliance_1_per_MPa' not in df.columns:
            print("Error: No J_video, J_stroke, or compliance_1_per_MPa columns found")
            sys.exit(1)
        print("  Warning: Legacy data format detected (compliance_1_per_MPa). "
              "Re-run tssp_sync for the new dual-source format.")

    required = {'target_stress_MPa', 'time_sec_creep'}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: Missing columns {missing}")
        sys.exit(1)

    curves = {}
    for stress, grp in df.groupby('target_stress_MPa'):
        g = grp.sort_values('time_sec_creep').copy()
        g = g[g['time_sec_creep'] > 0]

        J = None

        if has_J_video or has_J_stroke:
            # New format: pick column based on strain_source preference
            if strain_source == "video" and has_J_video:
                valid = g['J_video'].notna()
                if valid.any():
                    g = g[valid]
                    J = g['J_video'].values
            elif strain_source == "stroke" and has_J_stroke:
                valid = g['J_stroke'].notna()
                if valid.any():
                    g = g[valid]
                    J = g['J_stroke'].values
            elif strain_source == "any":
                # Prefer video, fallback to stroke per-group
                if 'J_video' in g.columns and g['J_video'].notna().any():
                    valid = g['J_video'].notna()
                    g = g[valid]
                    J = g['J_video'].values
                elif 'J_stroke' in g.columns and g['J_stroke'].notna().any():
                    valid = g['J_stroke'].notna()
                    g = g[valid]
                    J = g['J_stroke'].values

            # Fallback within new format: try the other source
            if J is None and strain_source == "video" and has_J_stroke:
                valid = g['J_stroke'].notna()
                if valid.any():
                    print(f"  {stress:.0f} MPa: J_video all NaN, falling back to J_stroke")
                    g = g[valid]
                    J = g['J_stroke'].values

        if J is None:
            # Legacy fallback
            if 'compliance_1_per_MPa' in g.columns:
                valid = g['compliance_1_per_MPa'].notna()
                g = g[valid]
                if len(g) > 0:
                    # Legacy data needs factor applied
                    nu = 0.43
                    if 'poisson_ratio' in g.columns and g['poisson_ratio'].notna().any():
                        nu = g['poisson_ratio'].dropna().iloc[0]
                    factor = 2.0 * (1.0 + nu)
                    J = g['compliance_1_per_MPa'].values * factor * 1000.0  # → 1/GPa
                    print(f"  {stress:.0f} MPa: legacy format, applying 2(1+nu)={factor:.3f}")

        if J is None or len(g) < 5:
            print(f"  Skipping {stress} MPa — insufficient data (< 5 points)")
            continue

        # Recompute J from total displacement (stroke_mm is raw MTS value
        # from test start).  The saved J_stroke/eps_stroke are zeroed to
        # creep start, but stroke_mm includes the elastic ramp — giving
        # non-zero J(0) that matches ASTM convention.
        # Time uses time_sec_creep (zeroed to creep/hold start).
        t = g['time_sec_creep'].values
        if 'stroke_mm' in g.columns:
            stroke = np.abs(g['stroke_mm'].values)
            sigma = np.abs(g['stress_MPa'].values)
            nu_val = (float(g['poisson_ratio'].iloc[0])
                      if 'poisson_ratio' in g.columns
                      and g['poisson_ratio'].notna().any()
                      else 0.43)
            factor = 2.0 * (1.0 + nu_val)
            eps_total = stroke / SPECIMEN_LENGTH_MM
            sigma_safe = np.where(sigma > 0.01, sigma, np.nan)
            J = factor * eps_total / sigma_safe
            src = "stroke_total"
        else:
            src = ("J_video" if (has_J_video and strain_source != "stroke")
                   else "J_stroke")

        # Convert 1/MPa → 1/GPa
        J = J * 1000.0

        # Filter to t > 0 and finite J
        valid = (t > 0) & np.isfinite(J)
        t = t[valid]
        J = J[valid]

        if len(t) < 5:
            print(f"  Skipping {stress} MPa — insufficient data after filtering")
            continue

        print(f"  {stress:.0f} MPa: {len(t)} points, source={src}")
        curves[float(stress)] = (t, J)

    print(f"Loaded bulk compliance for {len(curves)} stress levels: "
          f"{sorted(curves.keys())} MPa  (units: 1/GPa, t=0 at hold start)")
    return curves


def load_relaxation(creep_dir: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Load relaxation modulus E(t) = sigma(t) / eps_0 from TSSP data.

    For each relaxation CSV, reads the corresponding creep CSV to get
    eps_0 (final eps_stroke from the creep phase).  Multiple runs per
    stress are averaged.

    Returns {stress_MPa: (time_sec, E_MPa)}.
    """
    relax_files = sorted(creep_dir.glob("tssp_*_relaxation.csv"))
    relax_files = [f for f in relax_files if '_averaged_' not in f.name]

    if not relax_files:
        print("Error: No relaxation CSVs found in", creep_dir)
        sys.exit(1)

    # Group by stress level, accumulate per-run E(t) curves
    stress_runs: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {}

    for rf in relax_files:
        df_relax = pd.read_csv(rf)
        if 'time_sec_relax' not in df_relax.columns:
            print(f"  Skipping {rf.name}: no time_sec_relax column")
            continue

        stress = float(df_relax['target_stress_MPa'].iloc[0])
        run_label = df_relax['run_label'].iloc[0]

        # Find matching creep CSV for eps_0
        creep_file = creep_dir / rf.name.replace('_relaxation.csv', '_creep.csv')
        if not creep_file.exists():
            print(f"  Skipping {rf.name}: no matching creep CSV")
            continue

        df_creep = pd.read_csv(creep_file)
        if 'eps_stroke' not in df_creep.columns:
            print(f"  Skipping {rf.name}: creep CSV has no eps_stroke column")
            continue

        eps_stroke = df_creep['eps_stroke'].dropna()
        if len(eps_stroke) == 0:
            print(f"  Skipping {rf.name}: eps_stroke all NaN")
            continue
        eps_0 = float(eps_stroke.iloc[-1])
        if eps_0 <= 0:
            print(f"  Skipping {rf.name}: eps_0 = {eps_0:.6f} (non-positive)")
            continue

        t = df_relax['time_sec_relax'].values
        sigma = df_relax['stress_MPa'].values
        mask = t > 0
        t = t[mask]
        sigma = sigma[mask]
        if len(t) < 5:
            continue

        E = sigma / eps_0  # MPa
        print(f"  {stress:.0f} MPa run {run_label}: {len(t)} pts, "
              f"eps_0={eps_0:.6f}, E(0)={E[0]:.1f} MPa")

        stress_runs.setdefault(stress, []).append((t, E))

    # Average runs per stress level
    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for stress in sorted(stress_runs.keys()):
        runs = stress_runs[stress]
        if len(runs) == 1:
            curves[stress] = runs[0]
        else:
            # Interpolate all runs onto the shortest time grid
            t_common = runs[0][0]
            for t_r, _ in runs[1:]:
                t_common = t_common[t_common <= t_r[-1]]
            if len(t_common) < 5:
                curves[stress] = runs[0]
                continue
            E_stack = []
            for t_r, E_r in runs:
                interp_fn = interp1d(t_r, E_r, kind='linear',
                                     bounds_error=False, fill_value=np.nan)
                E_stack.append(interp_fn(t_common))
            E_avg = np.nanmean(E_stack, axis=0)
            valid = ~np.isnan(E_avg)
            curves[stress] = (t_common[valid], E_avg[valid])
            print(f"  {stress:.0f} MPa: averaged {len(runs)} runs → "
                  f"{valid.sum()} points")

    print(f"Loaded relaxation E(t) for {len(curves)} stress levels: "
          f"{sorted(curves.keys())} MPa")
    return curves


def load_astm_compliance(astm_dir: Path,
                         nu: float = 0.43,
                         ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Load ASTM D2990 24-hour creep data and compute shear compliance.

    Reads raw CSVs from astm_dir/<stress>MPa/.  Uses the same 3-row header
    parsing and 95%-max-stress hold detection as prepare_bulk_data.py.
    Computes J = 2(1+nu) * strain / stress  (1/GPa).
    Time zeroed to hold start (t=0 at constant-stress phase).
    Strain is total from test start (includes elastic ramp).
    Averages multiple samples per stress level.

    Returns {stress_MPa: (time_sec, J_1_per_GPa)}.
    """
    stress_dirs = sorted(astm_dir.iterdir())
    stress_runs: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {}

    for sd in stress_dirs:
        if not sd.is_dir():
            continue
        # Parse stress from directory name like "20MPa" or "22.5MPa"
        dirname = sd.name.replace('MPa', '').replace('mpa', '')
        try:
            target_stress = float(dirname)
        except ValueError:
            continue

        csv_files = sorted(sd.glob("*.csv"))
        for cf in csv_files:
            try:
                # 3-row header: row0 = test ID, row1 = column names, row2 = units
                with open(cf, 'r') as fh:
                    lines = fh.readlines()
                header_row = next((i for i, line in enumerate(lines)
                                   if 'Time' in line or 'time' in line.lower()), 0)
                df = pd.read_csv(cf, skiprows=header_row)
                df.columns = df.columns.str.strip('"')

                for col in ['Time', 'Force', 'Stroke']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['Time', 'Force', 'Stroke'])

                # Compute stress and strain
                area = SPECIMEN_AREA_MM2
                length = SPECIMEN_LENGTH_MM
                df['stress'] = df['Force'] / area
                df['strain'] = df['Stroke'] / length

                # 95% max-stress hold detection
                max_stress = df['stress'].max()
                threshold = 0.95 * max_stress
                hold_mask = df['stress'] >= threshold
                if not hold_mask.any():
                    print(f"  Skipping {cf.name}: no hold period detected")
                    continue
                hold_start = df[hold_mask].index[0]
                df_hold = df.loc[hold_start:].copy()

                # Time zeroed to hold start (t=0 at constant-stress hold)
                t_hold_start = df_hold['Time'].iloc[0]
                t_sec = (df_hold['Time'].values - t_hold_start) * 3600.0  # hr → sec

                sigma = df_hold['stress'].values
                eps_test_start = df['strain'].iloc[0]  # total strain from test start
                eps = df_hold['strain'].values - eps_test_start

                # J(t) = 2(1+nu) * eps / sigma  (1/GPa)
                with np.errstate(divide='ignore', invalid='ignore'):
                    J = 2.0 * (1.0 + nu) * eps / sigma * 1000.0
                valid = np.isfinite(J) & (t_sec > 0)
                t_sec = t_sec[valid]
                J = J[valid]

                if len(t_sec) < 10:
                    print(f"  Skipping {cf.name}: too few hold points")
                    continue

                avg_stress = float(np.mean(sigma[valid]))
                print(f"  {cf.name}: {len(t_sec)} pts, "
                      f"sigma_avg={avg_stress:.1f} MPa, "
                      f"J(end)={J[-1]:.3f} 1/GPa")
                stress_runs.setdefault(target_stress, []).append((t_sec, J))

            except Exception as e:
                print(f"  Error reading {cf.name}: {e}")
                continue

    # Average samples per stress
    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for stress in sorted(stress_runs.keys()):
        runs = stress_runs[stress]
        if len(runs) == 1:
            curves[stress] = runs[0]
        else:
            # Interpolate onto common time grid
            t_min = max(r[0][0] for r in runs)
            t_max = min(r[0][-1] for r in runs)
            n_pts = min(len(r[0]) for r in runs)
            t_common = np.linspace(t_min, t_max, n_pts)
            J_stack = []
            for t_r, J_r in runs:
                interp_fn = interp1d(t_r, J_r, kind='linear',
                                     bounds_error=False, fill_value=np.nan)
                J_stack.append(interp_fn(t_common))
            J_avg = np.nanmean(J_stack, axis=0)
            valid = ~np.isnan(J_avg)
            curves[stress] = (t_common[valid], J_avg[valid])
            print(f"  {stress:.0f} MPa: averaged {len(runs)} samples → "
                  f"{valid.sum()} points")

    print(f"Loaded ASTM compliance for {len(curves)} stress levels: "
          f"{sorted(curves.keys())} MPa  (nu={nu}, units: 1/GPa, t=0 at hold start)")
    return curves


# ───────────────────────────────────────────────────────────────────────────
# Step 2 — Plot unshifted J(t) on log-time (Fig. 3)
# ───────────────────────────────────────────────────────────────────────────
def plot_unshifted(curves: dict, out_dir: Path, *,
                   ylabel: str | None = None,
                   title_suffix: str = "",
                   filename: str = "tssp_unshifted_compliance.png"):
    """Unshifted creep compliance (or relaxation modulus) vs log10(time)."""
    fig, ax = plt.subplots(figsize=(14, 9))
    stresses = sorted(curves.keys())

    for stress in stresses:
        color, marker = get_stress_style(stress)
        t, J = curves[stress]
        log_t = np.log10(t)

        # Bin-average to get a clean line (handles multiple runs per stress)
        order = np.argsort(log_t)
        log_t_s, J_s = log_t[order], J[order]
        n_bins = min(500, len(log_t_s) // 3)
        if n_bins > 10:
            bin_edges = np.linspace(log_t_s.min(), log_t_s.max(), n_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_idx = np.clip(np.digitize(log_t_s, bin_edges) - 1, 0, n_bins - 1)
            bin_mean = np.array([np.mean(J_s[bin_idx == i])
                                 if (bin_idx == i).any() else np.nan
                                 for i in range(n_bins)])
            valid = np.isfinite(bin_mean)
            ax.plot(bin_centers[valid], bin_mean[valid], '-', color=color,
                    linewidth=2.0)
            t_m, J_m = _thin_log(bin_centers[valid], bin_mean[valid],
                                 n_markers=10)
        else:
            ax.plot(log_t_s, J_s, '-', color=color, linewidth=2.0)
            t_m, J_m = _thin_log(log_t_s, J_s, n_markers=10)

        ax.plot(t_m, J_m, linestyle='none', marker=marker, color=color,
                markersize=12, markerfacecolor='none', markeredgewidth=2.0,
                label=f'{stress:g} MPa')

    ax.set_xlabel(r'$\log_{10}\;t$ (s)')
    ax.set_ylabel(ylabel or r'Shear Creep Compliance $J(t)$ (1/GPa)')
    if title_suffix:
        ax.set_title(_wrap_title(title_suffix.strip(), fig, ax))
    else:
        ax.set_title(_wrap_title(
            'Bulk 1-hour Unshifted Shear Creep Compliance J(t) (1/GPa) vs. Creep Time (min) (log-log): Nine Stress Levels',
            fig, ax))
    ax.grid(False)
    ax.legend(fontsize=22, markerscale=1.3,
              handletextpad=0.4, columnspacing=1.0)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        out = out_dir / Path(filename).stem
        plt.savefig(f"{out}.{fmt}", bbox_inches='tight')
        print(f"Saved: {out}.{fmt}")
    plt.close()


# ───────────────────────────────────────────────────────────────────────────
# Step 3 — Find individual shift factors
# ───────────────────────────────────────────────────────────────────────────
def find_shift_factor(t_ref: np.ndarray, J_ref: np.ndarray,
                      t_i: np.ndarray, J_i: np.ndarray,
                      min_overlap_frac: float = 0.2,
                      min_overlap_points: int = 200) -> float:
    """
    Find log10(phi_sigma) that minimises RMSE over a sufficiently large overlap.

    Shifts curve i:  log10(t_i) - log10(phi) → must overlap with log10(t_ref).
    """
    log_t_ref = np.log10(t_ref)
    log_t_i = np.log10(t_i)

    # Build reference interpolant
    ref_interp = interp1d(log_t_ref, J_ref, kind='linear',
                          bounds_error=False, fill_value=np.nan)
    j_scale = max(np.nanstd(J_ref), np.nanstd(J_i), 1e-8)
    n_required = min(
        len(log_t_i),
        max(int(np.ceil(min_overlap_frac * len(log_t_i))), int(min_overlap_points)),
    )

    def cost(log_phi):
        shifted = log_t_i - log_phi
        in_range = (shifted >= log_t_ref.min()) & (shifted <= log_t_ref.max())
        n_overlap = int(in_range.sum())
        if n_overlap < n_required:
            return np.inf
        J_ref_at_shifted = ref_interp(shifted[in_range])
        err = J_i[in_range] - J_ref_at_shifted
        if np.isnan(err).all():
            return np.inf
        rmse = float(np.sqrt(np.nanmean(err ** 2)))
        overlap_frac = n_overlap / len(log_t_i)
        overlap_penalty = 0.02 * j_scale * (1.0 / overlap_frac - 1.0)
        return rmse + overlap_penalty

    # Bracket: the shift can't exceed the total log-time span by much.
    span = max(np.ptp(log_t_ref), np.ptp(log_t_i), 1.0)
    lower, upper = -2 * span, 2 * span

    # Coarse scan first to guarantee a feasible overlap region.
    grid = np.linspace(lower, upper, 801)
    grid_cost = np.array([cost(lp) for lp in grid], dtype=float)
    finite = np.isfinite(grid_cost)
    if not finite.any():
        raise ValueError(
            "No valid overlap found for shift-factor estimation. "
            "Try reducing --min-overlap-frac or --min-overlap-points."
        )
    idx = int(np.argmin(grid_cost))
    best_grid = float(grid[idx])

    # Local continuous refinement around best coarse point.
    lo_local = max(lower, best_grid - 0.5)
    hi_local = min(upper, best_grid + 0.5)
    if hi_local <= lo_local:
        return best_grid
    res = minimize_scalar(cost, bounds=(lo_local, hi_local), method='bounded')
    if np.isfinite(res.fun):
        return float(res.x)
    return best_grid


def _find_crossing_time(values: np.ndarray, log_times: np.ndarray,
                        target: float) -> float | None:
    """
    Find the log-time at which a curve crosses a target value.

    Works for both monotonically increasing (J(t)) and decreasing (E(t))
    curves by sorting on value before searching.

    Returns interpolated log10(t) at the first crossing, or None.
    """
    order = np.argsort(values)
    v_sorted = values[order]
    lt_sorted = log_times[order]

    idx = np.searchsorted(v_sorted, target)
    if 0 < idx < len(v_sorted):
        dv = v_sorted[idx] - v_sorted[idx - 1]
        if abs(dv) < 1e-15:
            return None
        frac = (target - v_sorted[idx - 1]) / dv
        return float(lt_sorted[idx - 1] + frac * (lt_sorted[idx] - lt_sorted[idx - 1]))
    return None


def find_shift_factor_loglevel(t_ref: np.ndarray, J_ref: np.ndarray,
                               t_i: np.ndarray, J_i: np.ndarray,
                               n_levels: int = 20) -> float:
    """
    Find log10(phi_sigma) using iso-compliance level crossings.

    For each compliance level J*, find the time at which the reference and
    target curves reach that level.  The shift factor is the median
    log10(t_i / t_ref) across all levels that both curves cross.

    Works for both increasing (creep J(t)) and decreasing (relaxation E(t))
    curves via _find_crossing_time().
    """
    J_ref_range = (J_ref.min(), J_ref.max())
    J_i_range = (J_i.min(), J_i.max())

    # Overlap in J-space: levels that both curves reach
    J_lo = max(J_ref_range[0], J_i_range[0])
    J_hi = min(J_ref_range[1], J_i_range[1])

    if J_hi <= J_lo:
        return 0.0

    margin = 0.05 * (J_hi - J_lo)
    levels = np.linspace(J_lo + margin, J_hi - margin, n_levels)

    log_t_ref = np.log10(t_ref)
    log_t_i = np.log10(t_i)

    shifts = []
    for J_star in levels:
        lt_ref = _find_crossing_time(J_ref, log_t_ref, J_star)
        lt_i = _find_crossing_time(J_i, log_t_i, J_star)
        if lt_ref is not None and lt_i is not None:
            shifts.append(lt_i - lt_ref)

    if len(shifts) < 3:
        return 0.0

    return float(np.median(shifts))


def find_shift_factor_2d(t_ref: np.ndarray, J_ref: np.ndarray,
                         t_i: np.ndarray, J_i: np.ndarray,
                         min_overlap_frac: float = 0.2,
                         min_overlap_points: int = 200) -> tuple[float, float]:
    """
    Joint optimisation of horizontal shift log_a and vertical scale b.

    Returns (log_a, b) where:
        shifted_time = log10(t_i) - log_a
        J_scaled = b * J_i
    and the pair minimises RMSE(J_scaled - J_ref_interp) in the overlap
    region, with a soft regularisation that penalises b far from 1.

    Optimisation is performed in (log_a, log_b) space so that b stays
    positive and the regularisation acts symmetrically in log-space.
    """
    log_t_ref = np.log10(t_ref)
    log_t_i = np.log10(t_i)

    ref_interp = interp1d(log_t_ref, J_ref, kind='linear',
                          bounds_error=False, fill_value=np.nan)
    j_scale = max(np.nanstd(J_ref), np.nanstd(J_i), 1e-8)
    n_required = min(
        len(log_t_i),
        max(int(np.ceil(min_overlap_frac * len(log_t_i))), int(min_overlap_points)),
    )

    # Regularisation weight: penalise log(b) ≠ 0 to keep b near 1
    reg_weight = 0.1 * j_scale

    def cost(params):
        log_a, log_b = params
        b = 10.0 ** log_b
        shifted = log_t_i - log_a
        in_range = (shifted >= log_t_ref.min()) & (shifted <= log_t_ref.max())
        n_overlap = int(in_range.sum())
        if n_overlap < n_required:
            return np.inf
        J_ref_at_shifted = ref_interp(shifted[in_range])
        J_scaled = b * J_i[in_range]
        err = J_scaled - J_ref_at_shifted
        if np.isnan(err).all():
            return np.inf
        rmse = float(np.sqrt(np.nanmean(err ** 2)))
        overlap_frac = n_overlap / len(log_t_i)
        overlap_penalty = 0.02 * j_scale * (1.0 / overlap_frac - 1.0)
        reg_penalty = reg_weight * log_b ** 2
        return rmse + overlap_penalty + reg_penalty

    # Coarse 2D grid scan (b grid in log-space: 10^-0.5 ≈ 0.32 to 10^0.5 ≈ 3.2)
    span = max(np.ptp(log_t_ref), np.ptp(log_t_i), 1.0)
    log_a_grid = np.linspace(-2 * span, 2 * span, 201)
    log_b_grid = np.linspace(-0.5, 0.5, 51)

    best_cost = np.inf
    best_log_a = 0.0
    best_log_b = 0.0
    for log_a in log_a_grid:
        for log_b in log_b_grid:
            c = cost((log_a, log_b))
            if c < best_cost:
                best_cost = c
                best_log_a = log_a
                best_log_b = log_b

    if not np.isfinite(best_cost):
        raise ValueError(
            "No valid overlap found for 2D shift-factor estimation. "
            "Try reducing --min-overlap-frac or --min-overlap-points."
        )

    # Refine with Nelder-Mead from best grid point
    res = minimize(cost, x0=[best_log_a, best_log_b], method='Nelder-Mead',
                   options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': 2000})
    if np.isfinite(res.fun) and res.fun < best_cost:
        return float(res.x[0]), 10.0 ** float(res.x[1])
    return best_log_a, 10.0 ** best_log_b


def compute_shift_factors(curves: dict,
                          sigma0: float,
                          min_overlap_frac: float = 0.2,
                          min_overlap_points: int = 200,
                          vertical_shift: bool = False,
                          shift_method: str = "loglevel") -> pd.DataFrame:
    """
    Compute individual shift factors for every stress level relative
    to the reference stress sigma0.

    shift_method:
        "rmse"     — minimise interpolation RMSE in the overlap region
                     (original method; works well when curves have similar magnitudes)
        "loglevel" — iso-compliance level-crossing method
                     (robust to large magnitude differences between curves)
    When vertical_shift is True, jointly optimises horizontal (log_phi)
    and vertical (b_sigma) shift factors using 2D grid + Nelder-Mead.

    Returns DataFrame: stress_MPa, sigma_minus_sigma0, phi_sigma,
                       log_phi_sigma, b_sigma
    """
    if sigma0 not in curves:
        available = sorted(curves.keys())
        print(f"Error: Reference stress {sigma0} MPa not in data. "
              f"Available: {available}")
        sys.exit(1)

    t_ref, J_ref = curves[sigma0]
    rows = []

    for stress in sorted(curves.keys()):
        if stress == sigma0:
            log_phi = 0.0
            b_sigma = 1.0
        else:
            t_i, J_i = curves[stress]
            if vertical_shift:
                log_phi, b_sigma = find_shift_factor_2d(
                    t_ref, J_ref, t_i, J_i,
                    min_overlap_frac=min_overlap_frac,
                    min_overlap_points=min_overlap_points,
                )
            elif shift_method == "loglevel":
                log_phi = find_shift_factor_loglevel(
                    t_ref, J_ref, t_i, J_i,
                )
                b_sigma = 1.0
            else:
                log_phi = find_shift_factor(
                    t_ref, J_ref, t_i, J_i,
                    min_overlap_frac=min_overlap_frac,
                    min_overlap_points=min_overlap_points,
                )
                b_sigma = 1.0

        phi = 10.0 ** log_phi
        rows.append({
            'stress_MPa': stress,
            'sigma_minus_sigma0': stress - sigma0,
            'phi_sigma': phi,
            'log_phi_sigma': log_phi,
            'b_sigma': b_sigma,
        })
        msg = (f"  sigma = {stress:5.1f} MPa  |  "
               f"log10(phi) = {log_phi:+.4f}  |  phi = {phi:.4f}")
        if vertical_shift:
            msg += f"  |  b = {b_sigma:.4f}"
        print(msg)

    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────────────────────
# Step 3b — GP-based shift factors (mastercurves library)
# ───────────────────────────────────────────────────────────────────────────
def _downsample_logspace(t: np.ndarray, J: np.ndarray,
                         n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Downsample to n log-spaced points (GP is O(n^3))."""
    if len(t) <= n:
        return t, J
    idx = np.unique(np.round(np.linspace(0, len(t) - 1, n)).astype(int))
    return t[idx], J[idx]


def compute_shift_factors_gp(curves: dict,
                              sigma0: float,
                              vertical_shift: bool = False) -> pd.DataFrame:
    """
    Compute shift factors using the mastercurves GP backend
    (Lennon, McKinley & Swan 2022).

    Works in log10(t) vs log10(J) space.  Horizontal shifting uses
    Multiply(scale="log") and optional vertical shifting likewise.

    Returns DataFrame with same columns as compute_shift_factors():
        stress_MPa, sigma_minus_sigma0, phi_sigma, log_phi_sigma, b_sigma
    Plus an extra column: phi_uncertainty (1-sigma from GP posterior).
    """
    import warnings
    from mastercurves import MasterCurve
    from mastercurves.transforms import Multiply

    if sigma0 not in curves:
        available = sorted(curves.keys())
        print(f"Error: Reference stress {sigma0} MPa not in data. "
              f"Available: {available}")
        sys.exit(1)

    stresses = sorted(curves.keys())
    xs, ys, states = [], [], []
    for stress in stresses:
        t, J = _downsample_logspace(*curves[stress])
        xs.append(np.log10(t))
        ys.append(np.log10(J))
        states.append(stress)
        print(f"  {stress:.0f} MPa: {len(t)} points (downsampled for GP)")

    mc = MasterCurve()
    mc.add_data(xs, ys, states)
    mc.add_htransform(Multiply(bounds=(1e-4, 1e4), scale="log"))
    if vertical_shift:
        mc.add_vtransform(Multiply(bounds=(1e-2, 1e2), scale="log"))
        print("  Mode: horizontal + vertical GP shifting")
    else:
        print("  Mode: horizontal-only GP shifting")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = mc.superpose()

    if isinstance(loss, list):
        print(f"  GP loss: {[f'{l:.6f}' for l in loss]}")
    else:
        print(f"  GP loss: {loss:.6f}")

    # ---- Manual re-referencing with proper uncertainty propagation ----
    # mc.change_ref() doesn't properly propagate uncertainties:
    # the GP's implicit reference (first sorted stress) gets da=0, and
    # dividing 0 by a_ref stays 0.  Instead we save raw GP outputs and
    # re-reference manually using: d(log_phi) = sqrt((da_k/a_k)^2 +
    # (da_ref/a_ref)^2), the standard relative-uncertainty formula for a ratio.

    # Convention: mastercurves Multiply(scale="log") applies the shift as
    # x_trans = x + ln(a) where x is log10(t). So in log10 space the shift
    # magnitude is ln(a), not log10(a). The Jazouli convention is
    # log10(t_red) = log10(t) - log10(phi), giving log10(phi) = -ln(a).

    idx_ref = mc.states.index(sigma0) if sigma0 in mc.states else 0

    # Save raw GP params/uncertainties before any modification
    raw_a = [float(mc.hparams[0][i]) for i in range(len(mc.states))]
    raw_da = []
    for i in range(len(mc.states)):
        if mc.huncertainties and mc.huncertainties[0] is not None:
            raw_da.append(float(mc.huncertainties[0][i]))
        else:
            raw_da.append(0.0)

    raw_b = None
    if vertical_shift and mc.vparams:
        raw_b = [float(mc.vparams[0][i]) for i in range(len(mc.states))]

    a_ref = raw_a[idx_ref]
    da_ref = raw_da[idx_ref]
    b_ref = raw_b[idx_ref] if raw_b is not None else 1.0

    rows = []
    for i, stress in enumerate(mc.states):
        # Re-referenced horizontal shift
        a_reref = raw_a[i] / a_ref
        log_phi = -np.log(a_reref)
        phi = 10.0 ** log_phi

        # Vertical shift
        b_sigma = 1.0
        if raw_b is not None:
            b_sigma = (raw_b[i] / b_ref) ** np.log(10)

        # Uncertainty on log_phi = -ln(a_k/a_ref)
        # d(ln(ratio)) = sqrt((da_k/a_k)^2 + (da_ref/a_ref)^2)
        if stress == sigma0:
            phi_unc = 0.0
        else:
            rel_k = raw_da[i] / raw_a[i] if raw_a[i] != 0 else 0.0
            rel_ref = da_ref / a_ref if a_ref != 0 else 0.0
            phi_unc = np.sqrt(rel_k**2 + rel_ref**2)

        rows.append({
            'stress_MPa': stress,
            'sigma_minus_sigma0': stress - sigma0,
            'phi_sigma': phi,
            'log_phi_sigma': log_phi,
            'b_sigma': b_sigma,
            'phi_uncertainty': phi_unc,
        })
        msg = (f"  sigma = {stress:5.1f} MPa  |  "
               f"log10(phi) = {log_phi:+.4f}  |  phi = {phi:.4f}")
        if raw_b is not None:
            msg += f"  |  b = {b_sigma:.4f}"
        if phi_unc > 0:
            msg += f"  |  unc = {phi_unc:.4f}"
        print(msg)

    print(f"  Re-referenced to {sigma0} MPa")
    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────────────────────
# Step 4 — Fit Eq. (7) to shift factors
# ───────────────────────────────────────────────────────────────────────────
def jazouli_eq7(delta_sigma, C1, C3):
    """log phi_sigma = -C1 * (sigma - sigma0) / (C3 + (sigma - sigma0))"""
    return -C1 * delta_sigma / (C3 + delta_sigma)


def fit_shift_equation(sf_df: pd.DataFrame) -> dict:
    """
    Fit shift factor models to the data.  Tries both:
      - Jazouli Eq. (7): log(phi) = -C1 * ds / (C3 + ds)   [WLF-type]
      - Linear:          log(phi) = m * ds                   [Wang & Fancey]

    Returns dict with keys:
      model, C1, C3, pcov, m, m_err, r2_wlf, r2_linear
    """
    mask = sf_df['sigma_minus_sigma0'] != 0.0
    ds = sf_df.loc[mask, 'sigma_minus_sigma0'].values
    lp = sf_df.loc[mask, 'log_phi_sigma'].values

    result = {
        'model': 'linear', 'C1': 0.0, 'C3': 0.0,
        'pcov': np.array([[np.inf, 0], [0, np.inf]]),
        'm': 0.0, 'm_err': np.nan, 'r2_wlf': np.nan, 'r2_linear': np.nan,
    }

    if len(ds) < 2:
        if len(ds) == 1 and ds[0] != 0:
            result['m'] = lp[0] / ds[0]
        return result

    # --- Linear fit: log(phi) = m * (sigma - sigma0) ---
    # Force through origin (reference point is at ds=0, log_phi=0)
    m_linear = float(np.sum(ds * lp) / np.sum(ds ** 2))
    lp_pred_lin = m_linear * ds
    ss_res_lin = np.sum((lp - lp_pred_lin) ** 2)
    ss_tot = np.sum((lp - np.mean(lp)) ** 2)
    r2_linear = 1.0 - ss_res_lin / ss_tot if ss_tot > 0 else 0.0
    m_err = np.sqrt(ss_res_lin / (len(ds) - 1) / np.sum(ds ** 2)) if len(ds) > 1 else np.nan
    result['m'] = m_linear
    result['m_err'] = m_err
    result['r2_linear'] = r2_linear
    print(f"  Linear fit: m = {m_linear:.6f} +/- {m_err:.6f}  (R² = {r2_linear:.4f})")

    # --- WLF-type fit: log(phi) = -C1 * ds / (C3 + ds) ---
    if len(ds) >= 3:
        try:
            popt, pcov = curve_fit(jazouli_eq7, ds, lp, p0=[1.0, 10.0],
                                   maxfev=5000)
            C1, C3 = float(popt[0]), float(popt[1])
            lp_pred_wlf = jazouli_eq7(ds, C1, C3)
            ss_res_wlf = np.sum((lp - lp_pred_wlf) ** 2)
            r2_wlf = 1.0 - ss_res_wlf / ss_tot if ss_tot > 0 else 0.0
            result['C1'] = C1
            result['C3'] = C3
            result['pcov'] = pcov
            result['r2_wlf'] = r2_wlf

            # Check if WLF is degenerate (C3 >> stress range → linear)
            ds_range = np.ptp(ds)
            if abs(C3) > 100 * ds_range:
                print(f"  WLF fit: C3 >> stress range (degenerate linear case)")
                result['model'] = 'linear'
            elif r2_wlf > r2_linear + 0.01:
                C1_err = np.sqrt(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else np.nan
                C3_err = np.sqrt(pcov[1, 1]) if np.isfinite(pcov[1, 1]) else np.nan
                print(f"  WLF fit:  C1 = {C1:.4f} +/- {C1_err:.4f}, "
                      f"C3 = {C3:.4f} +/- {C3_err:.4f}  (R² = {r2_wlf:.4f})")
                result['model'] = 'wlf'
            else:
                print(f"  WLF fit not significantly better than linear (R² = {r2_wlf:.4f})")
                result['model'] = 'linear'
        except RuntimeError as e:
            print(f"  WLF fit failed ({e}), using linear model")
            result['model'] = 'linear'
    else:
        print(f"  Only {len(ds)} points — using linear model only")

    print(f"  Selected model: {result['model']}")
    return result


# ───────────────────────────────────────────────────────────────────────────
# Helpers — shift application & binning
# ───────────────────────────────────────────────────────────────────────────
def _apply_shifts(curves: dict, sf_df: pd.DataFrame
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Apply shift factors to compliance curves.

    Returns (all_t_red, all_J) sorted by reduced time.
    """
    phi_map = dict(zip(sf_df['stress_MPa'], sf_df['phi_sigma']))
    b_map = dict(zip(sf_df['stress_MPa'], sf_df['b_sigma'])) \
        if 'b_sigma' in sf_df.columns else {}

    all_t_red = []
    all_J = []
    for stress in sorted(curves.keys()):
        t, J = curves[stress]
        phi = phi_map.get(stress, 1.0)
        b = b_map.get(stress, 1.0)
        all_t_red.extend((t / phi).tolist())
        all_J.extend((b * J).tolist())

    all_t_red = np.array(all_t_red)
    all_J = np.array(all_J)
    order = np.argsort(all_t_red)
    return all_t_red[order], all_J[order]


def _bin_average(t_red: np.ndarray, J: np.ndarray,
                 n_bins: int = 200
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin-average shifted data on a uniform log-time grid.

    Returns (log_t_centers, J_means, J_stds) with empty bins removed.
    """
    if len(t_red) < 20:
        return np.log10(t_red), J, np.zeros_like(J)

    log_t = np.log10(t_red)
    n_bins = min(n_bins, len(t_red) // 5)
    bin_edges = np.linspace(log_t.min(), log_t.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(log_t, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_means = np.array([J[bin_idx == i].mean()
                          if np.sum(bin_idx == i) > 0 else np.nan
                          for i in range(n_bins)])
    bin_stds = np.array([J[bin_idx == i].std()
                         if np.sum(bin_idx == i) > 1 else 0.0
                         for i in range(n_bins)])
    valid = ~np.isnan(bin_means)
    return bin_centers[valid], bin_means[valid], bin_stds[valid]


# ───────────────────────────────────────────────────────────────────────────
# Step 5 — Construct master curve (Fig. 4)
# ───────────────────────────────────────────────────────────────────────────
def build_master_curve(curves: dict, sf_df: pd.DataFrame,
                       out_dir: Path, *,
                       ylabel: str | None = None,
                       title: str | None = None,
                       filename: str = 'tssp_master_curve.png',
                       value_col: str = 'compliance_1_per_GPa') -> pd.DataFrame:
    """
    Shift all compliance data to reduced time and plot the master curve.

    Returns DataFrame with reduced_time_sec and <value_col>.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    stresses = sorted(curves.keys())
    phi_map = dict(zip(sf_df['stress_MPa'], sf_df['phi_sigma']))
    b_map = dict(zip(sf_df['stress_MPa'], sf_df['b_sigma'])) \
        if 'b_sigma' in sf_df.columns else {}

    all_t_red = []
    all_J = []

    for stress in stresses:
        color, marker = get_stress_style(stress)
        t, J = curves[stress]
        phi = phi_map.get(stress, 1.0)
        b = b_map.get(stress, 1.0)
        t_reduced = t / phi
        J_shifted = b * J
        log_t = np.log10(t_reduced)
        ax.plot(log_t, J_shifted, '-', color=color, linewidth=2.0)
        t_m, J_m = _thin(log_t, J_shifted, n_markers=10)
        ax.plot(t_m, J_m, linestyle='none', marker=marker, color=color,
                markersize=12, markerfacecolor='none', markeredgewidth=2.0,
                label=f'{stress:g} MPa')
        all_t_red.extend(t_reduced.tolist())
        all_J.extend(J_shifted.tolist())

    # Smoothed master line via binning on a uniform log-time grid
    all_t_red = np.array(all_t_red)
    all_J = np.array(all_J)
    order = np.argsort(all_t_red)
    all_t_red = all_t_red[order]
    all_J = all_J[order]

    # --- Master curve (cyan, prominent) ---
    n_bins_mc = 500
    bin_edges_mc = np.linspace(np.log10(all_t_red.min()),
                               np.log10(all_t_red.max()), n_bins_mc + 1)
    bin_centers_mc = 0.5 * (bin_edges_mc[:-1] + bin_edges_mc[1:])
    log_all = np.log10(all_t_red)
    bin_idx_mc = np.digitize(log_all, bin_edges_mc) - 1
    bin_idx_mc = np.clip(bin_idx_mc, 0, n_bins_mc - 1)
    bin_mean_mc = np.full(n_bins_mc, np.nan)
    for i in range(n_bins_mc):
        mask = bin_idx_mc == i
        if mask.sum() > 0:
            bin_mean_mc[i] = np.mean(all_J[mask])
    valid_mc = np.isfinite(bin_mean_mc)
    ax.plot(bin_centers_mc[valid_mc], bin_mean_mc[valid_mc], '-',
            color='#00CCCC', linewidth=4.0, solid_capstyle='round',
            zorder=10)
    mc_t_m, mc_J_m = _thin(bin_centers_mc[valid_mc], bin_mean_mc[valid_mc],
                           n_markers=10)
    ax.plot(mc_t_m, mc_J_m, linestyle='none', marker='H', color='#00CCCC',
            markersize=14, markeredgewidth=2.5, markerfacecolor='none',
            label='Master Curve', zorder=10)

    ax.set_xlabel(r'$\log_{10}\;t_{\mathrm{reduced}}$ (s)')
    ax.set_ylabel(ylabel or r'Shear Creep Compliance $J(t)$ (1/GPa)')
    title_text = title or 'TSSP Master Curve --- Bulk Creep Compliance (cf. Jazouli Fig. 4)'
    ax.set_title(_wrap_title(title_text, fig, ax))
    ax.legend(fontsize=22, markerscale=1.3)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        out = out_dir / f"{Path(filename).stem}.{fmt}"
        plt.savefig(out, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()

    return pd.DataFrame({
        'reduced_time_sec': all_t_red,
        value_col: all_J,
    })


def build_relaxation_master_curve(curves: dict, sf_df: pd.DataFrame,
                                   out_dir: Path) -> pd.DataFrame:
    """
    Shift all relaxation E(t) data to reduced time and plot the master curve.

    Same logic as build_master_curve() but with E(t) axis labels.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    stresses = sorted(curves.keys())
    phi_map = dict(zip(sf_df['stress_MPa'], sf_df['phi_sigma']))
    b_map = dict(zip(sf_df['stress_MPa'], sf_df['b_sigma'])) \
        if 'b_sigma' in sf_df.columns else {}

    all_t_red = []
    all_E = []

    for stress in stresses:
        color, marker = get_stress_style(stress)
        t, E = curves[stress]
        phi = phi_map.get(stress, 1.0)
        b = b_map.get(stress, 1.0)
        t_reduced = t / phi
        E_shifted = b * E
        log_t = np.log10(t_reduced)
        ax.plot(log_t, E_shifted, '-', color=color, linewidth=2.0)
        t_m, E_m = _thin(log_t, E_shifted, n_markers=10)
        ax.plot(t_m, E_m, linestyle='none', marker=marker, color=color,
                markersize=12, markerfacecolor='none', markeredgewidth=2.0,
                label=f'{stress:g} MPa')
        all_t_red.extend(t_reduced.tolist())
        all_E.extend(E_shifted.tolist())

    all_t_red = np.array(all_t_red)
    all_E = np.array(all_E)
    order = np.argsort(all_t_red)
    all_t_red = all_t_red[order]
    all_E = all_E[order]

    ax.set_xlabel(r'$\log_{10}\;t_{reduced}$ (s)')
    ax.set_ylabel(r'$E(t) = \sigma(t) / \varepsilon_0$ (MPa)')
    ax.set_title('TSSP Master Curve — Relaxation Modulus')
    ax.legend()
    plt.tight_layout()
    out = out_dir / 'tssp_master_curve_relaxation.png'
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

    return pd.DataFrame({
        'reduced_time_sec': all_t_red,
        'relaxation_modulus_MPa': all_E,
    })


def overlay_astm_on_master(curves_tssp: dict, sf_df: pd.DataFrame,
                           fit: dict, curves_astm: dict,
                           sigma0: float, out_dir: Path,
                           astm_stresses: list[float] | None = None):
    """
    Plot TSSP master curve with ASTM 24-hr data overlaid.

    ASTM data are shifted using the TSSP shift-factor model
    (linear: log(phi) = m*(sigma - sigma0)).

    astm_stresses: if given, only include these ASTM stress levels.
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    # --- TSSP master curve as a binned average line ---
    phi_map = dict(zip(sf_df['stress_MPa'], sf_df['phi_sigma']))
    b_map = dict(zip(sf_df['stress_MPa'], sf_df['b_sigma'])) \
        if 'b_sigma' in sf_df.columns else {}

    # Collect all shifted points
    all_log_t = []
    all_J = []
    for stress in sorted(curves_tssp.keys()):
        t, J = curves_tssp[stress]
        phi = phi_map.get(stress, 1.0)
        b = b_map.get(stress, 1.0)
        t_reduced = t / phi
        J_shifted = b * J
        valid = t_reduced > 0
        all_log_t.append(np.log10(t_reduced[valid]))
        all_J.append(J_shifted[valid])

    all_log_t = np.concatenate(all_log_t)
    all_J = np.concatenate(all_J)

    # Bin into evenly-spaced log-time bins and average
    n_bins = 500
    bin_edges = np.linspace(all_log_t.min(), all_log_t.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(all_log_t, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_mean = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() > 0:
            bin_mean[i] = np.mean(all_J[mask])

    valid_bins = np.isfinite(bin_mean)
    ax.plot(bin_centers[valid_bins], bin_mean[valid_bins], '-',
            color='#00CCCC', linewidth=4.0, solid_capstyle='round',
            zorder=5)
    mc_t_m, mc_J_m = _thin(bin_centers[valid_bins], bin_mean[valid_bins],
                           n_markers=10)
    ax.plot(mc_t_m, mc_J_m, linestyle='none', marker='H', color='#00CCCC',
            markersize=14, markeredgewidth=2.5, markerfacecolor='none',
            label='TSSP Master Curve', zorder=5)

    # --- ASTM curves: raw (unshifted) compliance ---
    if astm_stresses is not None:
        stresses_show = [s for s in sorted(curves_astm.keys())
                         if s in astm_stresses]
    else:
        stresses_show = sorted(curves_astm.keys())

    for stress in stresses_show:
        color, marker = get_stress_style(stress)
        t, J = curves_astm[stress]
        log_t = np.log10(t)
        ax.plot(log_t, J, '-', color=color, linewidth=2.0)
        t_m, J_m = _thin_log(log_t, J, n_markers=10)
        ax.plot(t_m, J_m, linestyle='none', marker=marker, color=color,
                markersize=12, markerfacecolor='none', markeredgewidth=2.0,
                label=f'ASTM {stress:.1f} MPa')
        print(f"  ASTM {stress:.1f} MPa: {len(t)} pts, "
              f"t=[{t[0]:.0f}–{t[-1]:.0f}] s, J(end)={J[-1]:.3f} 1/GPa")

    ax.set_xlabel(r'$\log_{10}\;t_{\mathrm{reduced}}$ (s)')
    ax.set_ylabel(r'Shear Creep Compliance $J(t)$ (1/GPa)')
    ax.set_title(_wrap_title(
        'Bulk 1-hour TSSP Mastercurve and 24-hour Shear Creep Compliance J(t) (1/GPa) vs. Creep Time (min) (log-log)',
        fig, ax))
    ax.legend(fontsize=22, markerscale=1.3,
              handletextpad=0.4, columnspacing=1.0)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        out = out_dir / f'tssp_master_astm_overlay.{fmt}'
        plt.savefig(out, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()


# ───────────────────────────────────────────────────────────────────────────
# Step 6 — Plot shift factors (Fig. 5)
# ───────────────────────────────────────────────────────────────────────────
def plot_shift_factors(sf_df: pd.DataFrame, fit: dict,
                       sigma0: float, out_dir: Path, *,
                       filename: str = 'tssp_shift_factors.png',
                       suptitle: str | None = None):
    """Shift factor data + model fit, with optional b_sigma subplot."""
    # Format sigma0 nicely: 15.0 → "15", 17.5 → "17.5"
    s0_str = f'{sigma0:g}'

    has_b = ('b_sigma' in sf_df.columns and
             not np.allclose(sf_df['b_sigma'].values, 1.0))

    if has_b:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    else:
        fig, ax = plt.subplots(figsize=(10, 7))

    default_suptitle = (r"\textbf{Bulk TSSP Horizontal and Vertical Shift Factors vs.\ Stress (MPa)}")
    if suptitle:
        # Split on newlines: bold each line separately
        lines = suptitle.split('\n')
        bold_lines = '\n'.join(r'\textbf{' + line + r'}' for line in lines)
        fig.suptitle(bold_lines, fontsize=30, y=1.02)
    else:
        fig.suptitle(default_suptitle, fontsize=30, y=1.02)

    ds = sf_df['sigma_minus_sigma0'].values
    lp = sf_df['log_phi_sigma'].values

    has_unc = ('phi_uncertainty' in sf_df.columns and
               sf_df['phi_uncertainty'].notna().any())
    if has_unc:
        unc = sf_df['phi_uncertainty'].values
        ax.errorbar(ds, lp, yerr=unc, fmt='ko', markersize=10,
                    capsize=5, capthick=1.5, label='Data (GP)')
    else:
        ax.plot(ds, lp, 'ko', markersize=10, label='Data')

    ds_fine = np.linspace(ds.min() - 1, ds.max() + 1, 200)

    if fit['model'] == 'wlf':
        C1, C3 = fit['C1'], fit['C3']
        safe = np.abs(C3 + ds_fine) > 0.01
        lp_fit = np.full_like(ds_fine, np.nan)
        lp_fit[safe] = jazouli_eq7(ds_fine[safe], C1, C3)
        ax.plot(ds_fine, lp_fit, 'r-', linewidth=1.5,
                label=f'WLF: $C_1$={C1:.3f}, $C_3$={C3:.3f}')
    else:
        m = fit['m']
        lp_fit = m * ds_fine
        ax.plot(ds_fine, lp_fit, 'r-', linewidth=1.5,
                label=f'Linear: $m$ = {m:.4f} MPa$^{{-1}}$')

    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.set_xlabel(r'$\sigma - \sigma_0$ (MPa)')
    ax.set_ylabel(r'$\log_{10}\;\varphi_\sigma$')
    ax.set_title(rf'\textbf{{Stress Shift Factor (ref = {s0_str} MPa)}}')
    ax.legend(fontsize=20, loc='lower left')

    if has_b:
        b_vals = sf_df['b_sigma'].values
        ax2.plot(ds, b_vals, 'ks', markersize=10, label='Data')
        # Linear fit to b_sigma vs (sigma - sigma0)
        b_coeffs = np.polyfit(ds, b_vals, 1)
        b_fit = np.polyval(b_coeffs, ds_fine)
        ax2.plot(ds_fine, b_fit, 'r-', linewidth=1.5,
                 label=f'Linear: slope = {b_coeffs[0]:.4f} MPa$^{{-1}}$')
        ax2.axhline(1, color='grey', linewidth=0.5)
        ax2.axvline(0, color='grey', linewidth=0.5)
        ax2.set_xlabel(r'$\sigma - \sigma_0$ (MPa)')
        ax2.set_ylabel(r'$b_\sigma$')
        ax2.set_title(rf'\textbf{{Vertical Shift $\bm{{b_\sigma}}$ (ref = {s0_str} MPa)}}')
        ax2.legend(fontsize=20, loc='best')

    plt.tight_layout()
    for fmt in ("png", "pdf"):
        out = out_dir / f"{Path(filename).stem}.{fmt}"
        plt.savefig(out, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()


# ───────────────────────────────────────────────────────────────────────────
# Step 6b — Compare master curves at different reference stresses
# ───────────────────────────────────────────────────────────────────────────
def plot_compare_references(curves: dict, ref_stresses: list[float],
                            out_dir: Path, *,
                            shift_method: str = "loglevel",
                            min_overlap_frac: float = 0.2,
                            min_overlap_points: int = 200,
                            vertical_shift: bool = False,
                            use_gp: bool = False):
    """Build and overlay master curves for different reference stresses."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, sigma0 in enumerate(ref_stresses):
        color, marker = get_stress_style(sigma0)
        print(f"\n  --- ref = {sigma0:.0f} MPa ---")
        if sigma0 not in curves:
            print(f"    Skipping {sigma0} MPa — not in data")
            continue

        if use_gp:
            sf_df = compute_shift_factors_gp(
                curves, sigma0, vertical_shift=vertical_shift)
        else:
            sf_df = compute_shift_factors(
                curves, sigma0,
                min_overlap_frac=min_overlap_frac,
                min_overlap_points=min_overlap_points,
                vertical_shift=vertical_shift,
                shift_method=shift_method)

        # Collect all shifted points for this reference stress
        phi_map = dict(zip(sf_df['stress_MPa'], sf_df['phi_sigma']))
        b_map = dict(zip(sf_df['stress_MPa'], sf_df['b_sigma'])) \
            if 'b_sigma' in sf_df.columns else {}

        all_log_t, all_J = [], []
        for stress in sorted(curves.keys()):
            t, J = curves[stress]
            phi = phi_map.get(stress, 1.0)
            b = b_map.get(stress, 1.0)
            all_log_t.extend(np.log10(t / phi).tolist())
            all_J.extend((b * J).tolist())

        ax.plot(all_log_t, all_J, '-', color=color, linewidth=0.8, alpha=0.5)
        t_m, J_m = _thin(np.array(all_log_t), np.array(all_J))
        ax.plot(t_m, J_m, linestyle='none', marker=marker, color=color,
                markersize=7, markerfacecolor='none',
                label=f'ref = {sigma0:.0f} MPa')

    ax.set_xlabel(r'$\log_{10}\;t_{reduced}$ (s)')
    ax.set_ylabel(r'$J(t) = 2(1+\nu)\,\varepsilon/\sigma$ (1/GPa)')
    ax.set_title('Master Curve Comparison — Different Reference Stresses')
    ax.legend()
    plt.tight_layout()
    out = out_dir / 'tssp_master_compare_references.png'
    plt.savefig(out, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close()


# ───────────────────────────────────────────────────────────────────────────
# Step 7 — Export
# ───────────────────────────────────────────────────────────────────────────
def export_results(master_df: pd.DataFrame, sf_df: pd.DataFrame,
                   fit: dict, sigma0: float, out_dir: Path, *,
                   prefix: str = 'tssp',
                   description: str = 'Bulk creep compliance'):
    """Save CSVs and parameter summary."""
    mc_path = out_dir / f'{prefix}_master_curve.csv'
    master_df.to_csv(mc_path, index=False)
    print(f"Saved: {mc_path}")

    sf_path = out_dir / f'{prefix}_shift_factors.csv'
    sf_df.to_csv(sf_path, index=False)
    print(f"Saved: {sf_path}")

    has_b = ('b_sigma' in sf_df.columns and
             not np.allclose(sf_df['b_sigma'].values, 1.0))

    param_path = out_dir / f'{prefix}_parameters.txt'
    with open(param_path, 'w') as f:
        f.write("TSSP Master Curve Parameters\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"{description}:\n")
        if has_b:
            f.write("  J_master(t_red) = b_sigma * J(t / a_sigma)\n\n")
        else:
            f.write("  J(t) = 2(1 + nu) * eps(t) / sigma(t)  [1/GPa]\n\n")

        f.write(f"Reference stress sigma0 = {sigma0:.1f} MPa\n\n")

        if fit['model'] == 'wlf':
            pcov = fit['pcov']
            C1, C3 = fit['C1'], fit['C3']
            C1_err = np.sqrt(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else np.nan
            C3_err = np.sqrt(pcov[1, 1]) if np.isfinite(pcov[1, 1]) else np.nan
            f.write("Shift factor model: WLF (Jazouli et al. 2005, Eq. 7)\n")
            f.write("  log10(phi) = -C1 * (sigma - sigma0) / "
                    "(C3 + (sigma - sigma0))\n\n")
            f.write(f"  C1 = {C1:.6f}")
            if np.isfinite(C1_err):
                f.write(f"  +/- {C1_err:.6f}")
            f.write("\n")
            f.write(f"  C3 = {C3:.6f}")
            if np.isfinite(C3_err):
                f.write(f"  +/- {C3_err:.6f}")
            f.write("\n")
            if not np.isnan(fit['r2_wlf']):
                f.write(f"  R² = {fit['r2_wlf']:.4f}\n")
        else:
            m = fit['m']
            m_err = fit['m_err']
            f.write("Shift factor model: Linear (Wang & Fancey 2017)\n")
            f.write("  log10(phi) = m * (sigma - sigma0)\n\n")
            f.write(f"  m = {m:.6f}")
            if np.isfinite(m_err):
                f.write(f"  +/- {m_err:.6f}")
            f.write(f"  MPa^-1\n")
            if not np.isnan(fit['r2_linear']):
                f.write(f"  R² = {fit['r2_linear']:.4f}\n")

        if has_b:
            f.write("\nVertical shift: enabled (b_sigma scales compliance)\n")
        f.write("\n")

        f.write("Individual shift factors:\n")
        if has_b:
            f.write(f"  {'Stress':>8s}  {'ds':>8s}  {'log phi':>10s}  "
                    f"{'phi':>10s}  {'b_sigma':>10s}\n")
            f.write(f"  {'(MPa)':>8s}  {'(MPa)':>8s}  {'':>10s}  "
                    f"{'':>10s}  {'':>10s}\n")
            f.write("  " + "-" * 54 + "\n")
            for _, row in sf_df.iterrows():
                f.write(f"  {row['stress_MPa']:8.1f}  "
                        f"{row['sigma_minus_sigma0']:+8.1f}  "
                        f"{row['log_phi_sigma']:+10.4f}  "
                        f"{row['phi_sigma']:10.4f}  "
                        f"{row['b_sigma']:10.4f}\n")
        else:
            f.write(f"  {'Stress':>8s}  {'ds':>8s}  {'log phi':>10s}  {'phi':>10s}\n")
            f.write(f"  {'(MPa)':>8s}  {'(MPa)':>8s}  {'':>10s}  {'':>10s}\n")
            f.write("  " + "-" * 42 + "\n")
            for _, row in sf_df.iterrows():
                f.write(f"  {row['stress_MPa']:8.1f}  "
                        f"{row['sigma_minus_sigma0']:+8.1f}  "
                        f"{row['log_phi_sigma']:+10.4f}  "
                        f"{row['phi_sigma']:10.4f}\n")
        f.write("\n")
        f.write(f"Master curve points: {len(master_df)}\n")
        f.write(f"Reduced time range: {master_df['reduced_time_sec'].min():.2f} – "
                f"{master_df['reduced_time_sec'].max():.2f} sec\n")

    print(f"Saved: {param_path}")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="TSSP master curve construction (Jazouli et al. 2005)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cd bulk/
  python ../src/tssp_master.py \\
      --compliance out/tssp/tssp_compliance_combined.csv \\
      --reference-stress 15

  python ../src/tssp_master.py \\
      --creep-dir out/tssp/ \\
      --reference-stress 15
        """,
    )
    parser.add_argument("--compliance", type=Path, default=None,
                        help="Path to tssp_compliance_combined.csv")
    parser.add_argument("--creep-dir", type=Path, default=None,
                        help="Directory with individual tssp_*_creep.csv files")
    parser.add_argument("--reference-stress", type=float, default=15.0,
                        help="Reference stress sigma0 in MPa (default: 15)")
    parser.add_argument("--strain-source", choices=["video", "stroke", "any"],
                        default="stroke",
                        help="Filter compliance data by strain source (default: stroke)")
    parser.add_argument("--min-overlap-frac", type=float, default=0.2,
                        help="Minimum overlap fraction for shift fitting (default: 0.2)")
    parser.add_argument("--min-overlap-points", type=int, default=200,
                        help="Minimum overlap points for shift fitting (default: 200)")
    parser.add_argument("--use-averaged", action="store_true",
                        help="Use averaged compliance (tssp_averaged_compliance.csv) "
                             "instead of individual runs")
    parser.add_argument("--vertical-shift", action="store_true",
                        help="Enable joint vertical shift factor b_sigma "
                             "(scales compliance magnitude)")
    parser.add_argument("--shift-method", choices=["loglevel", "rmse"],
                        default="loglevel",
                        help="Horizontal shift estimation method: "
                             "'loglevel' (iso-compliance level crossings, robust to "
                             "magnitude differences) or 'rmse' (interpolation RMSE "
                             "overlap). Default: loglevel")
    parser.add_argument("--use-mastercurves", action="store_true",
                        help="Use GP-based shifting from the mastercurves library "
                             "(Lennon, McKinley & Swan 2022) instead of built-in methods")
    parser.add_argument("--compare-references", type=float, nargs='+',
                        default=None, metavar="STRESS",
                        help="Compare master curves at multiple reference "
                             "stresses (e.g. --compare-references 15 18 21)")
    parser.add_argument("--data-type", choices=["creep", "relaxation"],
                        default="creep",
                        help="Data type: 'creep' for J(t) compliance, "
                             "'relaxation' for E(t) modulus (default: creep)")
    parser.add_argument("--astm-dir", type=Path, default=None,
                        help="Directory with ASTM 24-hr raw data "
                             "(e.g. bulkdata/astm/)")
    parser.add_argument("--overlay-astm", action="store_true",
                        help="Overlay ASTM 24-hr data on the TSSP master curve")
    parser.add_argument("--astm-stress", type=float, nargs='+', default=None,
                        metavar="STRESS",
                        help="Only show these ASTM stress levels in overlay "
                             "(e.g. --astm-stress 17.5)")
    parser.add_argument("--astm-only", action="store_true",
                        help="Build master curve from ASTM 24-hr data only "
                             "(requires --astm-dir)")
    parser.add_argument("--nu", type=float, default=0.43,
                        help="Poisson ratio for ASTM compliance (default: 0.43)")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom title for all plots")
    parser.add_argument("--thesis-figures", action="store_true",
                        help="Save figures to out/figures/ch3/ for thesis integration")
    parser.add_argument("--out-dir", type=Path, default=Path("out/tssp"),
                        help="Output directory (default: out/tssp)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.thesis_figures:
        from plot_style import THESIS_FIGURES_DIR
        out_dir = THESIS_FIGURES_DIR
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma0 = args.reference_stress

    apply_journal_style()

    # ---- ASTM-only mode ----
    if args.astm_only:
        if args.astm_dir is None:
            print("Error: --astm-only requires --astm-dir")
            sys.exit(1)
        print(f"\n[ASTM-only] Loading ASTM 24-hr compliance (nu={args.nu})...")
        curves = load_astm_compliance(args.astm_dir, nu=args.nu)
        if not curves:
            print("Error: No valid ASTM compliance curves loaded")
            sys.exit(1)

        print("\n[Step 2] Plotting unshifted ASTM compliance...")
        plot_unshifted(curves, out_dir,
                       title_suffix=args.title or ' ASTM 24-hr Creep Compliance',
                       filename='tssp_unshifted_astm.png')

        if args.use_mastercurves:
            print(f"\n[Step 3] Computing shift factors via mastercurves GP "
                  f"(ref = {sigma0} MPa)...")
            sf_df = compute_shift_factors_gp(
                curves, sigma0, vertical_shift=args.vertical_shift,
            )
        else:
            print(f"\n[Step 3] Computing shift factors (ref = {sigma0} MPa)...")
            sf_df = compute_shift_factors(
                curves, sigma0,
                min_overlap_frac=args.min_overlap_frac,
                min_overlap_points=args.min_overlap_points,
                vertical_shift=args.vertical_shift,
                shift_method=args.shift_method,
            )

        print("\n[Step 4] Fitting shift factor model...")
        fit = fit_shift_equation(sf_df)

        print("\n[Step 5] Constructing ASTM master curve...")
        master_df = build_master_curve(curves, sf_df, out_dir,
                                       title=args.title)

        print("\n[Step 6] Plotting shift factors...")
        plot_shift_factors(sf_df, fit, sigma0, out_dir,
                           suptitle=args.title)

        print("\n[Step 7] Exporting results...")
        export_results(master_df, sf_df, fit, sigma0, out_dir)

        print(f"\n{'='*55}")
        print("ASTM-only master curve construction complete!")
        print(f"{'='*55}")
        print(f"  Stress levels: {sorted(curves.keys())}")
        print(f"  Reference: {sigma0} MPa")
        if fit['model'] == 'wlf':
            print(f"  Model: WLF — C1 = {fit['C1']:.4f}, C3 = {fit['C3']:.4f}")
        else:
            print(f"  Model: Linear — m = {fit['m']:.4f} MPa^-1")
        print(f"  Output: {out_dir}/")
        return

    # ---- Relaxation mode ----
    if args.data_type == "relaxation":
        if args.creep_dir is None:
            print("Error: --data-type relaxation requires --creep-dir")
            sys.exit(1)

        print("\n[Step 1] Loading relaxation E(t) data...")
        curves = load_relaxation(args.creep_dir)
        if not curves:
            print("Error: No valid relaxation curves loaded")
            sys.exit(1)

        print("\n[Step 2] Plotting unshifted relaxation modulus...")
        plot_unshifted(curves, out_dir,
                       ylabel=r'$E(t) = \sigma(t) / \varepsilon_0$ (MPa)',
                       title_suffix=args.title or ' Relaxation Modulus',
                       filename='tssp_unshifted_relaxation.png')

        print(f"\n[Step 3] Computing shift factors (ref = {sigma0} MPa)...")
        sf_df = compute_shift_factors(
            curves, sigma0,
            min_overlap_frac=args.min_overlap_frac,
            min_overlap_points=args.min_overlap_points,
            vertical_shift=args.vertical_shift,
            shift_method=args.shift_method,
        )

        print("\n[Step 4] Fitting shift factor model...")
        fit = fit_shift_equation(sf_df)

        print("\n[Step 5] Constructing relaxation master curve...")
        master_df = build_relaxation_master_curve(curves, sf_df, out_dir)

        print("\n[Step 6] Plotting shift factors...")
        plot_shift_factors(sf_df, fit, sigma0, out_dir,
                           suptitle=args.title)

        print("\n[Step 7] Exporting results...")
        export_results(master_df, sf_df, fit, sigma0, out_dir)

        print(f"\n{'='*55}")
        print("Relaxation master curve construction complete!")
        print(f"{'='*55}")
        print(f"  Stress levels: {sorted(curves.keys())}")
        print(f"  Reference: {sigma0} MPa")
        if fit['model'] == 'wlf':
            print(f"  Model: WLF — C1 = {fit['C1']:.4f}, C3 = {fit['C3']:.4f}")
        else:
            print(f"  Model: Linear — m = {fit['m']:.4f} MPa^-1")
        print(f"  Output: {out_dir}/")
        return

    # ---- Normal creep mode ----
    # --- Step 1: Load compliance data ---
    print("\n[Step 1] Loading compliance data...")
    curves = load_compliance(
        args.compliance, args.creep_dir, strain_source=args.strain_source,
        use_averaged=args.use_averaged,
    )
    if not curves:
        print("Error: No valid compliance curves loaded")
        sys.exit(1)

    # --- Step 2: Plot unshifted compliance ---
    print("\n[Step 2] Plotting unshifted compliance...")
    plot_unshifted(curves, out_dir,
                   title_suffix=args.title or ' Bulk Creep Compliance')

    # --- Step 3: Compute individual shift factors ---
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
            curves,
            sigma0,
            min_overlap_frac=args.min_overlap_frac,
            min_overlap_points=args.min_overlap_points,
            vertical_shift=args.vertical_shift,
            shift_method=args.shift_method,
        )

    # --- Step 4: Fit shift factor model ---
    print("\n[Step 4] Fitting shift factor model...")
    fit = fit_shift_equation(sf_df)

    # --- Step 5: Build master curve ---
    print("\n[Step 5] Constructing master curve...")
    master_df = build_master_curve(curves, sf_df, out_dir,
                                   title=args.title)

    # --- Step 6: Plot shift factors ---
    print("\n[Step 6] Plotting shift factors...")
    plot_shift_factors(sf_df, fit, sigma0, out_dir,
                       suptitle=args.title)

    # --- Optional: Compare reference stresses ---
    if args.compare_references:
        print("\n[Step 6b] Comparing master curves at different reference "
              "stresses...")
        plot_compare_references(
            curves, args.compare_references, out_dir,
            use_gp=args.use_mastercurves,
            vertical_shift=args.vertical_shift,
            shift_method=args.shift_method,
            min_overlap_frac=args.min_overlap_frac,
            min_overlap_points=args.min_overlap_points,
        )

    # --- Optional: Overlay ASTM data ---
    if args.overlay_astm:
        if args.astm_dir is None:
            print("Warning: --overlay-astm requires --astm-dir; skipping")
        else:
            print(f"\n[Step 6c] Overlaying ASTM 24-hr data (nu={args.nu})...")
            curves_astm = load_astm_compliance(args.astm_dir, nu=args.nu)
            if curves_astm:
                overlay_astm_on_master(
                    curves, sf_df, fit, curves_astm, sigma0, out_dir,
                    astm_stresses=args.astm_stress)
            else:
                print("  No ASTM data loaded; skipping overlay")

    # --- Step 7: Export results ---
    print("\n[Step 7] Exporting results...")
    export_results(master_df, sf_df, fit, sigma0, out_dir)

    print(f"\n{'='*55}")
    print("Master curve construction complete!")
    print(f"{'='*55}")
    print(f"  Stress levels: {sorted(curves.keys())}")
    print(f"  Reference: {sigma0} MPa")
    if fit['model'] == 'wlf':
        print(f"  Model: WLF — C1 = {fit['C1']:.4f}, C3 = {fit['C3']:.4f}")
    else:
        print(f"  Model: Linear — m = {fit['m']:.4f} MPa^-1")
    print(f"  Output: {out_dir}/")


if __name__ == "__main__":
    main()
