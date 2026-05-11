#!/usr/bin/env python3
"""
Plot individual creep runs (no averaging) for both ASTM 24-hr and TSSP 1-hr data.

Produces four figures:
  1. ASTM strain vs time
  2. ASTM shear creep compliance vs time
  3. TSSP strain vs time
  4. TSSP shear creep compliance vs time

Usage (from bulk/):
    python ../scripts/bulk/plot_astm_individual.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src/ to path for plot_style
import textwrap
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
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

# Specimen constants
SPECIMEN_AREA_MM2 = np.pi * (12.7 / 2) ** 2  # 126.68 mm²
SPECIMEN_LENGTH_MM = 25.4                      # 1 inch
NU = 0.43                                      # Poisson's ratio

ASTM_DIR = Path("bulkdata/astm")
TSSP_DIR = Path("out/tssp")
OUT_DIR = Path("out/tssp")


def load_individual_runs(astm_dir: Path):
    """Load each ASTM CSV as a separate curve. Returns list of dicts."""
    runs = []
    for sd in sorted(astm_dir.iterdir()):
        if not sd.is_dir():
            continue
        dirname = sd.name.replace("MPa", "").replace("mpa", "")
        try:
            target_stress = float(dirname)
        except ValueError:
            continue

        for cf in sorted(sd.glob("*.csv")):
            try:
                with open(cf, "r") as fh:
                    lines = fh.readlines()
                header_row = next(
                    (i for i, line in enumerate(lines)
                     if "Time" in line or "time" in line.lower()),
                    0,
                )
                df = pd.read_csv(cf, skiprows=header_row)
                df.columns = df.columns.str.strip('"')
                for col in ["Time", "Force", "Stroke"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["Time", "Force", "Stroke"])

                df["stress"] = df["Force"] / SPECIMEN_AREA_MM2
                df["strain"] = df["Stroke"] / SPECIMEN_LENGTH_MM

                # 95% max-stress hold detection
                threshold = 0.95 * df["stress"].max()
                hold_mask = df["stress"] >= threshold
                if not hold_mask.any():
                    continue
                hold_start = df[hold_mask].index[0]
                df_hold = df.loc[hold_start:].copy()

                t_sec = (df_hold["Time"].values - df_hold["Time"].iloc[0]) * 3600.0
                sigma = df_hold["stress"].values
                eps_test_start = df["strain"].iloc[0]  # strain at test start (not hold start)
                eps = df_hold["strain"].values - eps_test_start

                # Shear compliance J(t) = 2(1+nu) * eps / sigma
                with np.errstate(divide="ignore", invalid="ignore"):
                    J = 2.0 * (1.0 + NU) * eps / sigma
                valid = np.isfinite(J) & (t_sec > 0)

                if valid.sum() < 10:
                    continue

                runs.append({
                    "stress": target_stress,
                    "label": cf.stem,
                    "t_sec": t_sec[valid],
                    "strain": eps[valid],
                    "J": J[valid],
                })
                print(f"  {cf.stem}: {valid.sum()} pts, "
                      f"sigma_avg={np.mean(sigma[valid]):.1f} MPa")
            except Exception as e:
                print(f"  Error reading {cf.name}: {e}")
    return runs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot individual ASTM and TSSP creep runs.")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom title for all plots")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--thesis-figures", action="store_true",
                        help="Save figures to out/figures/ch3/ for thesis integration")
    return parser.parse_args()


def main():
    global OUT_DIR
    args = parse_args()
    custom_title = args.title
    if args.thesis_figures:
        from plot_style import THESIS_FIGURES_DIR
        OUT_DIR = THESIS_FIGURES_DIR
    elif args.out_dir:
        OUT_DIR = Path(args.out_dir)
    apply_journal_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ASTM 24-hr creep data (individual runs)...")
    runs = load_individual_runs(ASTM_DIR)
    if not runs:
        print("No ASTM data found.")
        return

    astm_stresses = sorted(set(r["stress"] for r in runs))

    # Thin data for markers (every Nth point so they don't overlap)
    def thin(t, y, n_markers=40):
        step = max(1, len(t) // n_markers)
        return t[::step], y[::step]

    def thin_log(log_t, y, n_markers=20):
        """Pick n_markers evenly spaced in log10(t) space."""
        if len(log_t) < n_markers:
            return log_t, y
        targets = np.linspace(log_t.min(), log_t.max(), n_markers)
        idx = np.searchsorted(log_t, targets).clip(0, len(log_t) - 1)
        idx = np.unique(idx)
        return log_t[idx], y[idx]

    # --- Plot 1: Strain vs Time ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in runs:
        t_hr = r["t_sec"] / 3600.0
        color, marker = get_stress_style(r["stress"])
        ax.plot(t_hr, r["strain"] * 100, "-", color=color, linewidth=0.8)
        t_m, y_m = thin(t_hr, r["strain"] * 100)
        ax.plot(t_m, y_m, linestyle="none", marker=marker, color=color,
                markersize=7, markerfacecolor="none", label=r["label"])
    ax.set_xlabel("Creep Time (hr)")
    ax.set_ylabel("Creep Strain (%)")
    ax.set_title(_wrap_title(custom_title or "ASTM D2990 24-hr Creep -- Individual Runs", fig, ax))
    ax.legend(ncol=2)
    plt.tight_layout()
    out = OUT_DIR / "astm_individual_strain.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # --- Plot 2: Shear Creep Compliance vs Time ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in runs:
        t_hr = r["t_sec"] / 3600.0
        color, marker = get_stress_style(r["stress"])
        ax.plot(t_hr, r["J"] * 1000, "-", color=color, linewidth=0.8)
        t_m, y_m = thin(t_hr, r["J"] * 1000)
        ax.plot(t_m, y_m, linestyle="none", marker=marker, color=color,
                markersize=7, markerfacecolor="none", label=r["label"])
    ax.set_xlabel("Creep Time (hr)")
    ax.set_ylabel(r"Shear Creep Compliance $J(t)$ (1/GPa)")
    ax.set_title(_wrap_title(custom_title or f"ASTM D2990 24-hr Creep Compliance -- Individual Runs (nu={NU})", fig, ax))
    ax.legend(ncol=2)
    plt.tight_layout()
    out = OUT_DIR / "astm_individual_compliance.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # --- Plot 2b: ASTM Shear Creep Compliance vs log Time ---
    fig, ax = plt.subplots(figsize=(14, 9))
    for r in runs:
        color, marker = get_stress_style(r["stress"])
        t_pos = r["t_sec"][r["t_sec"] > 0]
        J_pos = (r["J"] * 1000)[r["t_sec"] > 0]
        log_t = np.log10(t_pos)
        ax.plot(log_t, J_pos, "-", color=color, linewidth=1.2)
        t_m, y_m = thin_log(log_t, J_pos, n_markers=20)
        ax.plot(t_m, y_m, linestyle="none", marker=marker, color=color,
                markersize=12, markerfacecolor="none", markeredgewidth=1.5,
                label=r["label"])
    ax.set_xlabel(r"$\log_{10}\;t$ (s)")
    ax.set_ylabel(r"Shear Creep Compliance $J(t)$ (1/GPa)")
    ax.set_title(_wrap_title(custom_title or "Bulk 24-hour Unshifted-Unaveraged Shear Creep Compliance J(t) (1/GPa) vs. Creep Time (min) (log-log): Five Stress Levels", fig, ax))
    ax.legend(ncol=2, fontsize=22, markerscale=1.3,
              handletextpad=0.4, columnspacing=1.0)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        out = OUT_DIR / f"astm_individual_compliance_log.{fmt}"
        plt.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()

    # ==================================================================
    # TSSP 1-hr individual runs
    # ==================================================================
    print("\nLoading TSSP 1-hr creep data (individual runs)...")
    tssp_runs = []
    for cf in sorted(TSSP_DIR.glob("tssp_*_run*_creep.csv")):
        df = pd.read_csv(cf)
        stress = df["target_stress_MPa"].iloc[0]
        label = df["run_label"].iloc[0]
        nu_val = df["poisson_ratio"].iloc[0] if "poisson_ratio" in df.columns else NU

        # Recompute J from stroke_mm (total displacement from test start)
        # This matches tssp_master.py's stroke_mm override and avoids stale
        # eps_stroke/J_stroke that may be zeroed to creep start.
        if "stroke_mm" not in df.columns or "stress_MPa" not in df.columns:
            continue
        valid = (df["stroke_mm"].notna() & df["time_sec_creep"].notna()
                 & df["stress_MPa"].notna())
        if valid.sum() < 10:
            continue

        factor = 2.0 * (1.0 + nu_val)
        stroke = np.abs(df.loc[valid, "stroke_mm"].values)
        sigma = np.abs(df.loc[valid, "stress_MPa"].values)
        t_sec = df.loc[valid, "time_sec_creep"].values
        eps = stroke / SPECIMEN_LENGTH_MM
        sigma_safe = np.where(sigma > 0.01, sigma, np.nan)
        J_vals = factor * eps / sigma_safe

        tssp_runs.append({
            "stress": stress,
            "label": label,
            "t_sec": t_sec,
            "strain": eps,
            "J": J_vals,
        })
        print(f"  {label}: {valid.sum()} pts, {stress:.0f} MPa")

    if not tssp_runs:
        print("No TSSP creep data found.")
        print("Done.")
        return

    # --- Plot 3: TSSP Strain vs Time ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in tssp_runs:
        t_min = r["t_sec"] / 60.0
        color, marker = get_stress_style(r["stress"])
        ax.plot(t_min, r["strain"] * 100, "-", color=color, linewidth=0.8)
        t_m, y_m = thin(t_min, r["strain"] * 100)
        ax.plot(t_m, y_m, linestyle="none", marker=marker, color=color,
                markersize=7, markerfacecolor="none", label=r["label"])
    ax.set_xlabel("Creep Time (min)")
    ax.set_ylabel("Creep Strain (%)")
    ax.set_title(_wrap_title(custom_title or "TSSP 1-hr Creep -- Individual Runs", fig, ax))
    ax.legend(ncol=2)
    plt.tight_layout()
    out = OUT_DIR / "tssp_individual_strain.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # --- Plot 4: TSSP Shear Creep Compliance vs Time ---
    fig, ax = plt.subplots(figsize=(14, 9))
    for r in tssp_runs:
        t_min = r["t_sec"] / 60.0
        color, marker = get_stress_style(r["stress"])
        ax.plot(t_min, r["J"] * 1000, "-", color=color, linewidth=1.2)
        t_m, y_m = thin(t_min, r["J"] * 1000, n_markers=20)
        ax.plot(t_m, y_m, linestyle="none", marker=marker, color=color,
                markersize=12, markerfacecolor="none", markeredgewidth=1.5,
                label=r["label"])
    ax.set_xlabel("Creep Time (min)")
    ax.set_ylabel(r"Shear Creep Compliance $J(t)$ (1/GPa)")
    ax.set_title(_wrap_title(custom_title or "Bulk 1-hour Unshifted-Unaveraged Shear Creep Compliance J(t) (1/GPa) vs. Creep Time (min) (log-log): Nine Stress Levels", fig, ax))
    ax.legend(ncol=5, fontsize=18, markerscale=1.2,
              handletextpad=0.4, columnspacing=1.0,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    for fmt in ("png", "pdf"):
        out = OUT_DIR / f"tssp_individual_compliance.{fmt}"
        plt.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()

    # --- Plot 4b: TSSP Shear Creep Compliance vs log Time ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in tssp_runs:
        color, marker = get_stress_style(r["stress"])
        t_pos = r["t_sec"][r["t_sec"] > 0]
        J_pos = (r["J"] * 1000)[r["t_sec"] > 0]
        ax.plot(np.log10(t_pos), J_pos, "-", color=color, linewidth=0.8)
        t_m, y_m = thin(np.log10(t_pos), J_pos)
        ax.plot(t_m, y_m, linestyle="none", marker=marker, color=color,
                markersize=7, markerfacecolor="none", label=r["label"])
    ax.set_xlabel(r"$\log_{10}\;t$ (s)")
    ax.set_ylabel(r"Shear Creep Compliance $J(t)$ (1/GPa)")
    ax.set_title(_wrap_title(custom_title or f"TSSP 1-hr Creep Compliance -- Individual Runs (nu={NU})", fig, ax))
    ax.legend(ncol=2)
    plt.tight_layout()
    out = OUT_DIR / "tssp_individual_compliance_log.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # ==================================================================
    # Plot 5: Averaged TSSP vs Averaged ASTM (first hour)
    # ==================================================================
    print("\nGenerating TSSP vs ASTM averaged comparison...")

    # --- Average ASTM runs per stress, trim to first hour (3600 s) ---
    from collections import defaultdict
    astm_by_stress = defaultdict(list)
    for r in runs:
        astm_by_stress[r["stress"]].append(r)

    astm_averaged = {}
    for stress in sorted(astm_by_stress):
        stress_runs = astm_by_stress[stress]
        if len(stress_runs) == 1:
            r = stress_runs[0]
            mask = r["t_sec"] <= 3600
            astm_averaged[stress] = {
                "t_sec": r["t_sec"][mask],
                "J_mean": r["J"][mask],
                "strain_mean": r["strain"][mask],
                "n": 1,
            }
        else:
            # Interpolate onto common grid (shortest run, capped at 3600 s)
            t_max = min(min(r["t_sec"][-1] for r in stress_runs), 3600.0)
            t_min = max(r["t_sec"][0] for r in stress_runs)
            n_pts = min(len(r["t_sec"]) for r in stress_runs)
            t_common = np.linspace(t_min, t_max, n_pts)
            t_common = t_common[t_common <= 3600]
            J_stack, eps_stack = [], []
            for r in stress_runs:
                J_stack.append(np.interp(t_common, r["t_sec"], r["J"]))
                eps_stack.append(np.interp(t_common, r["t_sec"], r["strain"]))
            astm_averaged[stress] = {
                "t_sec": t_common,
                "J_mean": np.mean(J_stack, axis=0),
                "strain_mean": np.mean(eps_stack, axis=0),
                "n": len(stress_runs),
            }

    # --- Load TSSP averaged compliance ---
    tssp_avg = pd.read_csv(TSSP_DIR / "tssp_averaged_compliance.csv")

    # --- Filter to <= 30 MPa ---
    astm_averaged = {s: d for s, d in astm_averaged.items() if s <= 30}
    tssp_avg = tssp_avg[tssp_avg["stress_level"] <= 30]

    # --- Collect all stress levels ---
    all_stresses = sorted(set(list(astm_averaged.keys()) +
                               list(tssp_avg["stress_level"].unique())))
    def _plot_comparison(ax, astm_data, tssp_data, y_col_astm, y_col_tssp,
                         ylabel, title):
        """Plot ASTM (solid+markers) and TSSP (dashed+markers) on same axes."""
        for stress, data in sorted(astm_data.items()):
            t_min = data["t_sec"] / 60.0
            y = data[y_col_astm]
            color, mk = get_stress_style(stress)
            ax.plot(t_min, y, "-", color=color, linewidth=1.2)
            t_m, y_m = thin(t_min, y)
            ax.plot(t_m, y_m, linestyle="none", marker=mk, color=color,
                    markersize=7, markerfacecolor="none",
                    label=f"ASTM {stress:.1f} MPa (N={data['n']})")

        for stress in sorted(tssp_data["stress_level"].unique()):
            grp = tssp_data[tssp_data["stress_level"] == stress].sort_values(
                "time_sec_creep")
            t_min = grp["time_sec_creep"].values / 60.0
            y = y_col_tssp(grp)
            color, mk = get_stress_style(stress)
            ax.plot(t_min, y, "--", color=color, linewidth=1.2)
            t_m, y_m = thin(t_min, y)
            ax.plot(t_m, y_m, linestyle="none", marker=mk, color=color,
                    markersize=7, markerfacecolor=color,
                    label=f"TSSP {stress:.0f} MPa (N={int(grp['n_runs'].iloc[0])})")

        ax.set_xlabel("Creep Time (min)")
        ax.set_ylabel(ylabel)
        ax.set_title(_wrap_title(title, fig, ax))
        ax.legend(ncol=2)

    # Add derived columns for plotting
    for s, d in astm_averaged.items():
        d["strain_mean_pct"] = d["strain_mean"] * 100
        d["J_mean_gpainv"] = d["J_mean"] * 1000

    # --- Plot 5a: Strain comparison ---
    fig, ax = plt.subplots(figsize=(10, 7))
    _plot_comparison(
        ax, astm_averaged, tssp_avg,
        y_col_astm="strain_mean_pct",
        y_col_tssp=lambda grp: (grp["J_mean"].values * grp["sigma_0_MPa"].iloc[0]
                                 / (2.0 * (1.0 + NU))) * 100,
        ylabel="Creep Strain (%)",
        title="Averaged Creep Strain: TSSP (dashed/filled) vs ASTM 1st hour (solid/open)",
    )
    plt.tight_layout()
    out = OUT_DIR / "tssp_vs_astm_strain.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # --- Plot 5b: Compliance comparison ---
    fig, ax = plt.subplots(figsize=(10, 7))
    _plot_comparison(
        ax, astm_averaged, tssp_avg,
        y_col_astm="J_mean_gpainv",
        y_col_tssp=lambda grp: grp["J_mean"].values * 1000,
        ylabel=r"Shear Creep Compliance $J(t)$ (1/GPa)",
        title=(r"Averaged Compliance: TSSP (dashed/filled) vs ASTM 1st hour (solid/open)"
               f" ($\\nu$={NU})"),
    )
    plt.tight_layout()
    out = OUT_DIR / "tssp_vs_astm_compliance.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
