"""
Extract Young's modulus and Poisson's ratio from ASTM D638 Type I dogbone tensile tests.

0-deg: physical extensometers for both axial (Ext.1, GL=25mm) and transverse (Ext.2, GL=5mm)
45-deg: video extensometer for axial (Ext.3/cam), physical for transverse (Ext.2, GL=5mm)

Transverse extensometer measures contraction (positive = narrowing). Its blades sit on
opposite specimen edges, so `transverse` [mm] is the actual specimen width change δw.
True transverse strain is δw / w_specimen, NOT Ext.2(Strain) (which is δw / 5 mm).

Poisson: ν = -ε_trans / ε_axial = (L_axial_gauge / w_specimen) × (Δtrans / Δaxial)
Fit window: axial strain in [NU_STRAIN_LO, NU_STRAIN_HI] %, which uses the
extensometer's own zero point and avoids both grip take-up and yielding.
Specimen dimensions from measured spreadsheet. w-prefix files excluded.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)

DIMS_CSV = "/Users/ethan/Downloads/ASTM Type I - Sheet1.csv"

STRAIN_LO = 0.05
STRAIN_HI = 0.25
NU_STRAIN_LO = 0.1
NU_STRAIN_HI = 0.5

# First three 0° tests (p-1, p-3, p-4) show ν > 0.5, attributed to warmup/seating on
# the transverse clip-on; excluded from the reported Poisson summary.
EXCLUDE_SAMPLES = {0: {1, 3, 4}, 45: set()}


def load_dimensions():
    dims = {}
    raw = pd.read_csv(DIMS_CSV, header=None)
    for orient, start_row in [(0, 2), (45, 20)]:
        for i in range(start_row + 1, start_row + 12):
            row = raw.iloc[i]
            sample = row[0]
            if pd.isna(sample) or str(sample).strip().upper() in ("AVG", "STDV", ""):
                break
            sample_num = int(sample)
            widths = [float(row[j]) for j in range(2, 5)]
            heights = [float(row[j]) for j in range(5, 8)]
            avg_w = np.mean(widths)
            avg_h = np.mean(heights)
            dims[(orient, sample_num)] = {
                "length": float(row[1]),
                "avg_width": avg_w, "avg_height": avg_h,
                "area": avg_w * avg_h,
            }
    return dims


def extract_sample_num(filename):
    name = os.path.splitext(filename)[0]
    parts = name.replace("w", "").split("-")
    return int(parts[-1])


def load_csv(path):
    df = pd.read_csv(path, skiprows=[0, 2], encoding="latin-1")
    df.columns = df.columns.str.strip().str.strip('"')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_axial_strain_col(df, orientation):
    has_cam = "Ext.3(Strain)" in df.columns
    if orientation == 45 and has_cam:
        return df["Ext.3(Strain)"].values, "Ext.3 (cam)"
    return df["Ext.1(Strain)"].values, "Ext.1 (physical)"


def get_axial_disp_col(df, orientation):
    """Return (disp_array_mm, label)."""
    has_cam = "cam" in df.columns
    if orientation == 45 and has_cam:
        return df["cam"].values, "cam"
    return df["axial"].values, "axial"


def get_axial_gauge_length(df, orientation):
    """Back out gauge length (mm) from disp vs reported strain: L = disp / (strain%/100)."""
    if orientation == 45 and "Ext.3(Strain)" in df.columns:
        disp_col, strain_col = "cam", "Ext.3(Strain)"
    else:
        disp_col, strain_col = "axial", "Ext.1(Strain)"
    m = (df[disp_col].abs() > 1e-4) & (df[strain_col].abs() > 0.01)
    if m.sum() < 5:
        return np.nan
    return float((df.loc[m, disp_col] / (df.loc[m, strain_col] / 100.0)).median())


def fit_E(stress, axial_pct, lo=STRAIN_LO, hi=STRAIN_HI):
    mask = (axial_pct >= lo) & (axial_pct <= hi) & np.isfinite(stress) & np.isfinite(axial_pct)
    if mask.sum() < 5:
        return np.nan, np.nan, mask
    strain_frac = axial_pct[mask] / 100.0
    slope, intercept, r, p, se = linregress(strain_frac, stress[mask])
    return slope, r**2, mask


def fit_poisson(axial_disp, trans_disp, axial_strain_pct, L_axial_gauge, w_specimen,
                strain_lo=NU_STRAIN_LO, strain_hi=NU_STRAIN_HI):
    """Proper Poisson: ν = (L_axial_gauge / w_specimen) × slope(Δtrans, Δaxial).

    Fit in an axial-strain window — the extensometer's own zero skips grip take-up,
    and the upper bound stays elastic. `transverse` [mm] is the true δw of the specimen.
    """
    mask = ((axial_strain_pct >= strain_lo) & (axial_strain_pct <= strain_hi) &
            np.isfinite(axial_disp) & np.isfinite(trans_disp) & np.isfinite(axial_strain_pct))
    if mask.sum() < 5 or not (np.isfinite(L_axial_gauge) and np.isfinite(w_specimen) and w_specimen > 0):
        return np.nan, np.nan, mask
    slope, intercept, r, p, se = linregress(axial_disp[mask], trans_disp[mask])
    nu = slope * L_axial_gauge / w_specimen
    return nu, r**2, mask


def analyze_file(path, orientation, dims):
    name = os.path.splitext(os.path.basename(path))[0]
    sample_num = extract_sample_num(os.path.basename(path))
    df = load_csv(path)

    dim_key = (orientation, sample_num)
    area = dims[dim_key]["area"] if dim_key in dims else np.nan
    w_spec = dims[dim_key]["avg_width"] if dim_key in dims else np.nan
    stress = df["Force"].values / area if np.isfinite(area) else df["Stress"].values

    axial_strain, axial_label = get_axial_strain_col(df, orientation)
    axial_disp, axial_disp_label = get_axial_disp_col(df, orientation)
    L_axial_gauge = get_axial_gauge_length(df, orientation)
    trans_disp = df["transverse"].values
    trans_strain_true = (trans_disp / w_spec) * 100.0 if np.isfinite(w_spec) and w_spec > 0 else np.full_like(trans_disp, np.nan)
    stroke = df["Stroke"].values

    E, r2_E, mask_E = fit_E(stress, axial_strain)
    nu, r2_nu, mask_nu = fit_poisson(axial_disp, trans_disp, axial_strain, L_axial_gauge, w_spec)
    uts = np.nanmax(stress)
    strain_break = np.nanmax(axial_strain)

    return {
        "name": name, "sample_num": sample_num,
        "orientation": orientation, "axial_source": axial_label,
        "area_mm2": area, "w_spec_mm": w_spec, "L_axial_gauge_mm": L_axial_gauge,
        "E_MPa": E, "r2_E": r2_E, "nu": nu, "r2_nu": r2_nu,
        "UTS_MPa": uts, "strain_break_pct": strain_break,
        "axial_strain": axial_strain, "trans_strain": trans_strain_true,
        "axial_disp": axial_disp, "trans_disp": trans_disp,
        "stress": stress, "stroke": stroke,
        "mask_E": mask_E, "mask_nu": mask_nu,
    }


def plot_stress_strain(results_list, orientation, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for r in results_list:
        ax.plot(r["axial_strain"], r["stress"], label=r["name"], linewidth=0.8)
    ax.set_xlabel("Axial Strain (%)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title(f"{orientation}° — Full Stress–Strain")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for r in results_list:
        mask = r["mask_E"]
        ax.plot(r["axial_strain"], r["stress"], linewidth=0.8, alpha=0.4)
        if mask.any():
            strain_fit = np.linspace(r["axial_strain"][mask].min(),
                                     r["axial_strain"][mask].max(), 50)
            stress_fit = r["E_MPa"] * (strain_fit / 100.0)
            ax.plot(strain_fit, stress_fit, "--", linewidth=1.5,
                    label=f"{r['name']} E={r['E_MPa']:.0f}")
    ax.set_xlabel("Axial Strain (%)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title(f"{orientation}° — Linear Region ({STRAIN_LO}–{STRAIN_HI}%)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, STRAIN_HI * 3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"stress_strain_{orientation}deg.png"), dpi=150)
    plt.close()


def plot_disp_vs_disp(results_list, orientation, out_dir):
    """Plot raw transverse disp (mm) vs axial disp (mm) over the ν-fit window."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in results_list:
        mask = r["mask_nu"]
        if mask.sum() < 2:
            continue
        ax_d = r["axial_disp"][mask]
        tr_d = r["trans_disp"][mask]
        ax.plot(ax_d, tr_d, linewidth=1.0, label=r["name"])

        if mask.sum() >= 5:
            slope, intercept, rr, p, se = linregress(ax_d, tr_d)
            x_fit = np.array([ax_d.min(), ax_d.max()])
            ax.plot(x_fit, slope * x_fit + intercept, "--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Axial Displacement (mm)")
    ax.set_ylabel("Transverse Displacement (mm, contraction +)")
    ax.set_title(f"{orientation}° — Raw Displacement (axial ε {NU_STRAIN_LO}–{NU_STRAIN_HI} %)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"disp_vs_disp_{orientation}deg.png"), dpi=150)
    plt.close()


def plot_poisson_strain(results_list, orientation, out_dir):
    """True ε_trans (δw / w_spec) vs axial strain; slope equals reported ν."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in results_list:
        mask = r["mask_nu"]
        if not mask.any():
            continue
        ax.plot(r["axial_strain"][mask], r["trans_strain"][mask],
                linewidth=1.0, label=f"{r['name']} ν={r['nu']:.3f}")
    ax.set_xlabel("Axial Strain (%)")
    ax.set_ylabel("Transverse Strain (%, contraction +)")
    ax.set_title(f"{orientation}° — Strain vs Strain (axial ε {NU_STRAIN_LO}–{NU_STRAIN_HI} %)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"poisson_strain_{orientation}deg.png"), dpi=150)
    plt.close()


def main():
    dims = load_dimensions()
    rows = []

    for orientation in [0, 45]:
        folder = os.path.join(BASE, str(orientation))
        if not os.path.isdir(folder):
            continue

        results_list = []
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".csv"):
                continue
            if fn.startswith("w"):
                continue  # skip broken w-prefix files
            if extract_sample_num(fn) in EXCLUDE_SAMPLES.get(orientation, set()):
                continue  # warmup/seating-affected samples
            path = os.path.join(folder, fn)
            r = analyze_file(path, orientation, dims)
            results_list.append(r)
            rows.append({
                "sample": r["name"], "sample_num": r["sample_num"],
                "orientation_deg": orientation, "axial_source": r["axial_source"],
                "area_mm2": r["area_mm2"],
                "E_MPa": r["E_MPa"], "r2_E": r["r2_E"],
                "nu": r["nu"], "r2_nu": r["r2_nu"],
                "UTS_MPa": r["UTS_MPa"], "strain_break_pct": r["strain_break_pct"],
            })

        if results_list:
            plot_stress_strain(results_list, orientation, OUT)
            plot_disp_vs_disp(results_list, orientation, OUT)
            plot_poisson_strain(results_list, orientation, OUT)

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(OUT, "dogbone_results.csv")
    df_out.to_csv(csv_path, index=False, float_format="%.4f")

    print("\n" + "=" * 100)
    print("ASTM D638 Type I — Young's Modulus & Poisson's Ratio")
    print(f"E fit: {STRAIN_LO}–{STRAIN_HI}% axial strain  |  ν fit: {NU_STRAIN_LO}–{NU_STRAIN_HI}% axial strain")
    excl_str = ", ".join(f"{o}°:{sorted(s)}" for o, s in EXCLUDE_SAMPLES.items() if s)
    print(f"Excluded: w-prefix files + warmup samples ({excl_str})")
    print("=" * 100)
    print(f"{'Sample':<12} {'Orient':>6} {'Area':>8} {'Axial Src':<16} {'E (MPa)':>10} {'r²_E':>8} "
          f"{'ν':>8} {'r²_ν':>8} {'UTS':>10} {'ε_break':>8}")
    print("-" * 100)
    for _, r in df_out.iterrows():
        E_str = f"{r['E_MPa']:>10.1f}" if np.isfinite(r['E_MPa']) else f"{'N/A':>10}"
        nu_str = f"{r['nu']:>8.3f}" if np.isfinite(r['nu']) else f"{'N/A':>8}"
        r2E_str = f"{r['r2_E']:>8.4f}" if np.isfinite(r['r2_E']) else f"{'N/A':>8}"
        r2nu_str = f"{r['r2_nu']:>8.4f}" if np.isfinite(r['r2_nu']) else f"{'N/A':>8}"
        area_str = f"{r['area_mm2']:>8.2f}" if np.isfinite(r['area_mm2']) else f"{'N/A':>8}"
        flag = " ***" if np.isfinite(r['nu']) and (r['nu'] > 0.5 or r['nu'] < 0) else ""
        print(f"{r['sample']:<12} {r['orientation_deg']:>4}°  {area_str} {r['axial_source']:<16} "
              f"{E_str} {r2E_str} "
              f"{nu_str} {r2nu_str} "
              f"{r['UTS_MPa']:>10.1f} {r['strain_break_pct']:>7.1f}%{flag}")

    print("\n" + "=" * 65)
    print("Summary by orientation")
    print("=" * 65)
    for orient in sorted(df_out["orientation_deg"].unique()):
        sub = df_out[df_out["orientation_deg"] == orient]
        good_E = sub[sub["r2_E"] >= 0.95]["E_MPa"].dropna()
        good_nu = sub[sub["r2_nu"] >= 0.95]["nu"].dropna()
        uts_vals = sub["UTS_MPa"].dropna()
        print(f"\n{orient}° (n={len(sub)}):")
        if len(good_E) > 0:
            print(f"  E   = {good_E.mean():>8.1f} ± {good_E.std():>6.1f} MPa  "
                  f"(n={len(good_E)}, range {good_E.min():.1f}–{good_E.max():.1f})")
        if len(good_nu) > 0:
            print(f"  ν   = {good_nu.mean():>8.4f} ± {good_nu.std():>6.4f}     "
                  f"(n={len(good_nu)}, range {good_nu.min():.4f}–{good_nu.max():.4f})")
        print(f"  UTS = {uts_vals.mean():>8.1f} ± {uts_vals.std():>6.1f} MPa  "
              f"(n={len(uts_vals)}, range {uts_vals.min():.1f}–{uts_vals.max():.1f})")

    print(f"\nCSV: {csv_path}")
    print(f"Plots: {OUT}/")


if __name__ == "__main__":
    main()
