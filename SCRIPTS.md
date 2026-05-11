# Scripts Manifest

This document lists every analysis and plotting script archived in this
repository, the thesis figures it produces, and how to run it. Scope is
**nanoindentation creep analysis, bulk compression creep analysis, and the
post-tracking TSSP master-curve construction**. Video tracking, calibration
GUIs, and unrelated diagram/illustration scripts are excluded.

The shared library lives in `src/`; standalone analysis CLIs live at the
project root and in `scripts/`.

---

## Shared Library (`src/`)

| Module | Purpose |
|---|---|
| `src/models.py` | 16 viscoelastic creep models — Burgers, generalized Kelvin-Voigt (1–9 elements), 3/5/7-term Prony, Peng stress-lock variants, Peng + viscoplastic dashpot. All inherit from `CreepModel` with `fit() / predict() / get_statistics()`. |
| `src/utils.py` | Compliance calculators (flat punch, Berkovich, conical, frustum), modulus ↔ compliance conversions, stress-from-load helpers. |
| `src/data_utils.py` | File discovery, raw-data loading, hold-period detection, MD5 caching, binned and interpolated cross-run averaging. |
| `src/analysis_functions.py` | Creep-strain and creep-compliance computation, drift correction, plateau detection. |
| `src/plotting_functions.py` | SEM band rendering, dual-Y overlays, per-folder coloring, watermark. |
| `src/plot_style.py` | Project-specific journal style, folder-keyed color/marker maps, stress-keyed style helpers. |
| `src/_figure_style.py` | Vendored thesis figure style (Tol palette, 18/16/14/12 pt hierarchy, `save_figure` helper). Originally an external module, vendored here for self-containment. |
| `method_config.json` | Hold-period configuration per nanoindentation test method. Consumed by `creepycrawlies.py`. |

---

## Nanoindentation Pipeline

| Script | Produces (thesis figures) | Purpose |
|---|---|---|
| `creepycrawlies.py` | 3.5.1.2.1–8 (J(t) overlays per method), 3.5.1.4.1–3 (load/displacement traces), 3.5.2.5.1 (modulus overlay), 3.5.1.3.2 (tan δ vs frequency) | Main nanoindentation CLI. Loads `*_DYN.txt` / `*_AVG.txt` / `*_QS.txt`, computes compliance for the chosen probe, applies SEM-band averaging across replicate runs, supports overlays and log axes. |
| `src/nanoindent/nano_master.py` | 3.5.2.3.2 (NI shift factors), 3.5.2.3.3 (NI master curve at σ₀ = 17.5 MPa) | Constructs Time-Stress Superposition master curve from nanoindentation compliance at multiple stress levels. |
| `scripts/compare_compliance.py` | Bulk-vs-NI compliance comparison plots | Cross-modality compliance comparison between bulk compression and nanoindentation runs at matched stresses. |

### Example invocations
```bash
# Per-method compliance overlay (Berkovich, log-log)
python creepycrawlies.py -d nano/103/ -i out/ -x 0 -y compliance \
    --calculate_compliance --log_x --log_y

# Multi-stress overlay
python creepycrawlies.py -d nano/103-17.5 nano/103-20 nano/103-22.5 \
    -i out/ -x 0 -y compliance --calculate_compliance --overlay

# NI TSSP master curve
python src/nanoindent/nano_master.py
```

---

## Bulk Compression Pipeline

| Script | Produces (thesis figures) | Purpose |
|---|---|---|
| `src/bulk/prepare_bulk_data.py` | (input prep) | Converts raw ASTM D2990 24-hour creep CSV exports into time–compliance CSVs ready for the plotters and TSSP master curve. |
| `src/bulk/plot_bulk_24hrs.py` | 3.5.2.1.1 (24-hour bulk J(t) overlay, log-time) | 24-hour bulk creep compliance overlay across stress levels (17.5–30 MPa). |
| `scripts/bulk/plot_astm_individual.py` | 3.5.2.1.1.1 (1-hour individual J(t) at nine stress levels) | Per-stress, per-run individual compliance traces from the ASTM D2990 1-hour tests (15–37.5 MPa, three runs each). |
| `scripts/bulk/plot_force_vs_time.py` | (QC / supplementary) | Force vs time per run for every TSSP test, useful for verifying load-hold quality. |

### Example invocations
```bash
# 24-hour bulk overlay
python src/bulk/plot_bulk_24hrs.py

# Individual ASTM 1-hour traces per stress
python scripts/bulk/plot_astm_individual.py

# Force vs time QC plots
python scripts/bulk/plot_force_vs_time.py
```

---

## Time-Stress Superposition (post-tracking)

The video tracking and calibration stages that produce the synchronized
strain CSVs are not archived in this release. The master-curve construction
script consumes the resulting compliance CSVs.

| Script | Produces (thesis figures) | Purpose |
|---|---|---|
| `src/tssp_master.py` | 3.5.2.2.1 (unshifted compliance, all stresses), 3.5.2.2.2 (shift factors vs stress + WLF C₁/C₃ fit), 3.5.2.2.3 (master curve at σ₀ = 15 MPa), 3.5.2.4.1 (TSSP master vs ASTM 24-hour overlay) | Constructs the bulk TSSP master curve. Reads per-stress compliance CSVs from `bulk/out/tssp/`, optimizes log-time horizontal shift factors, fits Williams–Landel–Ferry-style stress shift, and overlays the resulting master curve on independently measured 24-hour ASTM creep data. |

### Example invocation
```bash
# Bulk TSSP master curve + ASTM overlay
python src/tssp_master.py
```

---

## Poisson Ratio (Appendix C)

| Script | Produces | Purpose |
|---|---|---|
| `bulk/ASRMD638TypeI/analyze_dogbone.py` | Appendix C result: ν = 0.421 ± 0.025 (n = 5) | Extracts Young's modulus and Poisson's ratio from ASTM D638 Type I tensile dogbone tests. Applies the L_axial / w_specimen correction to the raw transverse-extensometer slope, fits in the 0.1–0.5 % axial-strain window, and excludes warmup samples. |

### Example invocation
```bash
python bulk/ASRMD638TypeI/analyze_dogbone.py
```

---

## Reproducing a Thesis Figure

1. Place the relevant raw data under `nano/<method>/`, `bulk/bulkdata/astm/<stress>/`,
   or `bulk/out/tssp/` (compliance CSVs) according to the layout described in
   the script docstrings and `PROJECT_SUMMARY.md`.
2. Identify the producing script for the figure of interest using the tables above.
3. Run that script. Output PDFs/PNGs are written under `out/` (nanoindent) or
   `bulk/out/tssp/` (bulk + TSSP).

---

## Notes on Reproducibility

- All cross-run averaged plots show **standard error of the mean (SEM)** at each
  x-location, computed across replicate runs (not within a single run). The full
  derivation is in `appendix_error_calculation.tex`.
- The Poisson ratio used downstream is **ν = 0.421** (App. C result). Earlier
  pipeline runs used the placeholder ν = 0.43 from the Tough1500 datasheet;
  the difference is well within the per-sample standard deviation of the
  measured value.
- Nanoindentation hold periods are auto-detected from the load gradient but can
  be overridden per method in `method_config.json`.
- The thesis figure style (Tol palette, 18/16/14/12 pt font hierarchy) is
  vendored in `src/_figure_style.py` so the repository is self-contained.
