# AM Creep Analysis — Project Summary

Viscoelastic creep analysis framework for additively manufactured Tough1500
photopolymer resin. Covers three experimental modalities (nanoindentation, bulk
compression, tensile) and a Time-Stress Superposition (TSSP) master-curve
pipeline that ties them together.

This document describes the **architecture, modules, and dependencies** of the
analysis code archived in this release. For the TSSP video-tracking pipeline
details, see [`bulk/TSSP_PROJECT.md`](bulk/TSSP_PROJECT.md). For a one-script-at-
a-time index, see [`SCRIPTS.md`](SCRIPTS.md).

---

## 0. Release Scope

This Zenodo archive contains the **post-acquisition analysis and figure-
producing code**: the shared library (`src/`), the nanoindentation CLI
(`creepycrawlies.py`), the bulk compression fitters (`src/bulk/`,
`scripts/bulk/`), the TSSP master-curve construction (`src/tssp_master.py`),
and the Poisson-ratio analysis (`bulk/ASRMD638TypeI/analyze_dogbone.py`).
See [`SCRIPTS.md`](SCRIPTS.md) for the exact per-figure script inventory.

The architecture diagrams below also document the **upstream video-tracking,
calibration GUI, and per-method fit-scripting stages** for context. Those
stages are part of the full lab pipeline but are **not archived in this
release** — they require lab-specific raw data (videos, calibration JSONs)
and instrument-specific tooling. They are described in
[`bulk/TSSP_PROJECT.md`](bulk/TSSP_PROJECT.md) for reproducibility.

> **Status note — bulk dot-tracking video pipeline (in progress).**
> The orange-dot tracking and per-frame strain extraction stage that
> produces `bulk/out/series_all.csv` (used as input to `src/tssp_sync.py`
> → `src/tssp_master.py`) is **still under active development** and is
> deliberately excluded from this v1.0.0 archive. The TSSP master-curve
> construction in this release operates on the already-extracted per-test
> compliance CSVs that the video pipeline produces; the video pipeline
> itself (HSV mask tuning, dot-label disambiguation, Poisson-ratio
> extraction from edge tracking) is being iterated on and will be
> archived in a later release once stabilized. Until then, the
> `bulk/TSSP_PROJECT.md` document is the source of truth for its current
> state, including known issues (e.g. the 21 MPa Poisson-ratio anomaly).

---

## 1. Scope

| Modality | Instrument / standard | Data location | Primary outputs |
|---|---|---|---|
| **Nanoindentation** | Bruker Hysitron TI Premier (Berkovich / conical / flat-punch / frustum tips) | `nano/<method>[-<load>]/` | Creep compliance J(t), modulus, fitted Burgers / GKV / Prony / Peng parameters |
| **Bulk compression** | Tinius Olsen / MTS load frame, custom video tracking | `bulk/TSSP/`, `bulk/bulkdata/astm/`, `bulk/video/` | Creep compliance J(t), relaxation modulus E(t), Poisson's ratio ν |
| **Tensile (ASTM D638)** | Type V (modulus screening), Type I (modulus + ν, transverse extensometer) | `ASTMD638TypeV/`, `bulk/ASRMD638TypeI/` | Young's modulus E, Poisson's ratio ν |
| **TSSP master curve** | Post-processing across stress levels | `bulk/out/tssp/` | Shift factors a_σ, WLF-style fit, master J(t) curve |
| **FEM (subproject)** | FEniCSx axisymmetric indentation | `fem_nanoindent/` | Simulated load-depth-time response for inverse fitting |

The pipeline takes raw instrument output → cleaned per-test traces → cross-run
averaged curves with SEM bands → fitted viscoelastic models → TSSP master
curve → comparison plots between modalities.

---

## 2. Architecture

### 2.1 Layered design

```
┌────────────────────────────────────────────────────────────────────────┐
│  Layer 4 — Analysis CLIs & fitting scripts                             │
│  creepycrawlies.py, scripts/, src/{nanoindent,bulk,tensile}/           │
│  ──────────  produce thesis figures / CSV summaries  ──────────        │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │ imports
┌─────────────────────────────▼──────────────────────────────────────────┐
│  Layer 3 — Domain pipelines                                            │
│  src/tssp_sync.py · src/tssp_batch.py · src/tssp_master.py             │
│  src/nanoindent/nano_master.py · src/bulk/prepare_bulk_data.py         │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │ imports
┌─────────────────────────────▼──────────────────────────────────────────┐
│  Layer 2 — Shared analysis library (src/)                              │
│  models.py · utils.py · data_utils.py · analysis_functions.py          │
│  plotting_functions.py · plot_style.py · _figure_style.py              │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │ imports
┌─────────────────────────────▼──────────────────────────────────────────┐
│  Layer 1 — Third-party scientific Python                               │
│  numpy · pandas · scipy · matplotlib · lmfit · (opencv · mastercurves) │
└────────────────────────────────────────────────────────────────────────┘
```

Every module above Layer 2 imports only from layers ≤ its own. The shared
library has no domain-specific knowledge of TSSP or nanoindentation — domain
logic lives in Layer 3.

### 2.2 Per-modality data flow

**Nanoindentation:**

```
nano/<method>/*_DYN.txt ─► data_utils.find_files
                          │
                          ▼
                    data_utils.load_and_clean_data ─► .cache/<md5>.pkl
                          │   (detect hold period, compute strain & compliance)
                          ▼
                    analysis_functions.calculate_creep_compliance
                          │   (J = ε_eff / σ_hold; probe-specific compliance via utils)
                          ▼
                    data_utils.average_data_with_bins   ── or ── average_data_no_bins
                          │   (cross-run mean, SD, n at each x; SEM in plotter)
                          ▼
                    plotting_functions.plot_data_with_stdev
                          │
                          ▼
                    models.<CreepModel>.fit  (lmfit nonlinear least squares)
                          │
                          ▼
                    out/figures/ch3/*.pdf  +  fitted_parameters.csv
```

**Bulk compression (1-hour, 24-hour, and TSSP runs):**

```
bulk/bulkdata/astm/<stress>/*.csv ─► src/bulk/prepare_bulk_data.py
                                    │   (ASTM D2990 raw → time–compliance CSV)
                                    ▼
                          src/bulk/plot_bulk_24hrs.py        scripts/bulk/plot_astm_individual.py
                          src/bulk/fit_bulk_averaged.py      src/bulk/plot_bulk_pp_300s.py
                                    │
                                    ▼
                              bulk/out/*.pdf, fitted parameters
```

**TSSP (video-tracked compression):**

```
bulk/video/{scale,highFPS,CreepFPS}/<stress>/   bulk/TSSP/<stress>/*.csv
        │                                              │
        ▼ (interactive GUIs:)                          │
   scale_calibrate.py  → scale_calibration.json        │
   dot_correct.py      → dot_positions.json            │
   deformation_start.py → deformation_start.json       │
        │                                              │
        ▼                                              │
   bulk/src/video_phase3_4.py                          │
        │   (HSV mask → dot centroid tracking)         │
        ▼                                              │
   bulk/out/series_all.csv  ──────► src/tssp_sync.py ◄─┘
                                          │   (video strain ↔ tensile force sync)
                                          ▼
                                  src/tssp_batch.py
                                          │   (loops all stress levels)
                                          ▼
                                  bulk/out/tssp/tssp_<test>.csv,
                                                 tssp_combined.csv,
                                                 tssp_compliance_combined.csv
                                          │
                                          ▼
                                  src/tssp_master.py
                                          │   (log-time shift factor optimization
                                          │    + WLF-style C₁/C₃ fit;
                                          │    optionally via `mastercurves` GPR)
                                          ▼
                                  bulk/out/tssp/{shift_factors,master_curve}*.pdf
```

**Tensile (Poisson ratio):**

```
bulk/ASRMD638TypeI/<orientation>/*.csv ─► analyze_dogbone.py
                                          │   (L_axial/w_specimen correction,
                                          │    0.1–0.5 % strain window fit)
                                          ▼
                                    ν = 0.421 ± 0.025
```

### 2.3 Caching

`src/data_utils.py` caches processed per-test data under `.cache/`. The cache
key is an MD5 of `(file paths, mod-times, processing parameters)`. The cache
is invalidated automatically when any input changes. Pass `--no-cache` to any
CLI to bypass.

### 2.4 Figure style

All thesis figures share a single style module: `src/_figure_style.py` (vendored
from a separate style repo for self-containment).
* Tol qualitative palette (16 colorblind-safe colors)
* Serif (Computer Modern) with optional LaTeX rendering
* 28/24/22/18 pt title/subtitle/label/tick hierarchy
* PDF as default save format via `save_figure()`

`src/plot_style.py` adds project-specific helpers: deterministic folder→color
maps, stress-keyed style lookups, and `apply_journal_style()` (bold variant
of the base style).

---

## 3. Core Library (`src/`)

### 3.1 `src/models.py` — Creep model library *(1185 lines)*

All models inherit from a `CreepModel` ABC with uniform `fit()`, `predict()`,
`get_parameters()`, and `get_statistics()` (R², RMSE, AIC, BIC). Fitting uses
`lmfit.Model` with smart parameter bounds and (optional) Differential
Evolution warm-starts.

| Name | Free parameters | Equation summary |
|---|---|---|
| `logarithmic` | 3 | h(t) = a·ln(t+b) + c |
| `power-law` | 3 | J(t) = J₀·(t/t₀)ⁿ |
| `burgers` | 4 | Spring + Kelvin-Voigt element + dashpot |
| `gen-kelvin-N` (N=1–9) | 2N+2 | N Kelvin-Voigt elements in series + free spring + dashpot |
| `prony-3/5/7` | 3/5/7 | Prony series with fixed log-spaced τᵢ |
| `peng-2/3` | varies | Peng stress-lock (stress-dependent activation thresholds) |
| `peng-vp` | varies | Peng + viscoplastic dashpot above yield |

Functional alternatives are also exposed for direct `lmfit` use:
`burgers_model`, `m_plus_2vk_model` … `m_plus_7vk_model`, `prony_3param`,
`generalized_kv_model`.

```python
from src.models import get_model
model = get_model('gen-kelvin-5')
model.fit(time_data, compliance_data)
print(model.get_statistics())   # {'R2': ..., 'RMSE': ..., 'AIC': ..., 'BIC': ...}
```

### 3.2 `src/utils.py` — Compliance & geometry *(621 lines)*

Probe-specific compliance calculators (closed-form solutions from the
indentation mechanics literature):

| Probe | Function | Compliance formula |
|---|---|---|
| Flat punch | `calculate_flat_punch_compliance` | J = 2R/(1−ν²) · h/P |
| Berkovich | `calculate_berkovich_compliance` | J = 4·tan(α)/[π(1−ν)F₀] · h² |
| Conical | `calculate_conical_compliance` | Same as Berkovich with user α |
| Frustum | `calculate_frustum_compliance` | Variable area A = π(R + (h−100)tanα)² |
| Dispatcher | `calculate_compliance(probe_type, …)` | Routes by `probe` string |

Plus material conversions (`modulus_to_compliance`, `youngs_to_shear_modulus`,
`bulk_modulus`) and stress helpers (`calculate_stress_from_load_area`,
`calculate_stress_from_indentation`).

### 3.3 `src/data_utils.py` — Loading, caching, averaging *(992 lines)*

* `find_files()` — glob `*_DYN.txt`, `*_AVG.txt`, `*_QS.txt`, `* DC.txt`
* `load_and_clean_data()` — tab-delimited loader, hold-period detection,
  per-test compliance, MD5 cache write/read
* `average_data_with_bins()` — linear/log/quantile-binned cross-run mean
  with SEM (`sem_by_run` for per-run pre-averaging)
* `average_data_no_bins()` — interpolate all runs onto the longest valid grid
* `extract_method_from_path()` — auto-detect test method from folder name →
  hold period via `method_config.json`

### 3.4 `src/analysis_functions.py` — Calculations *(785 lines)*

* `calculate_creep_compliance(load, displacement, probe, σ_hold, …)`
* `calculate_stress(load, area)` — units flow µN/nm² → GPa
* `calculate_creep_strain(displacement, h₀)` — auto or manual start
* `detect_hold_period(time, load)` — gradient-based plateau detection
* `compute_strain_recovery`, `fit_linear_sections` for tensile post-processing

### 3.5 `src/plotting_functions.py` — Visualization *(812 lines)*

* `plot_data_with_stdev()` — averaged trace + SEM band (SD via `--nosem`),
  dual Y-axis support, folder- and column-keyed colors, watermark
* `plot_individual_files()` — per-run scatter / line traces
* `_std_to_sem()` — divide SD by √n with NaN guard

### 3.6 Other shared modules

| Module | Purpose |
|---|---|
| `src/plot_style.py` | Folder/stress style maps, publication method labels (`'103'` → `'Method 6'`), `apply_journal_style()` |
| `src/_figure_style.py` | Vendored Tol palette + font hierarchy + `save_figure()` |
| `src/tdm_reader.py`, `src/tdm_drift.py` | National Instruments TDM/TDMS file readers (used for raw load-cell drift QC) |

---

## 4. Domain Pipelines

### 4.1 Nanoindentation — `creepycrawlies.py` *(1034 lines)*

Main CLI for everything nanoindentation. Wraps `data_utils`,
`analysis_functions`, and `plotting_functions`. Test method is auto-detected
from folder name; hold period is looked up in `method_config.json`.

```bash
# Displacement vs time (per-folder average + SEM band)
python creepycrawlies.py -d nano/103/ -i out/ -x 0 -y 1

# Berkovich creep compliance
python creepycrawlies.py -d nano/103/ -i out/ -x 0 -y compliance --calculate_compliance

# Overlay multiple stress levels
python creepycrawlies.py -d nano/103-17.5 nano/103-20 nano/103-22.5 \
    -i out/ -x 0 -y compliance --calculate_compliance --overlay --log_x --log_y

# Averaging modes
--nobin            # interpolate runs onto common grid
--sem-by-run       # SEM across per-run binned means
--nosem            # plot raw SD instead of SEM
--no-cache         # bypass .cache/
```

Per-method fitting scripts live in `src/nanoindent/` (consume the same
`models.py` library):

| Script | Purpose |
|---|---|
| `fit_flatpunch_pp_improved.py` | Flat punch + Peng stress-lock fits |
| `fit_nano_103_comparison.py` | Method 103 — M+1VK / 3VK / 5VK across stresses |
| `fit_nano_103_prony2.py` | 2-term Prony fits on Method 103 |
| `fit_nano_44_*.py` | Method 44 viscoplastic + per-stress fits |
| `fit_nano_75_comparison.py` | Method 75 QS file stress-lock comparison |
| `compare_peng_parameters.py` | Cross-method Peng parameter comparison |
| `plot_nano_103_kv_avg.py` | GKV overlays on averaged data |
| `plot_nano_103_prony2_avg.py` | Prony overlays on averaged data |
| `nano_master.py` | NI TSSP master curve (σ₀ = 17.5 MPa) |

### 4.2 Bulk compression

| Script | Purpose |
|---|---|
| `src/bulk/prepare_bulk_data.py` | Raw ASTM D2990 → time–compliance CSV |
| `src/bulk/plot_bulk_24hrs.py` | 24-hour J(t) overlay, log time axis |
| `src/bulk/plot_bulk_pp_300s.py` | Peng M+5VK fit on first 300 s |
| `src/bulk/fit_bulk_averaged.py` | Stress-lock fits on averaged data |
| `src/bulk/fit_bulk_individual.py` | Per-stress, no-averaging fits |
| `src/bulk/test_mastercurves.py` | GPR-based TSSP via `mastercurves` package |
| `scripts/bulk/plot_astm_individual.py` | Individual 1-hour J(t) traces at 9 stress levels |
| `scripts/bulk/plot_force_vs_time.py` | QC: force vs time per run |

### 4.3 TSSP video-tracking pipeline

Three CLIs under `src/` orchestrate the video → strain → compliance flow.
Run from `bulk/` so relative paths resolve.

| Stage | Script | Output |
|---|---|---|
| 1 | `src/scale_calibrate.py` | `out/scale_calibration.json` (px/mm) |
| 2 | `src/dot_correct.py` | `out/dot_positions.json` |
| 3 | `src/deformation_start.py` | `out/deformation_start.json` |
| 4 | `bulk/src/video_phase3_4.py` | `out/series_all.csv` (per-frame dot positions) |
| 5 | `src/tssp_sync.py` | `out/tssp/tssp_<test>.csv` (synced strain + J(t)) |
| 6 | `src/tssp_batch.py` | `out/tssp/tssp_combined.csv` |
| 7 | `src/plot_tssp_sync.py` | Phase plots, J(t), E(t), σ–ε figures |
| 8 | `src/tssp_master.py` *(1871 lines)* | Master curve, shift factors, WLF C₁/C₃ |

Detailed step-by-step instructions, calibration JSON formats, and dot-detection
parameters live in [`bulk/TSSP_PROJECT.md`](bulk/TSSP_PROJECT.md).

### 4.4 Tensile (Poisson ratio)

`bulk/ASRMD638TypeI/analyze_dogbone.py` extracts E and ν from ASTM D638 Type I
dogbones with a transverse extensometer. Applies the L_axial/w_specimen
geometry correction to the raw transverse slope, fits in the 0.1–0.5 % axial-
strain window, excludes warmup samples. Result: **ν = 0.421 ± 0.025** (n = 5).

### 4.5 Cross-modality

`scripts/compare_compliance.py` overlays bulk and nanoindentation compliance
at matched stresses to assess inter-modality consistency.

---

## 5. Dependencies

### 5.1 Core (required for all analysis & plotting)

```
numpy        >= 1.20
pandas       >= 1.3
scipy        >= 1.7       # interpolate, ndimage, optimize, signal, stats
matplotlib   >= 3.4
lmfit        >= 1.0
```

Install with:
```bash
pip install numpy pandas scipy matplotlib lmfit
```

### 5.2 Optional / pipeline-specific

| Package | Used by | Purpose |
|---|---|---|
| `opencv-python` (`cv2`) | TSSP calibration GUIs, video tracking | Frame I/O, HSV masking, mouse-click ROI selection |
| `mastercurves` | `src/bulk/test_mastercurves.py`, optional path in `src/tssp_master.py` | Gaussian-process-regression-based master-curve shift-factor optimization |
| `seaborn` | (some QC plots) | Convenience styling |
| `pytest`, `pytest-cov` | `fem_nanoindent/tests/`, `tests/` | Test runner |
| `jupyter` | Ad-hoc exploration | Notebook environment |

Install only what you need:
```bash
pip install opencv-python mastercurves pytest
```

### 5.3 FEM subproject (`fem_nanoindent/`) — separate environment

FEniCSx requires conda for a working install. Suggested:
```bash
conda create -n fenicsx
conda activate fenicsx
conda install -c conda-forge fenics-dolfinx mpich pyvista
pip install -r fem_nanoindent/requirements.txt
```

Additional pip-installable deps:
```
gmsh         >= 4.11      # mesh generation
mpi4py       >= 3.1
petsc4py     >= 3.18
pyvista      >= 0.40      # optional, for visualization
```

### 5.4 Standard library

`argparse`, `pathlib`, `json`, `csv`, `dataclasses`, `abc`, `typing`,
`hashlib`, `pickle`, `re`, `os`, `sys`, `math`, `copy`, `itertools`,
`logging`, `datetime`, `collections`, `textwrap`, `struct`, `xml.etree`.

### 5.5 Python version

Developed and tested on **Python 3.10+** (see `.python-version`). All `typing`
annotations use modern PEP 604 (`X | None`) syntax in newer modules.

---

## 6. Repository Layout

```
AM_Creep_Analysis/
├── creepycrawlies.py          # Nanoindentation CLI (1034 lines)
├── method_config.json         # Hold periods per nano test method
├── README.md                  # Quick start
├── SCRIPTS.md                 # Script-by-script inventory + thesis figures
├── PROJECT_SUMMARY.md         # This file
├── CITATION.cff               # Software citation metadata
├── .zenodo.json               # Zenodo deposit metadata
├── LICENSE                    # MIT
│
├── src/                       # Shared library + domain pipelines
│   ├── models.py              #   Creep model library (16 models)
│   ├── utils.py               #   Compliance + geometry calculators
│   ├── data_utils.py          #   File discovery, loading, caching, averaging
│   ├── analysis_functions.py  #   Stress / strain / compliance / drift correction
│   ├── plotting_functions.py  #   SEM bands, overlays, dual-Y
│   ├── plot_style.py          #   Project style + folder/stress maps
│   ├── _figure_style.py       #   Vendored Tol palette + save_figure helper
│   ├── tssp_sync.py           #   TSSP: single-test sync (1262 lines)
│   ├── tssp_batch.py          #   TSSP: batch all stress levels
│   ├── tssp_master.py         #   TSSP: master curve construction (1871 lines)
│   ├── plot_tssp_sync.py      #   TSSP: visualization
│   ├── scale_calibrate.py     #   GUI: ruler → px/mm
│   ├── dot_correct.py         #   GUI: click 4 dots → positions
│   ├── deformation_start.py   #   GUI: mark deformation frame
│   ├── edge_calibrate.py      #   GUI: mark specimen edges (Poisson)
│   ├── poisson_manual.py      #   GUI: manual ν measurement
│   ├── nanoindent/            #   Per-method nano fitting + nano_master.py
│   ├── bulk/                  #   Bulk compression fitting + 24h plots
│   ├── tensile/               #   ASTM D638 Type V Young's modulus
│   ├── qc/                    #   QC and validation scripts
│   └── tools/                 #   Zotero / NotebookLM / Google Drive helpers
│
├── scripts/                   # Standalone analysis CLIs
│   ├── compare_compliance.py  #   Bulk vs NI compliance comparison
│   ├── bulk/                  #   ASTM individual traces, force-time QC
│   └── …
│
├── bulk/                      # Bulk compression data + TSSP video pipeline
│   ├── TSSP/<stress>/         #   Synced tensile CSVs from MTS
│   ├── bulkdata/astm/<stress>/ #  Raw ASTM D2990 24-hour creep
│   ├── video/{scale,highFPS,CreepFPS}/<stress>/  # Calibration + tracking videos
│   ├── out/                   #   Calibration JSONs, series_all.csv, tssp/
│   ├── src/                   #   Video tracking pipeline
│   ├── ASRMD638TypeI/         #   Type I dogbone (E + ν measurement)
│   ├── TSSP_PROJECT.md        #   Pipeline-specific docs
│   └── PROJECT_SUMMARY.md
│
├── nano/                      # Nanoindentation raw data (organized by method/load)
├── ASTMD638TypeV/             # Tensile Type V data (Young's modulus screening)
├── fem_nanoindent/            # FEniCSx FEM subproject (separate env)
├── out/                       # Top-level analysis outputs (PDFs, CSVs)
├── tests/                     # Library tests (pytest)
├── old/                       # Pre-consolidation legacy scripts (archival)
└── .cache/                    # MD5-keyed processed-data cache
```

---

## 7. Data Conventions

### 7.1 Nanoindentation

* Tab-delimited text from Bruker TriboScan: `*_DYN.txt`, `*_AVG.txt`,
  `*_QS.txt`, `* DC.txt`
* Folder name encodes method + load: `103-17.5`, `44-500`, `19_hold`
* Key columns: `Test Time (s)`, `Indent Disp. (nm)`, `Indent Act. Load (µN)`,
  `Contact Area (nm^2)`, `Storage Mod. (GPa)`, `Loss Mod. (GPa)`
* Hold period auto-detected by gradient or overridden in `method_config.json`

### 7.2 Bulk (TSSP)

* New-style tensile CSV: `bulk/TSSP/<stress>/<stress>mpa-s<sample>-<type>-<run>.csv`
  – Row 0 = test ID, Row 1 = headers (Time, Force, Stroke), Row 2 = units
  (hr, N, mm), Row 3+ = data
* Raw ASTM D2990: `bulk/bulkdata/astm/<stress>/`
* Calibration JSONs in `bulk/out/`:
  `scale_calibration.json`, `deformation_start.json`, `dot_positions.json`,
  `edge_positions.json`, `poisson_manual.json`
* Camera orientation: `rot90cw` (image right = real bottom)

### 7.3 Tensile (ASTM D638)

* Type V: wide CSV, 24 specimens × 3 columns (Time, Force, Stroke).
  Gauge 7.62 mm, width 3.18 mm, thickness 3.2 mm, A = 10.18 mm²
* Type I: 5 specimens with axial + transverse extensometers, used for ν

---

## 8. Material Constants

| Property | Value | Source |
|---|---|---|
| Material | Tough1500 photopolymer resin | Formlabs datasheet |
| Specimen (bulk) | 12.7 mm dia × 25.4 mm cylinder, A = 126.68 mm² | Geometry |
| Poisson's ratio (measured) | **ν = 0.421 ± 0.025** (n = 5) | ASTM D638 Type I, this work |
| Poisson's ratio (placeholder) | 0.43 | Tough1500 datasheet |
| Young's modulus | 1.5–2.0 GPa | ASTM D638 Type V, this work |
| Berkovich half-angle | 70.3° | Tip geometry |
| Berkovich area coefficient | 24.5 h² (nm²) | Calibration |
| Flat punch area | 3.62 × 10⁻¹⁰ m² (362 µm²) | Tip geometry |

### Peng stress-lock thresholds

| Lock | Stress (MPa) |
|---|---|
| 1 | 9.9 |
| 2 | 14.9 |
| 3 | 19.9 |
| 4 | 24.9 |
| 5 | 29.9 |
| Yield | 33.0 |

Model selection by stress: σ ≤ 9.9 → Burgers, 9.9–14.9 → M+2VK,
14.9–19.9 → M+3VK, etc.

---

## 9. Reproducing a Thesis Figure

1. Place raw data under `nano/<method>/`, `bulk/bulkdata/astm/<stress>/`, or
   `bulk/out/tssp/` per the script docstrings.
2. Look up the figure in `SCRIPTS.md` to find the producing script.
3. Run the script. PDFs land under `out/figures/ch3/` (nanoindent / shared) or
   `bulk/out/tssp/` (TSSP / bulk).

The exact provenance of every shaded band, fit, and shift factor is documented
in this file (Appendix A below), in `appendix_error_calculation.tex`, and in
`appendix_poisson_ratio.tex`.

---

## 10. Citation

If you use this software, please cite the Zenodo deposit. The full metadata
is in [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json).

```
Rocheville, E. J. (2026). AM Creep Analysis: Viscoelastic Creep Analysis Framework
for Additively Manufactured Polymers (Version 1.0.0) [Computer software].
Zenodo. https://doi.org/10.5281/zenodo.20128706
```

Zenodo mints a versioned DOI automatically for each GitHub release once the
repository is enabled in the Zenodo–GitHub integration (Settings → Webhooks
on Zenodo, then create a tagged release on GitHub).

---

## Appendix A — Error Calculation in Plots

All shaded bands (and error bars) shown in the averaged compliance,
displacement, and strain plots represent the **standard error of the mean
(SEM)** computed *across replicate test runs*, not within a single run. The
full procedure is implemented in `src/data_utils.py` (averaging) and
`src/plotting_functions.py` (rendering).

### A.1 Aggregating replicate runs onto a common x-grid

For a given folder of replicate tests, each run produces a time series
`(x_i, y_i)` where `x_i` is the independent variable (e.g. test time) and
`y_i` is the dependent variable (e.g. creep compliance). Because runs are
sampled at slightly different time points, the data must be aligned before
any cross-run statistic can be computed. Two alignment modes are supported:

**Mode 1 — Binned averaging** (`average_data_with_bins`, `sem_by_run=True`):
- Construct `B` bin edges across the pooled x-range (linear by default,
  log-spaced if `--log_x` is set, or quantile-based if `--adaptive` is set).
- For each run independently, group its samples by bin and take the per-bin
  mean. This yields a matrix `Y ∈ ℝ^{R × B}`, where `R` is the number of runs
  and `B` is the number of bins.

**Mode 2 — Interpolated averaging** (`average_data_no_bins`):
- Pick the longest valid run as the reference x-grid `x_ref` (after excluding
  any run whose x-span is < 50 % of the median span; truncated runs are
  dropped).
- Linearly interpolate each remaining run onto `x_ref` (NaN outside its
  native range).
- This again yields a matrix `Y ∈ ℝ^{R × B}` with `B = |x_ref|`.

In both modes, each column of `Y` contains the values of all runs at one
x-location.

### A.2 Per-x mean, standard deviation, and sample size

For each column `j` (each bin or each grid point) we compute:

$$
\bar{y}_j \;=\; \frac{1}{n_j}\sum_{i \in V_j} Y_{ij}
\qquad
s_j \;=\; \sqrt{\frac{1}{n_j}\sum_{i \in V_j} \left(Y_{ij} - \bar{y}_j\right)^2}
\qquad
n_j \;=\; |V_j|
$$

where `V_j` is the set of runs that contribute a finite value at column `j`.
The standard deviation uses the population convention (`ddof = 0`, i.e.
divide by `n_j` rather than `n_j − 1`); see `np.nanstd(..., ddof=0)` in
`average_data_with_bins`/`average_data_no_bins`. These three quantities are
stored as `<col>_mean`, `<col>_std`, and `<col>_n` on the returned DataFrame.

### A.3 Conversion from standard deviation to standard error

The plotted band is the **standard error of the mean**, obtained from the
per-column standard deviation by dividing by the square root of the run
count. This is done in `_std_to_sem` in `src/plotting_functions.py`:

$$
\mathrm{SEM}_j \;=\; \frac{s_j}{\sqrt{n_j}}
$$

When `n_j < 2` the SEM is undefined and is set to NaN (no band is drawn at
that x). The SEM characterises the uncertainty of the *across-run mean*,
i.e. how reproducibly the cohort estimates the central tendency of the
underlying response — distinct from the spread of individual runs (the SD).
SEM is the appropriate metric when the quantity of interest is the mean
compliance/displacement curve of the material rather than the run-to-run
variability itself.

### A.4 What is rendered

For each averaged trace the plot shows:
- The **line** at `ȳ_j` versus `x_j` (per-x cohort mean).
- A **shaded band** between `ȳ_j − k·SEM_j` and `ȳ_j + k·SEM_j`, where `k`
  is the optional `--error_scale` multiplier (default `k = 1`, i.e. ±1 SEM ≈
  a 68 % confidence band under approximate normality of the cohort mean).
  In free-style scatter mode (`--fs`/`--fs_lines`) the band is replaced by
  matching error bars.

Passing `--nosem` swaps the SEM band for the raw standard deviation `s_j`,
and `--error_scale K` scales the band by an arbitrary factor `K` (e.g.
`K = 1.96` for an approximate 95 % interval).

### A.5 Summary

```
Replicate runs ──▶ align (bin or interpolate) ──▶ Y ∈ ℝ^{R × B}
                                                    │
                                ┌───────────────────┼───────────────────┐
                                ▼                   ▼                   ▼
                         mean ȳ_j           std s_j (ddof=0)        n_j (≥ 2)
                                                    │
                                                    ▼
                                          SEM_j = s_j / √n_j
                                                    │
                                                    ▼
                                       plot: ȳ_j ± k·SEM_j  (k = error_scale)
```

The convention applied to every averaged figure in this work is therefore:
lines are cohort means across replicates and shaded bands are ±1 SEM of that
cohort mean, computed independently at each x-location.

---

## Appendix B — References

- Peng et al. (2015) — Nanoindentation creep of nonlinear viscoelastic
  polypropylene, *Polymer Testing* 43:38–43
- Thapa & Cheng (2024) — Flat punch viscoelastic solution
- Oliver & Pharr (1992) — Hardness and elastic modulus,
  *J. Mater. Res.* 7(6):1564–1583
- Wang & Fancey (2017) — Application of time–stress superposition to
  viscoelastic behavior of polyamide 6,6 fiber and its "true" elastic modulus,
  *Mechanics of Time-Dependent Materials*
- Jazouli et al. (2005) — Application of time–stress equivalence to nonlinear
  creep of polycarbonate, *Polymer Testing* 24(4):463–467
- Christensen (1982) — *Theory of Viscoelasticity*, Academic Press
