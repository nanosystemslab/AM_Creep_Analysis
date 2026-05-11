# TSSP Project Summary

**Last updated:** 2026-02-11
**Working directory:** `/Users/ethan/Desktop/AM_Creep_Analysis`
**Data root:** `bulk/` (all TSSP data, videos, and outputs live here)

---

## Goal

Extract time-resolved strain from video-tracked orange dots on cylindrical compression samples, synchronize with tensile tester force/stroke data, and compute creep compliance J(t) and relaxation modulus E(t) across multiple stress levels for Time-Stress Superposition (TSSP) master curve construction.

### Test Protocol (2025+)
- 1-hour creep under constant stress + 1-hour constant-stroke relaxation
- 3 videos per test:
  1. **Scale video** (~1 sec, `.mp4`) — ruler in frame for px/mm calibration
  2. **highFPS video** (~5 min, `.mkv`, 100 fps) — high frame rate for Poisson ratio + early loading
  3. **CreepFPS video** (~2 hr, `.mkv`, 1 fps) — full creep + relaxation
- **Specimen:** 12.7 mm dia cylinder, area = 126.68 mm², Tough1500 resin (nu ~ 0.43)

---

## Data Inventory

### Tensile CSV files (`bulk/TSSP/<stress>/`)

| Stress | New-style CSV (used by pipeline) | Old CSVs | .xmak files |
|--------|----------------------------------|----------|-------------|
| **15** | `15mpa-s1-relax-1.csv`           | `15-1-test-1.csv`, `15-1-test-2.csv` | 3 |
| **18** | `18mpa-s2-relax-1.csv`           | `18-1-relax-1.csv`, `18-3-test-1.csv` | 2 |
| **21** | `21mpa-s3-relax-1.csv`           | `21-4-test-1.csv` | 3 |
| **24** | —                                | — | 1 |
| **27** | —                                | — | 1 |
| **30** | —                                | — | 1 |
| **33** | —                                | — | 1 |
| **36** | —                                | — | 1 |
| **39** | —                                | — | 1 |

CSV format: Row 0 = test ID, Row 1 = headers (Time, Force, Stroke), Row 2 = units (hr, N, mm), Row 3+ = data.

### Videos (`bulk/video/`)

| Stress | highFPS | CreepFPS | Scale |
|--------|---------|----------|-------|
| **15** | `15mpa-s1-poisson-1.mkv` | `15mpa-s1-relax-1.mkv` | `15mpa-s1-scale-1.mp4` |
| **18** | `18mpa-s2-poisson-1.mkv` | `18mpa-s2-relax-1.mkv` | `18mpa-s2-scale-1.mp4` |
| **21** | `21mpa-s3-poisson-1.mkv` | `21mpa-s3-relax-1.mkv` | `21mpa-s3-scale-1.mp4` |
| **24–39** | empty folders | empty folders | folders for 24 only |

**Naming:** `<stress>mpa-s<sample>-<type>-<run>` (e.g. `18mpa-s2-relax-1`)

### Tracked Video Data (`bulk/out/series_all.csv`)

11,152 rows. Columns: `video_type, test_folder, fps, frame_index, time_sec, detected_count, labels_ok, top_x, top_y, bottom_x, bottom_y, left_x, left_y, right_x, right_y, dv_px, dh_px`

| Test | highFPS frames | highFPS duration | creepFPS frames | creepFPS duration |
|------|---------------|------------------|-----------------|-------------------|
| 15   | 3004          | 300.3 sec        | 698             | 6970 sec          |
| 18   | 3004          | 300.3 sec        | 721             | 7200 sec          |
| 21   | 3004          | 300.3 sec        | 721             | 7200 sec          |

### Calibration Files (`bulk/out/`)

**scale_calibration.json** — px/mm from ruler videos:
- 15mpa: 32.70 px/mm
- 18mpa: 21.90 px/mm
- 21mpa: 22.00 px/mm

**deformation_start.json** — manual deformation start times from highFPS:
- 15: frame 1258, t=12.58 sec
- 18: frame 990, t=9.90 sec
- 21: frame 1020, t=10.20 sec

**dot_positions.json** — manual dot positions + ROI for each test

**poisson_manual.json** — manual Poisson ratio overrides (from `poisson_manual.py`)
- Used as fallback when auto-detection fails (e.g. nearly-square dot diamond)
- Priority: manual JSON > auto ramp-endpoint calculation

---

## Scripts

### Active Copies (use these — `src/`)

The `src/` directory at the project root has the latest versions. The `bulk/src/` copies are older (Feb 2) and should not be used for TSSP sync/batch/plot, though `bulk/src/video_*.py` and `bulk/src/validate_tracking.py` are the authoritative video processing scripts.

| Script | Location | Purpose | Run from |
|--------|----------|---------|----------|
| `video_phase3_4.py` | `bulk/src/` | Dot tracking → `series_all.csv` | `cd bulk/` |
| `video_utils.py` | `bulk/src/` | Shared detection/labeling helpers | (imported) |
| `validate_tracking.py` | `bulk/src/` | Interactive mask/tracking debugger | `cd bulk/` |
| `plot_dot_diagnostics.py` | `bulk/src/` | Dot position + mask diagnostic plots | `cd bulk/` |
| `scale_calibrate.py` | `src/` | OpenCV GUI: click 2 points → px/mm | project root |
| `dot_correct.py` | `src/` | OpenCV GUI: click 4 dots → positions | project root |
| `deformation_start.py` | `src/` | OpenCV GUI: mark deformation frame | project root |
| `poisson_manual.py` | `src/` | OpenCV GUI: mark before/after frames → nu | `cd bulk/` |
| `tssp_sync.py` | `src/` | Core sync pipeline (single test) | `cd bulk/` |
| `tssp_batch.py` | `src/` | Batch all stress levels | `cd bulk/` |
| `plot_tssp_sync.py` | `src/` | Phase plots, compliance, relaxation | `cd bulk/` |

### Pipeline Flow

```
[1] scale_calibrate.py   → out/scale_calibration.json     (manual, once per test)
[2] dot_correct.py       → out/dot_positions.json         (manual, once per test)
[3] deformation_start.py → out/deformation_start.json     (manual, once per test)
[4] video_phase3_4.py    → out/series_all.csv             (dot tracking, slow)
[5] tssp_sync.py         → out/tssp/*.csv + summaries     (sync + compliance)
[6] tssp_batch.py        → out/tssp/tssp_combined.csv     (runs [5] for all tests)
[7] plot_tssp_sync.py    → out/tssp/*.png                 (all visualization)
[8] plot_dot_diagnostics.py → out/dot_diagnostics/*.png   (QC: dot positions + masks)
```

Steps 1–3 are interactive OpenCV GUIs. Step 4 is CPU-intensive. Steps 5–8 are fast.

### Key Commands

```bash
# --- From project root ---

# Scale calibration (interactive)
python src/scale_calibrate.py --video bulk/video/scale/15mpa/15mpa-s1-scale-1.mp4 --test 15mpa

# Manual dot positions (interactive)
python src/dot_correct.py --video bulk/video/highFPS/15/15mpa-s1-poisson-1.mkv --test 15

# Manual deformation start (interactive)
python src/deformation_start.py --scan-dir bulk/video/highFPS/

# Manual Poisson ratio (with ramp timing guide)
cd bulk/
python ../src/poisson_manual.py --scan-dir video/highFPS/ \
  --tssp-dir TSSP/ --deformation-start out/deformation_start.json

# --- From bulk/ directory ---
cd bulk/

# Dot tracking (processes all videos)
python src/video_phase3_4.py --scope all --dot-positions out/dot_positions.json --frame-step 10

# Single test TSSP
python ../src/tssp_sync.py \
  --tensile TSSP/15/15mpa-s1-relax-1.csv \
  --series out/series_all.csv \
  --test 15 \
  --scale out/scale_calibration.json \
  --deformation-start out/deformation_start.json

# Batch all tests
python ../src/tssp_batch.py --auto \
  --scale out/scale_calibration.json \
  --deformation-start out/deformation_start.json \
  --poisson out/poisson_manual.json

# Optional: force stroke-based compliance when video strain is missing
python ../src/tssp_batch.py --auto \
  --scale out/scale_calibration.json \
  --deformation-start out/deformation_start.json \
  --poisson out/poisson_manual.json \
  --allow-stroke-fallback

# Plots
python ../src/plot_tssp_sync.py --dir out/tssp/

# Dot diagnostics
python src/plot_dot_diagnostics.py --mask-frames 0,100,500,3000
```

---

## Current Processing Results (2026-02-11)

### Phase Detection

| Test | Duration | Ramp start | Ramp end | Creep→Relax | Creep dur |
|------|----------|-----------|----------|-------------|-----------|
| 15   | 7204 sec (2.00 hr) | 0.100 s | 3.850 s | 3602.9 s | 60.0 min |
| 18   | 7205 sec (2.00 hr) | 0.100 s | 4.300 s | 3604.4 s | 60.0 min |
| 21   | 7205 sec (2.00 hr) | 0.100 s | 4.500 s | 3604.6 s | 60.0 min |

### Poisson Ratio (from ramp endpoints)

| Test | nu | Status |
|------|-----|--------|
| 15   | 0.416 | Reasonable (expected ~0.43) |
| 18   | 0.467 | Slightly high |
| **21** | **0.210** | **Suspicious — investigate** |

### Creep Compliance

| Test | sigma_0 (MPa) | J range (1/MPa) | Points |
|------|---------------|------------------|--------|
| 15   | 15.00 | 0.000334 – 0.001306 | 7200 |
| 18   | 18.00 | 0.000479 – 0.002105 | 7202 |
| 21   | 21.00 | 0.000341 – 0.001720 | 7201 |

### Output Files (`bulk/out/tssp/`)

Per-test: `tssp_<test>.csv`, `tssp_<test>_creep.csv`, `tssp_<test>_relaxation.csv`, `tssp_<test>_summary.txt`
Combined: `tssp_combined.csv`, `tssp_compliance_combined.csv`
Plots: `tssp_phases_*.png`, `tssp_compliance_{linear,log}.png`, `tssp_shear_compliance_{linear,log}.png`, `tssp_relaxation_modulus_{linear,log}.png`, `tssp_sync_*.png`, `tssp_sync_*_zoom.png`, `tssp_stress_strain_*.png`, `tssp_poisson_*.png`

### Diagnostic Plots (`bulk/out/dot_diagnostics/`)

Per-test/video-type: `dot_diagnostics_<test>_{high,creep}.png`, `dot_mask_<test>_{high,creep}.png`

---

## Dot Detection Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| HSV lower | (5, 120, 120) | Orange hue, moderate sat/val |
| HSV upper | (25, 255, 255) | |
| Morph kernel | 5×5 ellipse | OPEN then CLOSE |
| Min area | 30.0 px² | |
| Max area | 1% of frame | |
| Circularity min | 0.6 | |
| Orientation | rot90cw | Camera rotated: image right = real bottom |
| Labeling | top=min Y, bottom=max Y, left=min X, right=max X | In image coords, then mapped via orientation |
| ROI padding | 30 px | Around auto-detected dot bounding box |

---

## Known Issues & Debug Log

### ISSUE: 21 MPa Poisson ratio is 0.210 (expected ~0.43)
- **Status:** Open
- **Likely cause:** Dot tracking quality during ramp, or video orientation/label issue for this specific test
- **Next steps:** Check `dot_diagnostics_21_high.png` for label swaps or tracking jumps during ramp window

### ISSUE: 15 MPa creepFPS shorter than expected
- **Status:** Noted
- 698 frames × 1 fps = ~6970 sec vs expected ~7200 sec
- May be a truncated recording; 18 and 21 MPa are full 7200 sec
- Probably not critical — creep phase is 3600 sec and is fully covered

### ISSUE: Scale calibration key mismatch
- **Status:** Resolved (in tssp_sync.py)
- Scale JSON uses keys like `15mpa`, `18mpa` while test_folder is `15`, `18`
- `load_scale_calibration()` tries multiple key variants: exact, `<n>mpa`, `<n>-test`

### ISSUE: Two copies of TSSP scripts
- **Status:** Known, acceptable
- `src/` has the latest, most-featured versions (997, 300, 500 lines) — updated Feb 11
- `bulk/src/` has older copies (645, 353, 184 lines) — last updated Feb 2
- The `bulk/src/` copies are only used for video processing (`video_phase3_4.py`, `video_utils.py`, `validate_tracking.py`)
- TSSP processing uses `src/` versions, run from `bulk/` directory with `python ../src/tssp_sync.py`

---

## What's Next (Data Collection In Progress)

### Immediate
- [ ] Investigate 21 MPa Poisson ratio anomaly
- [ ] Continue collecting data for stress levels 24–39 MPa
- [ ] As new data arrives: run manual calibration tools (steps 1–3), then video tracking (step 4), then batch process (steps 6–8)

### For Each New Stress Level
1. Export tensile CSV from MTS software → `bulk/TSSP/<stress>/<stress>mpa-s<n>-relax-<run>.csv`
2. Copy videos → `bulk/video/{scale,highFPS,CreepFPS}/<stress>/` using naming convention
3. Run `scale_calibrate.py` (click ruler) → appends to `scale_calibration.json`
4. Run `dot_correct.py` (click 4 dots) → appends to `dot_positions.json`
5. Run `deformation_start.py` (mark frame) → appends to `deformation_start.json`
6. Re-run `video_phase3_4.py --scope all` → updates `series_all.csv`
7. Re-run `tssp_batch.py --auto` → updates all outputs

### Future Analysis
- [ ] Fit Prony series / generalized Kelvin-Voigt to compliance curves
- [ ] Build TSSP master curve (shift factors)
- [ ] Compare bulk TSSP compliance with nanoindentation compliance
- [ ] Relaxation modulus analysis

---

## Specimen Geometry & Constants

| Property | Value |
|----------|-------|
| Specimen diameter | 12.7 mm |
| Cross-sectional area | 126.68 mm² |
| Nominal height | 25.4 mm (1 inch) |
| Poisson's ratio (Tough1500) | 0.43 |
| Shear compliance factor | 2(1+nu) = 2.86 |
