Project Summary - AM_Creep_Analysis (bulk)

Goal
- Extract axial and transverse strains from four orange dots on compression samples.
- Compute Poisson ratio using only the first 5 seconds after load ramp start (sample loaded).
- Process highFPS (~60 fps) for early behavior and creepFPS (1 fps) for long-term creep.
- Output a combined CSV for all tests, plus debug artifacts for verification.

Naming Convention
- Format: `<stress>mpa-s<sample>-<type>-<run>`
- Examples: `15mpa-s1-poisson-1` (highFPS), `18mpa-s2-relax-1` (creepFPS/tensile), `18mpa-s2-scale-1` (scale)

Data Layout
- Tensile CSVs: `TSSP/<stress>/<stress>mpa-s<n>-relax-<run>.csv`
- Scale videos: `video/scale/<stress>mpa/<stress>mpa-s<n>-scale-<run>.mp4`
- High FPS videos: `video/highFPS/<stress>/<stress>mpa-s<n>-poisson-<run>.mkv`
- Creep FPS videos: `video/CreepFPS/<stress>/<stress>mpa-s<n>-relax-<run>.mkv`
- Dots are in a diamond: top/bottom are axial (vertical), left/right are transverse (horizontal).
- Camera is rotated: image right corresponds to real bottom (orientation mapping: rot90cw).
- Intended dot coordinates (mm) relative to center: L(-2,0), R(2,0), T(0,2), B(0,-2).

Key Decisions
- Use engineering strain: epsilon = (L - L0) / L0.
- Poisson ratio from transverse vs axial strain over first 5 seconds after load ramp start.
- Start of deformation detected from vertical distance time series, not from video start time.
- Scale calibration via `src/scale_calibrate.py` (OpenCV GUI: click 2 points 10mm apart on ruler).
- Manual dot position marking via `src/dot_correct.py` (OpenCV GUI: click 4 dots). Replaces auto-ROI.
- Orientation handled via mapping (rot90cw) plus optional left/right or top/bottom label swaps if the video is mirrored.
- ROI from manual dot positions (preferred) or auto-ROI from the first frame's 4 dots (with padding).

Incremental Implementation Plan
Phase 1: Video manifest + basic I/O
- Scan for .mkv in highFPS and creepFPS.
- Extract fps, duration, resolution, frame count.
- Output: out/video_manifest.csv.

Phase 2: Dot detection on a single frame
- Convert to HSV, threshold orange.
- Morphological open/close to clean noise.
- Find contours, filter by area and circularity.
- Keep best 4 blobs and label as top/bottom/left/right.
- Output: out/debug/<video>_detect.png.

Phase 3: Stable dot labeling
- Label rules: top=min y, bottom=max y, left=min x, right=max x.
- Validate on several frames to confirm stable ordering.

Phase 4: Tracking across time (highFPS)
- Per frame: detect dots, record x/y for each dot.
- Compute dv = distance(top,bottom), dh = distance(left,right).
- Output: out/series_all.csv (combined) with time, x/y, dv, dh.

Phase 5: Deformation start detection
- Smooth dv(t) with a small median filter.
- Find early baseline window (about 1s) with minimal std.
- Start = first time dv deviates by > k * std for N consecutive frames.
- Output: out/debug/<video>_start.png with dv(t) and start marker.

Phase 6: Poisson ratio (first 3s after start)
- Window [t_start, t_start + 3s].
- Axial strain: (dv - dv0) / dv0, transverse strain: (dh - dh0) / dh0.
- Fit line: epsilon_trans vs epsilon_axial, Poisson ratio = -slope.
- Output: out/poisson_summary.csv (combined, one row per video).

Phase 7: CreepFPS processing
- Repeat dot tracking at 1 fps.
- Compute axial strain vs time.
- Output: out/creep_series_<video>.csv.

Phase 8: QC and batch runner
- QC metrics: detection rate, missing frames, max jump size.
- Warnings when detection is unstable.
- Batch processing entry point for all videos.

Planned Outputs
- out/video_manifest.csv
- out/series_all.csv
- out/creep_series_<video>.csv
- out/poisson_summary.csv (combined)
- out/debug/<video>_detect.png
- out/debug/<video>_start.png

Parameters (initial defaults, to tune after first run)
- HSV orange threshold range (to be tuned from sample frames).
- Morph kernel: 3x3 or 5x5.
- Contour filters: area min/max, circularity > 0.7.
- Deformation detection: k=3, N=5 frames, baseline window ~1s.
- ROI padding: 30 px; optional manual ROI override as x,y,w,h.

Assumptions
- Camera is fixed, dots are always in frame and isolated.
- Axial compression aligns with vertical axis in image (top/bottom dots).
- Combined CSV output for now.

Open Items
- Finalize HSV thresholds from real frames.
- Decide if optional mm/px estimate is needed for reporting displacement.

---

## TSSP (Time-Stress Superposition) Implementation

### Test Protocol
- **Old (2024)**: 2-hour creep test, 2 videos per test (highFPS + creepFPS)
- **New (2025+)**: 1-hour creep + 1-hour constant-stroke relaxation, 3 videos per test:
  1. **Scale video** (~1 sec) — ruler in frame for px/mm calibration
  2. **highFPS video** (~10 min) — high frame rate for Poisson ratio
  3. **Creep+Relaxation video** (~2 hr) — full test at 1 fps

### Architecture (decoupled from dot tracking)
```
[1] video_phase3_4.py   -> out/series_all.csv      (dot tracking, unchanged)
[2] scale_calibrate.py  -> out/scale_calibration.json  (manual px/mm)
[3] tssp_sync.py        -> out/tssp/*.csv + plots   (sync + phases + compliance)
[4] tssp_batch.py       -> batch runner              (all stress levels)
[5] plot_tssp_sync.py   -> compliance + phase plots  (visualization)
```

Each step runs independently; no imports between TSSP and dot-tracking modules.

### Scripts
- `src/scale_calibrate.py` - OpenCV GUI for px/mm calibration from ruler scale videos
  - Click 2 points on ruler -> px_per_mm saved to JSON
  - Single-test or batch mode (--scan-dir)

- `src/tssp_sync.py` - Core synchronization pipeline (creep + relaxation)
  - `load_tensile_csv()` - loads CSV, converts Time hr to sec, computes stress
  - `detect_load_ramp_start()` - finds when Force > threshold
  - `detect_ramp_end()` - finds when stress reaches target level
  - `detect_creep_relaxation_transition()` - auto-detect or manual override
  - `label_phases()` - assigns 'ramp', 'creep', 'relaxation' phase labels
  - `apply_scale_calibration()` - converts px to mm using calibration
  - `compute_strain_from_video()` - engineering strain from dot distances
  - `combine_video_segments()` - merges highFPS/creepFPS with gap interpolation
  - `synchronize_data()` - aligns video strain onto tensile time grid
  - `compute_creep_compliance()` - J(t) = eps_axial / sigma_0
  - `process_test()` - full 10-step pipeline for single test

- `src/tssp_batch.py` - Batch processing for multiple stress levels
  - `find_test_pairs()` - scans tssp/ directory for matching tests
  - `batch_process()` - processes all tests, combines compliance data
  - Outputs: `tssp_combined.csv`, `tssp_compliance_combined.csv`

- `src/plot_tssp_sync.py` - Visualization
  - Phase-annotated time series (stress + strain with colored regions)
  - Creep compliance J(t) overlay (linear + log, all stress levels)
  - Relaxation modulus E(t) overlay
  - Sync verification plots with phase boundaries
  - Stress-strain and Poisson ratio plots

### Usage
```bash
# Scale calibration
python src/scale_calibrate.py --video video/scale/15-test/scale_001.mkv --test 15-test
python src/scale_calibrate.py --scan-dir video/scale/

# Process single test
python src/tssp_sync.py \
  --tensile tssp/15/15-1-test-1.csv \
  --series out/series_all.csv \
  --test 15-test \
  --scale out/scale_calibration.json \
  --poisson out/poisson_manual.json

# Process single test with manual creep duration
python src/tssp_sync.py \
  --tensile tssp/15/15-1-test-1.csv \
  --series out/series_all.csv \
  --test 15-test \
  --creep-duration 3600 \
  --poisson out/poisson_manual.json

# Batch all tests
python src/tssp_batch.py --auto \
  --scale out/scale_calibration.json \
  --poisson out/poisson_manual.json

# Optional: force stroke-based compliance when video strain is missing
python src/tssp_batch.py --auto \
  --scale out/scale_calibration.json \
  --poisson out/poisson_manual.json \
  --allow-stroke-fallback

# Plot results
python src/plot_tssp_sync.py --dir out/tssp/
```

### Data Layout
- Tensile CSV files: `tssp/<stress_level>/<test>.csv`
- Scale videos: `video/scale/<test>/scale_*.mkv`
- highFPS videos: `video/highFPS/<test>/highFPS_*.mkv`
- Creep+relaxation videos: `video/creepFPS/<test>/creepFPS_*.mkv`
- Processed video series: `out/series_all.csv`
- Scale calibration: `out/scale_calibration.json`

### Output Files (in out/tssp/)
- `tssp_<test>.csv` - Full synchronized data with phase column
- `tssp_<test>_creep.csv` - Creep phase only with compliance
- `tssp_<test>_relaxation.csv` - Relaxation phase only
- `tssp_<test>_summary.txt` - Processing summary per test
- `tssp_combined.csv` - All tests combined
- `tssp_compliance_combined.csv` - Creep compliance across stress levels
- `tssp_phases_<test>.png` - Phase-annotated time series
- `tssp_compliance_linear.png` / `tssp_compliance_log.png` - J(t) overlays
- `tssp_relaxation_modulus_linear.png` / `..._log.png` - E(t) overlays
- `tssp_sync_<test>.png` - Sync verification plots

### Specimen Geometry
- 12.7 mm diameter cylinder
- Cross-sectional area: 126.68 mm²
- Stress = Force / Area

### Known Issues
- Occasional extreme strain outliers from dot detection failures
- Phase auto-detection may need tuning for non-standard test durations
- Video tracking at 100fps is slow; use --frame-step 10 for 10fps effective
