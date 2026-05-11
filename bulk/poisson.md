# Poisson Ratio — ASTM D638 Type I (Tough1500)

Measured Poisson's ratio from tensile dogbone tests to replace the 0.43 placeholder
used across the AM_Creep_Analysis pipeline (TSSP, edge tracking, FEM).

- Script: `ASRMD638TypeI/analyze_dogbone.py`
- Data: `ASRMD638TypeI/{0,45}/*.csv`
- Specimen dimensions: `~/Downloads/ASTM Type I - Sheet1.csv`
- Outputs: `ASRMD638TypeI/out/dogbone_results.csv` + PNGs

## Result

**ν = 0.421 ± 0.025**  (0° print, n = 5, samples p-5 through p-9)

Per-sample: 0.456, 0.418, 0.411, 0.433, 0.390. All r²_ν ≥ 0.9994.

45° samples processed but not used — we switched to video extensometer on newer tests
and the cam gauge length (~59–68 mm) exceeds the Type I parallel section (~50 mm), so
those values bridge into the fillet region.

## Why the original script was wrong

Old `fit_poisson()` regressed raw `transverse_disp` against `axial_disp` and called
the slope ν. That is **not** Poisson's ratio when the two extensometers have
different gauge references.

The transverse clip-on has a 5 mm arm span. Its blades sit on opposite specimen
edges, so the raw `transverse` column is the true specimen width change δw. But
`Ext.2(Strain)` reports `δw / 5 mm`, not `δw / w_specimen`. True transverse strain
requires dividing by the actual specimen width (~13 mm), not the arm span.

Correct formula:
```
ν = -ε_trans / ε_axial = (L_axial_gauge / w_specimen) × slope(Δtrans_disp, Δaxial_disp)
```

For Type I 0° with Ext.1 GL = 25 mm and w_specimen ≈ 13 mm, the correction factor
is ≈ 1.92×. The original values (0.21–0.26) were off by that factor.

## Fit window: axial strain, not stroke

Initially used stroke ≤ 3 mm, then tried stroke ≤ 1 mm — both problematic:

- **Stroke ≤ 3 mm** — specimen already past proportional limit (~30 MPa, ~1.8 %
  axial strain) so plastic deformation biases the slope low.
- **Stroke ≤ 1 mm** — regression has nonzero intercept (~+0.0016 mm) from
  grip take-up and clip-on seating. Offset dominates the small window, inflating
  the slope and giving unphysical ν > 0.5 on several samples.

Final window uses **axial strain in 0.1–0.5 %** (via `Ext.1(Strain)`):

- The extensometer's own zero skips grip take-up.
- Upper bound (0.5 % → ~10 MPa stress) stays well below yield.
- r²_ν jumps to ≥ 0.998 on all samples.

## Sample exclusion: warmup on p-1, p-3, p-4

Even with the clean strain window, the first three tested samples gave ν > 0.5:

| Sample | ν (0.1–0.5 % window) |
|---|---|
| p-1 | 0.578 |
| p-3 | 0.588 |
| p-4 | 0.539 |
| p-5 | 0.456 |
| p-6 | 0.418 |
| p-7 | 0.411 |
| p-8 | 0.433 |
| p-9 | 0.390 |

The break falls exactly at the tested-order transition between early and later
samples. Attributed to warmup / transverse clip-on seating drift on the first few
tests of the session. Excluded via the `EXCLUDE_SAMPLES = {0: {1, 3, 4}}` constant
in the script.

## Type I vs Type V comparison

| Metric | Type I (0°, n = 5) | Type V (n = 21) |
|---|---|---|
| E | 1902 ± 62 MPa (Ext.1 physical, 25 mm GL) | 347 ± 21 MPa (apparent, from stroke) |
| UTS | 46.8 ± 0.4 MPa | 56.9 ± 1.8 MPa |
| Area | 42 mm² | 12.8 mm² |

**E is not comparable** — Type V strain was computed from crosshead stroke, which
includes apparatus compliance dominating the true gauge deformation by ~6×. Type I
with Ext.1 gives the true material modulus (~Tough1500 datasheet 2.0 GPa). Ignoring
Type V from here on because no extensometer data exists for those tests.

**UTS gap is mostly real** — Weibull size effect (~3× smaller cross-section) plus
effectively higher strain rate on the smaller gauge.

## Apparatus compliance (side finding)

Using Type I 0° data with both stroke and Ext.1, fitting (stroke − axial) vs Force
in the 0.05–0.25 % elastic window:

```
C_apparatus = 1.84 ± 0.04 µm/N  (n = 5, CV = 2 %)
```

- C_gauge (specimen inside 25 mm ext.1) = 0.31 µm/N
- C_apparatus (everything else) = 1.84 µm/N
- C_total ≈ 2.15 µm/N

The apparatus compliance lumps together machine (frame, load cell, grips) plus
non-gauge specimen (parallel section outside ext.1, fillets, tab). Analytical
estimate splits it as roughly ~1.0 µm/N machine + ~0.8 µm/N non-gauge specimen,
but the lumped number is what you'd subtract to correct Type I stroke-only data.
Separating the two cleanly would need a stiff reference bar or a multi-GL test.

Next step (deferred): add `scripts/tensile/calculate_apparatus_compliance.py` and
run varied peak loads on Type I to verify load-independence of C_apparatus.

## Files changed in this work

- `bulk/ASRMD638TypeI/analyze_dogbone.py`:
  - Added `get_axial_gauge_length()` (auto-extracts from disp vs reported strain)
  - Fixed `fit_poisson()` formula — now applies L_axial / w_spec correction
  - Switched fit window from stroke to axial strain (`NU_STRAIN_LO=0.1`, `NU_STRAIN_HI=0.5`)
  - Added `EXCLUDE_SAMPLES` constant for warmup exclusion
  - `trans_strain` stored in results dict is now the true transverse strain
    (δw / w_specimen × 100), so the plotted slope in `poisson_strain_*.png`
    matches the reported ν value.
