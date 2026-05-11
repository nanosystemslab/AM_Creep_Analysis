# Video Processing Pipeline — Noise Audit

Review of the dot-tracking pipeline in `bulk/src/` (`video_phase3_4.py`, `video_utils.py`, `dot_tracker.py`, `visualize_mask_pipeline.py`). Measurements taken on a stationary top-dot at 21 MPa (200 frames, no load) and on 5 consecutive static frames for per-pixel codec noise.

## Measured Noise Floor

Sub-pixel std on one static dot, 200 frames:

| Method | X-std (px) | Y-std (px) | Peak-peak (px) |
|---|---|---|---|
| Binary moments (current `detect_dots`) | 0.106 | 0.066 | 0.53 |
| NCC + parabolic on **saturation** | **0.023** | **0.017** | 0.10 |
| NCC + parabolic on grayscale | 0.029 | 0.027 | 0.15 |
| 2D Gaussian on saturation (hybrid default) | 0.086 | 0.101 | 0.55 |
| 2D Gaussian on grayscale | 0.054 | 0.047 | 0.31 |

Per-pixel frame-to-frame noise on H.264 MKV (5 static frames): Gray=0.66, **S=1.82**, V=1.03, H=0.58. Chroma subsampling makes saturation the noisiest channel — yet the current code uses S for Gaussian fitting and the mask threshold lives on S.

Real dot circularity at 21 MPa: 0.56, 0.60, 0.58, 0.68 — three of four sit at or below the 0.6 threshold.

## Root Cause

Dots are soft Gaussian blobs. A hard HSV `inRange` threshold cuts across the intensity gradient; S-channel noise (±1.8 values) flips boundary pixels in/out each frame, the binary mask wiggles, and `cv2.moments` on that mask propagates the wiggle into the centroid. Every downstream strain/Poisson value carries this jitter.

---

## Issues (Prioritized)

### P1 — Mask-noise wins (ship today)

1. **Binary moments is the fallback default** — `video_phase3_4.py:842`. 4–5× noisier than NCC. Either make `--template-track` the default, or replace the `cv2.moments` centroid in `video_utils.py:163-168` with an intensity-weighted centroid on the masked patch (cuts std 0.11 → 0.07 px).

2. **Hybrid tracker's Gaussian fit runs on saturation** — `dot_tracker.py:330`, `:454`, `:491`. On this codec saturation is the worst channel. Switch to grayscale (or pre-blur S with a 3×3 Gaussian). Expected ~2× noise drop.

3. **Circularity threshold 0.6 is too tight** — `video_phase3_4.py:152`, `dot_tracker.py:65`. Real dots sit at 0.56–0.68; one extra perimeter pixel from codec noise kills the detection. Drop to 0.4.

4. **Morph order is OPEN → CLOSE** — `video_utils.py:143-144`. Erodes dot edges first, then reconnects unevenly. Swap to CLOSE → OPEN, or replace with a single 3×3 median filter.

5. **HSV thresholds are fixed constants** — `5,120,120`–`25,255,255`. Make them per-test, sampled during `dot_correct.py` from the user-clicked dot centers.

### P2 — Hybrid-tracker bugs / hygiene

6. **`_annotate_hybrid_frame` and `HYBRID_METHOD_COLORS` defined twice** — `video_phase3_4.py:402,410` and `:469,477`. Second definitions shadow the first. Delete the duplicates.

7. **Unconditional template refresh every 100 frames** — `video_phase3_4.py:828-838`. A refresh during the load ramp at t≈11s bakes motion blur into the template. Gate refresh on NCC ≥ 0.9 AND position std over last N frames < 0.1 px.

8. **Gaussian fit silently falls back to weighted centroid** — `dot_tracker.py:147,158,164,171`. When the log-linear fit fails, noise doubles without warning. Add a counter; if failure rate > ~10%, bail to NCC.

### P3 — Pipeline level

9. **Loading starts at t≈11s, not t=0** (per memory). First ~10s of "Poisson window" data are pre-load jitter. Use `tssp_sync.detect_load_ramp_start` as the Poisson window origin, not video frame 0.

10. **No temporal smoothing on positions.** Creep/Poisson is smooth by physics. A causal 5-frame median on `dv_px`/`dh_px` at plot time halves reported noise with ~0.05 s lag — no reprocessing needed.

11. **Codec is the floor.** Chroma subsampling is why S is so noisy; no post-processing recovers it. If the capture rig allows, record lossless (FFV1 or uncompressed AVI).

---

## Recommended Next Step

Highest ROI: combine **P1.1** (swap default off binary moments) with **P1.3** (loosen circularity) — both fix the path that generated the current `series_all.csv`. A `--compare-methods` debug mode that writes side-by-side dv/dh traces from binary / NCC+parabolic / Gaussian-gray for one test would let the improvement be verified before re-running the full batch.

## Artifacts from This Audit

- `/tmp/roi_21_f1200_4x.png` — 4× upscaled ROI at loading onset (shows soft Gaussian dots)
- `/tmp/mask_21_f1200_4x.png` — corresponding binary mask (shows edge raggedness)
- `/tmp/roi_21_f100.png`, `/tmp/mask_21_f100.png` — pre-load reference crops
