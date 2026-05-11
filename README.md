# AM Creep Analysis Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python framework for analyzing viscoelastic creep behavior in additively manufactured
Tough1500 photopolymer resin from nanoindentation, bulk compression, and tensile testing.
This release archives the analysis and plotting pipeline used to produce the
creep, compliance, and Time-Stress Superposition (TSSP) master-curve figures of
the accompanying thesis chapter.

## Setup

```bash
cd AM_Creep_Analysis
pip install numpy pandas scipy matplotlib lmfit
```

## Quick Usage

```bash
# Plot nanoindentation displacement vs time
python creepycrawlies.py -d nano/103/ -i out/ -x 0 -y 1

# Calculate and plot creep compliance (Berkovich)
python creepycrawlies.py -d nano/103/ -i out/ -x 0 -y compliance --calculate_compliance

# Overlay multiple stress levels
python creepycrawlies.py -d nano/103-17.5 nano/103-20 nano/103-22.5 \
    -i out/ -x 0 -y 1 --overlay

# Bulk 24-hour creep overlay
python src/bulk/plot_bulk_24hrs.py

# Bulk TSSP master curve
python src/tssp_master.py

# Nanoindentation TSSP master curve
python src/nanoindent/nano_master.py
```

## Key Modules

- **`src/models.py`** — 16 viscoelastic creep models (Burgers, generalized Kelvin-Voigt,
  Prony series, Peng stress-lock, Peng + viscoplastic dashpot)
- **`src/utils.py`** — Compliance calculators for flat punch, Berkovich, conical,
  frustum probes
- **`creepycrawlies.py`** — Main CLI for nanoindentation analysis with caching and
  SEM error bars
- **`src/tssp_master.py`** — Bulk TSSP master-curve construction (shift factors +
  WLF-style fit + ASTM 24-hour overlay)
- **`src/nanoindent/nano_master.py`** — Nanoindentation TSSP master-curve construction

## Documentation

- **[SCRIPTS.md](SCRIPTS.md)** — Inventory of every analysis/plotting script in this
  release with the thesis figures it produces and how to run it
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** — Full project structure, modules,
  data conventions, and material constants

## Citation

If you use this software, please cite it via the metadata in
[`CITATION.cff`](CITATION.cff). A versioned DOI is published on
Zenodo for each tagged release.

```
Rocheville, E. J. (2026). AM Creep Analysis: Viscoelastic Creep Analysis Framework
for Additively Manufactured Polymers (Version 1.0.0) [Computer software].
Zenodo. https://doi.org/<DOI>
```

(Replace `<DOI>` with the Zenodo DOI minted at release time.)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file
for details.

## References

- Peng et al. (2015) — Nanoindentation creep of nonlinear viscoelastic polypropylene,
  *Polymer Testing* 43:38–43
- Thapa & Cheng (2024) — Flat punch viscoelastic solution
- Oliver & Pharr (1992) — Hardness and elastic modulus, *J. Mater. Res.* 7(6):1564–1583
- Wang & Fancey (2017) — Application of time–stress superposition to viscoelastic
  behavior of polyamide 6,6 fiber and its "true" elastic modulus,
  *Mechanics of Time-Dependent Materials*
- Jazouli et al. (2005) — Application of time–stress equivalence to nonlinear creep
  of polycarbonate, *Polymer Testing* 24(4):463–467
