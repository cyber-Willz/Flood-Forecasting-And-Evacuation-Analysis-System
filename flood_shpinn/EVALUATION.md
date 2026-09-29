# v0.3 evaluation (Fort Worth TX DEM, 651 cells, 3000 epochs, seed 1, single run each)

**Bottom line: v0.2 remains the accuracy-safe default. The v0.3 additions are optional and experimental.**
Nothing here measures accuracy against *observed* flooding; no observed extent was reachable (see "Observed data").

| Run | Physics | SHPINN vs own RK4 (NSE / extent acc.) | Evac-window pop. agreement | Pass |
|---|---|---|---|---|
| v0.2 baseline | k_dir=0, D=.35, lam=.06 | 0.997 / 95.1% | 99.96% | yes |
| moderate routing | k_dir=0.5 | 0.815 / 89.6% | 29.3% (16,586 people "unsafe") | no |
| calibrated (grid-edge) | k_dir=1, D=1.6, lam=.01 | 0.007 / 59.4% (autoscaled state) | 20.8% | no |

## Reference mask is not evidence of skill
`real_data/reference.csv` is a HAND (height above nearest drainage) map from the same DEM, not observed flooding.
Against it: ranking cells by low elevation alone (no physics) AUC 0.918; 9x9 local relief 0.933; v0.2 physics 0.811;
calibrated RK4 physics 0.901 (held-out folds 0.881 / 0.925). A 5-epoch network that predicts no flooding scores 0.928.
The evaluator now prints the trivial baseline; a model must beat it to claim skill.

## Negative results
- Rescaling the state (`--depth-scale auto`) did not help the calibrated case (made it worse).
- Calibrated parameters land on the search-grid edge (D max, lam min), so the fit is not identified.
- The current trainer cannot fit strongly directed physics; next: loss weighting toward deep cells, more epochs/capacity.
- Single seed per configuration; the shipped v0.2 results (NSE 0.999) used shim crates, this run the real crates (0.997).

## Observed data
`real_data/observed/2025_Jul_TX_Flood.csv`: 149 USGS high-water marks, July 2025 Guadalupe River flood (Kerr/Kendall Co., TX), from
github.com/dinukem/HighWaterFIM (Apache-2.0). Water-surface elevation (NAVD88, ft) only; `height_above_gnd` is empty.
NOT used for validation: its DEM was unavailable (repo link 404; 3DEP/FEMA/USGS/NOAA/Zenodo blocked in the build sandbox).
To use it: supply a 3DEP DEM for a Kerrville sub-reach of <= ~2,000 cells, compare DEM + predicted peak depth to the marks.
`real_data/prep_observed_fema.py` (FEMA SFHA rasteriser) is likewise untested.

## Tests
`cargo test --release -p flood_shpinn`: 3 pass (AUC, routing-operator mass conservation, k_dir=0 reproduces v0.2, depths >= 0).
