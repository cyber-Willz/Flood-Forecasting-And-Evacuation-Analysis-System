# TX July-2025 high-water-mark evaluation of flood_shpinn v0.3

**No model code was changed.** This package is the v0.3 crate unmodified plus a real-data evaluation harness and its results.
The only "improved" setting is a physics parameter set (`--k-dir 1 --d 1.6 --lam 0.01`, from the crate's own v0.3 options);
it improves on the shipped v0.2 defaults but still does not beat terrain-only baselines (see below).

## Data
- DEM: `2025_Jul_TX_Flood_DEM.tif` (10 m, EPSG:5070) from the HighWaterFIM Drive folder. Not included (174 MB): place it in `dem/`.
- Marks: `flood_shpinn/real_data/observed/2025_Jul_TX_Flood.csv` (149 USGS HWMs, 148 with elevation).
- Rain: `windows/rain.csv` — hand-built 12 h series (265 mm total) approximating press/NWS reports (6.5 in/3 h at Hunt, 10-12 in overall). Assumption, not a gauge record.
- Placeholders: pop = 10, imperv = 0.20, outlets = 2 lowest edge cells, shelters = 2 highest edge cells.

## Method
4 windows of 40x40 cells (100 m) around the densest mark clusters (`scripts/prep_windows.py`). Reference "flooded" = cells whose
10 m DEM is below an inverse-distance-interpolated water surface within 800 m of a mark (HighWaterFIM-style; interpolated, not observed).
`scripts/score.py <tag>` scores peak depth from `flood_demo` (RK4 physics; network run for 1 epoch only) via AUC and mark rank vs terrain baselines.

## Run
    cargo build --release -p flood_shpinn --example flood_demo     # from repo root (rustc 1.75 ok)
    cd eval_tx_hwm/scripts && python3 hwm_probe.py && python3 prep_windows.py
    sh runv.sh phys 0   # etc.; add --k-dir/--d/--lam for variants; then: python3 score.py phys

## Results (4 windows; AUC 0.5 = chance)
| Ranking | AUC vs interpolated extent | mean rank of mark cells |
|---|---|---|
| low elevation only | 0.955-0.980 | 0.72-0.76 |
| height above nearest low ground (9x9) | 0.907-0.987 | 0.65-0.71 |
| v0.2 defaults | 0.63-0.70 | 0.34-0.56 |
| k_dir 0.5 | 0.75-0.86 | 0.55-0.62 |
| k_dir 1, D 1.6, lam 0.01 | 0.87-0.94 | 0.60-0.66 |

Water-surface error at marks: 2.5-4.7 m for every variant, no better than assuming zero depth. Marks are mostly within ~1 m of the DEM.

## Limits
Rain-driven ponding model only; the Guadalupe flood was upstream-fed river flooding. Elevation baselines share the DEM with the reference
(circular advantage); mark rank is the non-circular check. Neural surrogate not retrained on these windows (sandbox compute), so results are for its RK4 target.
Single storm, 4 windows, 10-16 marks each. The EU CEMS-EFAS data was unreachable from the sandbox.

---
## Addendum: July 16, 2026 event (second real event, same area) — gauge crests
Still **no model code changed**. Run: `sh run26.sh e26phys 1 rain2026.csv` (add `--k-dir 1 --d 1.6 --lam 0.01` for the calibrated variant), windows 1 (Kerrville) and 4 (Center Point, new), then `python3 score_gauge.py`.

Observations (public web sources; provisional): Kerrville USGS 08166200 (30.0533, -99.1634; gauge zero 1601.14 ft NAVD88) crest 30.06 ft;
Center Point USGS 08166250 (29.9878, -99.11; datum 1529.54 ft) crest 37.94 ft (sources give 37.94-38.73). 2025 comparison = nearest HWM to each gauge (61 m and 29 m).
Rain 2026 = 18.84 in (479 mm) reported for one day near Ingram, spread over an ASSUMED 12 h triangular hyetograph (`windows/rain2026.csv`); the timing and the spatial pattern are assumptions.

| Gauge / event | observed river depth above gauge zero | model peak depth v0.2 / calibrated |
|---|---|---|
| Kerrville 2026 | 9.16 m | 0.13 / 0.33 m |
| Kerrville 2025 | 11.05 m | 0.08 / 0.19 m |
| Center Point 2026 | 11.56 m | 0.13 / 0.38 m |
| Center Point 2025 | 12.18 m | 0.08 / 0.23 m |

Model reaches 0.6-3.6% of observed stage. Observed 2026/2025 stage ratio 0.83 (Kerrville) and 0.95 (Center Point); the model predicts 1.6-2.0x (it scales with rain), the wrong direction.
Note: 100 m block-mean ground at Center Point sits ~11.5 m above the gauge datum, so "WSE error vs block-mean DEM" there looks small by coincidence; use the stage comparison.
Sources disagree on some crests (Kerrville 2025: 34.29 vs 37.51 ft; Center Point 2026: 37.94 vs 38.73 ft) — the HWM cross-check supports the higher Kerrville value.


---
## v0.4 update
The river-routing / upstream-inflow component and its scoring on these same windows are documented in `../EVALUATION_v04.md` (code: `flood_shpinn/src/river.rs`, `flood_shpinn/examples/river_eval.rs`, `scripts/score_river.py`). The statements above describe v0.3 and remain accurate for it; note the Kerrville-2026 crest used in the July-2026 addendum (30.06 ft) conflicts with sources reporting about 17 ft — see EVALUATION_v04.md.
