# flood_shpinn v0.4 — river-routing / upstream-inflow component, scored on the same TX windows

**Bottom line.** v0.3 could not represent the July-2025 Guadalupe flood at all (peak depth 0.10 m at v0.2 defaults, ≤2.4 m calibrated, water-surface error identical to assuming zero depth). v0.4 adds a fluvial layer that takes an upstream inflow hydrograph and routes it down the windows. On the same five windows it now places the water surface within 1.2–3.0 m of the marks (pooled 2.2 m over 60 marks), beats the zero-depth baseline in 4 of 5 windows, and reproduces the Hunt→Kerrville wave timing to within about 0.6 h. It does **not** beat a low-elevation ranking on mark rank, its discharge is inconsistent with the one independent discharge estimate I found (1.8× high at Kerrville), and it **cannot** reproduce the July-2026 event with upstream inflow alone. Details and caveats below; read "What this does not show" before relying on any number.

## What was added
`flood_shpinn/src/river.rs` (new, physics-only, no Burn) and `examples/river_eval.rs`:

1. **Thalweg extraction** — least-cost path over the DEM from the lowest boundary cell far from the outlet to the outlet; non-increasing bed profile along it.
2. **HAND against that thalweg** → **synthetic rating curve** (connected inundated volume → area A, hydraulic radius R, Manning Q = A R^(2/3) S^(1/2) / n), invertible to turn discharge into stage.
3. **Muskingum–Cunge routing** of an upstream `Hydrograph` through each window and the gaps between windows, with optional lateral inflow. Default is constant-parameter (exactly volume-conserving); `--mc variable` re-evaluates coefficients each step and is *not* conservative (measured +8 % volume in a unit test) — kept only for sensitivity runs.
4. **Connected inundation** — `WSE_i = bed(nearest thalweg) + stage`, depth = max(WSE − z, 0), kept only where 8-connected to the channel.
5. Optional `--pluvial v02|cal` overlays the unchanged v0.3 ponding model by cell-wise max (the two are decoupled; the linear ponding model has no momentum and cannot carry thousands of m³/s).
6. `Cargo.toml`: the neural surrogate is now behind a default feature `nn`; `--no-default-features` builds the physics stack (catchment, truth, evac, calib, river) in ~80 s with no Burn. Model code for v0.3 is otherwise untouched.

Tests: 5 new (`tests/river.rs`: thalweg/bed monotone, rating monotone + stage round-trip, inundation connected & monotone in stage, routing volume-conserving/attenuating/lagging, lateral volume) + the 3 existing v0.3 tests — 8/8 pass. Rust geometry was cross-checked cell-for-cell against an independent Python prototype (thalweg identical in all 5 windows; bed profile identical in 4, differing by up to 1.2 m in window 4 where equal-cost path ties resolve differently).

## Setup (unchanged inputs, one added window)
Same 4 windows (40×40 cells, 100 m block-mean 3DEP DEM), same 148 USGS HWMs and interpolated-extent reference, same metrics as `score.py`; window 4 (Center Point) from the 2026 addendum has marks but no reference extent. Chain order down the river: w0 → w2 → w3 → w1 (Kerrville) → w4 (Center Point). Upstream inflow at w0 = **8,892 m³/s (314,000 cfs), the USGS indirect measurement at Hunt** (EGUsphere preprint egusphere-2026-1750). Hydrograph *shape* is assumed (Nash-type wave, `t0=3 h, tp=1.5 h, shape=12`); the Hunt gauge failed on the rising limb, so no observed hydrograph exists.

## Results — same windows, same metrics
Reference-extent AUC (0.5 = chance), mark rank (mean percentile of the marked cells; 0.5 = chance), and WSE RMSE at marks (m):

| window | AUC v0.2 / v0.3-cal / **v0.4** / low-elev | mark rank v0.2 / cal / **v0.4** | WSE RMSE zero-depth / **v0.4** / best-uniform-stage floor |
|---|---|---|---|
| w0 | 0.633 / 0.876 / **0.977** / 0.980 | 0.56 / 0.60 / **0.65** | 3.13 / **2.58** / 2.58 |
| w1 Kerrville | 0.658 / 0.872 / **0.964** / 0.955 | 0.48 / 0.64 / **0.69** | 3.88 / **1.34** / 1.20 |
| w2 | 0.692 / 0.920 / **0.996** / 0.975 | 0.34 / 0.66 / **0.76** | 4.65 / **2.23** / 1.28 |
| w3 | 0.703 / 0.936 / **0.994** / 0.976 | 0.35 / 0.65 / **0.76** | 2.56 / **3.04** / 0.85 |
| w4 Center Pt | – | – | 2.87 / **1.27** / 1.27 |

Extent skill at >0.10 m vs reference (w0–w3): POD 0.95–1.00, FAR 0.31–0.57, CSI 0.43–0.67 (over-predicts extent, as expected of a uniform-stage HAND fill).

**How to read this honestly**
- **AUC is circular.** The reference "flooded" mask is the DEM below an interpolated water surface, and v0.4's depth is a HAND fill of the same DEM. AUC 0.96–1.0 mostly says the model is a competent terrain-plus-stage map; it beats the low-elevation baseline by only 0.00–0.02 (and is 0.003 worse in w0). Do not present it as skill against observed flooding.
- **Mark rank is the non-circular check, and v0.4 does not beat low elevation** (0.65–0.76 vs the 0.72–0.76 recorded for elevation alone). It does clearly beat v0.2/v0.3 (0.34–0.66).
- **Water-surface error is the real gain**: v0.3 was no better than assuming zero depth in every window; v0.4 is better in 4 of 5, matching the best-possible uniform stage in w0, w1 (within 0.14 m) and w4. It is *worse* than zero depth in w3 (+2.9 m bias): that reach is the flattest (S = 0.00156) and the routed Q there is too high for its rating.
- The three tuned settings (n = 0.05, hydrograph shape 12, rise 1.5 h) were chosen using these same marks. A split-sample check (fit on w0/2/3, test on w1/4, and the reverse) picks the same setting both ways (held-out RMSE 1.31 and 2.64 m), so it is stable, but this is a fit on 60 marks in one storm, not out-of-sample validation. A different, parameter-free choice (n = 0.06, shape 3, rise 2.5 h) pools to 3.1 m RMSE.

## Analysis against real online events

**July 4, 2025 (calibration event).**
- *Timing (independent of the marks):* reports put the Hunt crest at about 05:10 CDT and the Kerrville crest at 06:45 (Kerrville rose 32 ft in 90 min, peak 34.29 ft). The model gives a Hunt→Kerrville lag of 2.2 h; observed ≈ 1.6 h. **Rejected on this ground:** settings with extra reach storage (`k_scale = 3`) score slightly better on WSE (2.07 vs 2.21 m pooled) but delay the wave ~6 h, which contradicts the record.
- *Discharge (independent of the marks): inconsistent.* Reported peaks are 8,892 m³/s at Hunt and "more than 134,000 cfs" (3,794 m³/s) at Kerrville — attenuation ≈ 0.43. The model attenuates only to 0.77 (6,828 m³/s at Kerrville) because uniform-flow rating storage under-represents floodplain storage. The stage is right while the flow is 1.8× high, i.e. Manning n and the missing attenuation are compensating. Consistently, the rating needs 6,207 m³/s to reach the reported Kerrville crest (1.6× the estimate) yet gives only 237 m³/s at the USGS low-flow pair 9.88 ft ↔ 15,900 cfs (450 m³/s, 0.5×): the sign of the rating error flips with stage, as expected when a 100 m block-mean DEM cannot resolve the real channel. **Treat modelled Q as unreliable; the stage/extent result is the defensible output.**
- Hunt-to-Center Point stage: the reported Center Point crest of 39.61 ft implies 8,599 m³/s on this rating vs 6,476 m³/s routed — the model under-shoots there by ~25 % in discharge (WSE bias only −0.14 m because the block-mean bed differs from the gauge bed).

**July 16, 2026 (out-of-sample event; sourced crests, no invented hydrology).** I inverted the rating instead of driving it with an assumed hyetograph. Reported stages: Center Point 37.94 ft (NOAA via Spectrum News) / 38.73 ft (Kerr County Lead) vs 39.61 ft in 2025; Hunt 19.30 ft (Kerr County Lead) vs 37.52 ft in 2025; Kerrville "nearly 17 ft" (CNN, Texas Tribune). On this rating Center Point needs **7,700–8,150 m³/s (90–95 % of 2025)** while Hunt's stage was about half of 2025's. So the 2026 Center Point flood was **mostly generated downstream of Hunt** (rain centred near Ingram/Kerrville, 18.84 in reported near Ingram) and an upstream-inflow-only model cannot reproduce it; the `--lateral ID:hydro.csv` input exists for this, but I did not run a 2026 scenario because I have no sourced lateral hydrograph or Hunt discharge for that day and would rather not invent them. Rating-inferred Kerrville flow of ~1,170 m³/s (17 ft) vs ~7,700 m³/s at Center Point 2026 says the same thing: more than 6× growth between the two gauges.

**Correction to the earlier addendum.** The July-2026 addendum in `eval_tx_hwm/README.md` used Kerrville crest **30.06 ft**; the sources I found (CNN, Texas Tribune) report roughly 17 ft. On this rating 30.06 ft would need 4,725 m³/s versus 1,172 m³/s for 17 ft, so the earlier "2026/2025 stage ratio 0.83 at Kerrville" is not supported by these sources. The Center Point figures (37.94 / 38.73 ft) are consistent with the earlier addendum, which noted the source disagreement. I did not re-verify gauge datums independently, except that the Center Point datum reproduces the nearest 2025 HWM within 0.1 m (39.61 ft → 478.28 m vs 478.38 m).

## What this does not show
- No observed hydrograph was available for 2025, so hydrograph shape and timing are assumptions; only the Hunt peak (USGS indirect measurement) and crest timings from press reports anchor it.
- One storm, five windows, 60 marks; three parameters tuned on those marks.
- 100 m block-mean DEM: the channel is under-resolved (low-flow rating error 0.5×, flood rating error 1.6×). The 10 m DEM (174 MB, not in the archive) would improve this; it was not available.
- Uniform stage per window (marks show a real WSE slope of about 0.002 along the thalweg, which uniform stage ignores — the "best uniform stage" floor of 0.85–2.6 m is the limit of this design).
- Placeholders unchanged: pop = 10, imperv = 0.20, outlets/shelters from DEM edges. Evacuation-window outputs are not computed by the river layer (columns left empty).
- The neural SHPINN surrogate was **not** retrained or evaluated on the river layer; v0.4's river component is a physics solver.
- Verified: the default-feature (`nn`) build compiles; the v0.3 path (`flood_demo`, window 1) reproduces the archived peak depths exactly (max difference 0.0 m over 1,600 cells); all 8 tests pass with and without `nn`. `flood_demo` prints FAIL for 1-epoch runs (the untrained network misses its RK4 target) exactly as the archived logs do; that check is about the surrogate, not the physics or the river layer.

## Reproduce
```
cargo build --release -p flood_shpinn --no-default-features --example river_eval
cd eval_tx_hwm
../target/release/examples/river_eval --chain windows/chain_2025.csv --out results_v04/final --tag v04 --rain windows/rain.csv --pluvial cal
python3 scripts/score_river.py results_v04/final v04 --json results_v04/final_score.json
python3 scripts/sens_grid.py          # 36-case sensitivity grid (writes results_v04/sens_grid.json)
cargo test --release -p flood_shpinn --no-default-features
```
Outputs written per window are compatible with `scripts/score.py`. Files: `results_v04/final_score.{json,txt}`, `gauge_inversion.json`, `sens_grid.json`, `v04_summary.png`.


