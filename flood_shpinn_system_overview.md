# flood_shpinn — what it is, what it is for, and how it is built

*Describes version 0.5 (workspace `flood_shpinn_v0.3_tx_hwm_eval`). Written from the source code and the v0.4/v0.5 evaluation reports; every performance number quoted here comes from those reports.*

---

## 1. Purpose

`flood_shpinn` is an experimental **flood-forecasting and evacuation-window analysis** system written in Rust. Given a gridded terrain model (a DEM turned into a table of cells), a rainfall series, and optionally an upstream river discharge, it aims to answer three questions for a study area:

1. **Where will water go?** Peak flood depth for every cell.
2. **When?** Arrival and peak times of the flood at each cell, and how a flood wave moves down a river.
3. **What does that mean for people?** For each populated cell: *how long do residents have to reach a shelter before the route closes?* (the "evacuation window").

It is also a **research vehicle**. The project's central experiment is whether a *physics-informed neural network on a spectral hypergraph* (SHPINN, built from the author's `shpinn` and `spectral_hypergraph` crates) can act as a fast surrogate for flood physics on a GIS-derived graph. A large part of the repository is therefore honest measurement of where that idea works, where it does not, and what physics is missing.

### What it is intended to be used for
- Rapid, low-cost **scenario exploration** on small study areas (tens to a few thousand cells): "what if this storm, or this upstream flow, hits this reach?"
- **Evacuation-timing studies**: comparing shelter reachability under different depth forecasts.
- **Benchmarking** surrogate models and simple physics against observed data (USGS high-water marks, gauge crests, flood-extent references) on real events. The evaluation harness is built to expose weak results rather than hide them.
- A base for **research on graph/hypergraph PINNs** applied to environmental hazards.

### What it is not
It is not an operational forecasting or warning system, and its outputs should not be used for life-safety decisions. Its own reports state the limits plainly: it was checked on one storm (the July 2025 Guadalupe River flood, Texas), five 4 km windows and 60 high-water marks; the DEM is a coarse 100 m block mean; population and imperviousness are placeholders; and river discharge is still not reliably reproduced.

---

## 2. Design at a glance

The system has **two independent flood mechanisms** that are combined at the end, plus an evacuation analysis layer and an evaluation harness.

```
            GIS cell table (cells.csv)                    rainfall (rain.csv)
      row,col,elev_m,pop,imperv,role                       hour,mm_per_h
                    │                                            │
                    ▼                                            │
          ┌──────────────────┐                                   │
          │  catchment.rs    │  grid → spectral hypergraph       │
          │  (Laplacian,     │  Laplacian, downhill operator,    │
          │   flow weights)  │  convergence weights              │
          └────────┬─────────┘                                   │
                   │                                             │
      ┌────────────┴─────────────┐                               │
      ▼                          ▼                               ▼
 PLUVIAL (rain on the cells)                         FLUVIAL (river inflow)
 truth.rs  reduced-order runoff PDE (RK4)            river.rs / river1d.rs
 floodnet.rs  SHPINN surrogate of that PDE           thalweg + HAND + router
      │                                                          │
      └────────────── cell-wise max of depths ───────────────────┘
                                   │
                                   ▼
                     evac.rs  time-dependent evacuation windows
                                   │
                                   ▼
             calib.rs / metrics.rs / eval_tx_hwm/scripts  →  skill scores
```

### 2.1 Input data model (`catchment.rs`, `rainfall.rs`)
- **`cells.csv`**: one row per raster cell — `row, col, elev_m, pop, imperv, role`, where `role` marks `outlet` (water leaves) or `shelter` cells.
- **`rain.csv`**: `hour, mm_per_h`, interpolated piecewise-linearly.
- From the grid the code builds a **spectral hypergraph** (via the `spectral_hypergraph` crate) and its Laplacian; a **directed downhill operator** that moves each cell's water to lower neighbours in proportion to elevation drop; and **convergence weights** that emphasise valley cells.

### 2.2 Pluvial layer — rain falling on the study area (`truth.rs`, `floodnet.rs`)
- **Physics (`truth.rs`)**: a reduced-order runoff model, solved with RK4:
  `ds/dt = w·q(t) − λ·s − D·Δs − k_dir·L·s`
  where `q(t)` is runoff generation from rainfall and imperviousness, `λ` is infiltration/drain loss, `D·Δ` is lateral spreading along the hypergraph, and `k_dir·L` is directed downhill routing. The system is constructed so that depths stay non-negative without clamping. This RK4 solution is the **ground truth** for the surrogate.
- **Surrogate (`floodnet.rs`, feature `nn`, uses Burn)**: a network trained to satisfy that PDE's residual. The v0.2 trainer uses hard initial/boundary constraints, spectral-eigenmap plus physical features (elevation, imperviousness, weight, time), random collocation times, and a semi-analytic time derivative.
- **Limitation by construction**: this is a *linear ponding* model with no momentum. It cannot carry river-scale discharge, which is why the fluvial layer exists.

### 2.3 Fluvial layer — water arriving from upstream (`river.rs`, `river1d.rs`)
The rain-only model cannot represent a flood generated upstream and entering the study area through its edge (as in July 2025). The fluvial layer adds that.

Common steps for each 40×40-cell window:
1. **Thalweg extraction** — least-cost path over the DEM from the lowest boundary cell far from the outlet to the outlet, with a non-increasing bed profile.
2. **HAND** (height above nearest drainage) measured against that thalweg. HAND is the basis of the geometry used to turn flow into stage.

Two routers are available (`river_eval --router`):

| | `mc` (v0.4) | `d1d` (v0.5, default) |
|---|---|---|
| Method | Muskingum–Cunge, constant parameters | 1D local-inertial (momentum + continuity) |
| Geometry | one reach-averaged rating curve per window | ~400 m sections, each with its own area, width and conveyance tables |
| Roughness | one Manning n | separate channel and floodplain n (split at bankfull height) |
| Storage | implied by the rating | explicit, from the flooded width at each stage |
| Water surface | one uniform stage per window | peak surface per node, interpolated along the channel |
| Conservation | exact for constant parameters | exact by construction (area-table inversion + positivity limiter) |

Windows are chained down the river (Hunt → … → Kerrville → Center Point); gaps between windows reuse the adjacent window's section tables. The upstream boundary is a discharge hydrograph, the downstream boundary is normal depth, and optional lateral hydrographs can be added per window. The peak surface is then flooded onto the DEM and kept only where hydraulically connected to the channel.

The two layers are **decoupled**: the final depth is the cell-wise maximum of the fluvial depth and the (optional) pluvial ponding depth.

### 2.4 Evacuation analysis (`evac.rs`)
Roads are the 4-connected cell adjacency. A cell can be entered at time `t` only if its depth is below a threshold; travel time per cell grows with depth. For each populated cell the code scans departure times and reports when the window to reach a shelter first closes. *(The river layer does not currently compute these columns; they are left empty in its output.)*

### 2.5 Calibration and skill (`calib.rs`, `metrics.rs`)
- Compares the peak-depth map to a reference "flood-prone" mask using ROC-AUC, CSI, POD and FAR, and can fit `(k_dir, D, λ)` with a two-fold spatial cross-validation report.
- The reports deliberately add baselines (low-elevation ranking, zero depth, a best-possible uniform stage) so that impressive-looking scores can be checked against trivial alternatives.

### 2.6 Evaluation harness (`eval_tx_hwm/`)
- **Data**: five windows on the Guadalupe River, 148 USGS high-water marks, interpolated-extent reference, gauge crests and reported crest times.
- **Scripts**: `score_river.py` (water-surface error at marks, AUC, mark rank, extent skill, gauge crests), `tune_v05.py` (144-case parameter study plus split-sample check), `sens_grid.py`, and window/data preparation scripts.
- **Drivers**: `examples/river_eval.rs` (physics-only, no Burn) and `examples/flood_demo.rs` (neural surrogate).

---

## 3. Code layout

```
Cargo.toml                    workspace: krylov_ds, nbsc, spectral_hypergraph, shpinn, flood_shpinn
flood_shpinn/
  src/catchment.rs            GIS → grid → hypergraph, downhill operator
  src/rainfall.rs             rainfall series
  src/truth.rs                reduced-order runoff PDE + RK4 ground truth
  src/floodnet.rs             SHPINN surrogate            [feature `nn`, needs Burn]
  src/river.rs                thalweg, HAND, rating curve, Muskingum–Cunge (v0.4)
  src/river1d.rs              1D compound-section router (v0.5)
  src/evac.rs                 evacuation windows
  src/calib.rs, metrics.rs    reference-mask skill and calibration
  examples/flood_demo.rs      surrogate + evacuation demo
  examples/river_eval.rs      fluvial evaluation driver
  tests/                      v03.rs, river.rs, river1d.rs
eval_tx_hwm/                  windows, scripts, results (v0.2 … v0.5)
EVALUATION_v04.md, EVALUATION_v05.md
```
The neural network sits behind the default feature `nn`; building with `--no-default-features` gives the physics-only stack, which compiles in about a minute and needs no Burn.

## 4. Typical workflows

```bash
# Fluvial evaluation on the Texas windows (physics only)
cargo build --release -p flood_shpinn --no-default-features --example river_eval
cd eval_tx_hwm
../target/release/examples/river_eval --chain windows/chain_2025.csv \
    --out results_v05/final --tag v05 --rain windows/rain.csv --pluvial cal
python3 scripts/score_river.py results_v05/final v05

# Rain-driven surrogate + evacuation windows on any cell table
cargo run --release -p flood_shpinn --example flood_demo -- \
    --gis cells.csv --rain rain.csv --cell-m 510.4 --epochs 3000 --out out
```
`river_eval` key options: `--router d1d|mc`, `--q-peak/--inflow` (upstream hydrograph), `--n-ch --n-fp --hbf --carve --seg`, `--lateral ID:file.csv`, `--pluvial none|v02|cal`.

## 5. Where it stands (summary of the reports)

- **v0.3** (rain-only ponding) could not represent the July 2025 river flood; its water-surface error was no better than assuming zero depth.
- **v0.4** added the fluvial layer and reached about 2.2 m pooled water-surface error on 60 marks.
- **v0.5** reduced that to about 1.8 m and beat zero depth in all five windows, but the simulated discharge at Kerrville is still about 1.5× the one reported value, mark rank does not beat simple low-elevation ranking, and the AUC score is partly circular because the reference is built from the same DEM.

## 6. Known limitations

- One storm, one river, 60 marks; hydrograph shape is assumed (no observed hydrograph exists for 2025) and was tuned on the same marks.
- 100 m block-mean DEM hides the true channel; a 1–3 m lidar DEM is the most valuable upgrade.
- Population and imperviousness are placeholders; outlets and shelters come from DEM edges.
- The fluvial and pluvial layers are not coupled.
- The neural SHPINN has not been retrained or evaluated on the river layer; earlier reports found the v0.3 options did not improve the surrogate.
- Evacuation outputs are not produced by the river layer.

## 7. Roadmap (from the evaluation reports)

1. Lateral (tributary / rain-on-channel) inflow — required to reproduce the July 2026 event.
2. Ineffective-flow (dead-water) storage and a gauge-constrained discharge term in the calibration loss.
3. Higher-resolution DEM and hydro-conditioning.
4. Retrain the SHPINN as a residual correction on top of the physics solver, rather than as its replacement.
