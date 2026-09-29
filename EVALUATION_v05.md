# flood_shpinn v0.5 — 1D compound-section router, same TX windows, same metrics

**Bottom line.** v0.5 replaces v0.4's Muskingum–Cunge + one-stage-per-window design with a 1D local-inertial router along the thalweg
chain (`src/river1d.rs`). On the same five windows / 60 marks it lowers the pooled water-surface RMSE from **2.21 m to 1.79 m** (zero-depth
baseline 3.54 m), beats the zero-depth baseline in **5 of 5** windows (v0.4: 4 of 5; the w3 failure is fixed: 3.04 → 1.24 m), cuts the
Hunt→Kerrville lag error from +0.6 h to +0.25 h, and reduces the Kerrville discharge over-estimate from 1.8× to 1.5×. It does **not** fix
the discharge inconsistency, does **not** improve mark rank over low elevation, and is worse than v0.4 in w0 (3.04 vs 2.58 m). The
parameters used are a-priori defaults, not fitted to the marks (see "Tuning" — the score is nearly flat in them).

## What changed (from the literature review, implemented and tested)
1. **1D local-inertial solver** (`river1d::simulate`, Bates et al. 2010 form): momentum with semi-implicit compound-conveyance friction
   `S_f = Q|Q|/K²`, node continuity by inverting each section's own area table (`A_new = A + dt·net/dx`) so volume closes exactly even where
   flooded width jumps with stage, and a positivity limiter so a node cannot release more than it holds. Upstream boundary = inflow
   hydrograph, downstream = normal depth. Replaces reach-averaged uniform-flow Muskingum–Cunge, which is outside its validity range at ~50 %
   attenuation. The v0.4 router remains as `--router mc` and reproduces the archived v0.4 numbers exactly.
2. **Per-section compound geometry** (`window_geometry`): every ~400 m of thalweg gets its own A(h), B(h), K(h) from the HAND field.
   Conveyance is split into channel cells (`HAND ≤ h_bf`, `n_ch`) and floodplain cells (`n_fp`); storage uses the full connected width.
3. **Slope-following water surface** (`window_wse`): peak water-surface elevation per node, interpolated along the channel, then flooded and
   connectivity-filtered per cell. v0.4 applied one stage to a whole 4–5 km window.
4. **Optional bathymetry carve** (`--carve`): lowers thalweg cells in the geometry only (B-HAND-style). Tested; it did not help here (see grid).
5. `river_eval --router d1d|mc`, new flags `--n-ch --n-fp --hbf --carve --seg --alpha`; `--lateral ID:hydro.csv` is wired into the 1D router
   (tested for volume closure; not run on real data). New outputs: `<tag>_w<id>_wse.csv`, `<tag>_profile.csv`.
   `scripts/score_river.py` uses the model's own surface when a `_wse.csv` exists; `scripts/tune_v05.py` is the parameter study.

## Results (defaults: n_ch 0.05, n_fp 0.10, h_bf 3 m, section 400 m, no carve; v0.4 hydrograph unchanged)
| window | WSE RMSE zero-depth | v0.4 | **v0.5** | best-uniform-stage floor | WSE bias v0.4 → v0.5 |
|---|---|---|---|---|---|
| w0 | 3.13 | **2.58** | 3.04 | 2.58 | +0.11 → −0.49 |
| w1 Kerrville | 3.88 | 1.34 | **0.69** | 1.20 | +0.60 → −0.63 |
| w2 | 4.65 | 2.23 | **1.26** | 1.28 | +1.82 → +0.83 |
| w3 | 2.56 | 3.04 | **1.24** | 0.85 | +2.92 → +1.17 |
| w4 Center Pt | 2.87 | 1.27 | **0.82** | 1.27 | −0.14 → −0.37 |
| pooled (60 marks) | 3.54 | 2.21 | **1.79** | 1.66 | |

v0.5 sits below the uniform-stage "oracle" in w1, w2, w4 because the surface now slopes along the channel, which that floor ignores.

Independent gauge crests (datums from the earlier addendum, not used in any tuning): Kerrville observed 498.45 m → v0.4 +0.65 m, **v0.5 +0.01 m**;
Center Point observed 478.28 m → v0.4 −1.19 m, **v0.5 −0.55 m**.

Wave: Hunt→Kerrville peak lag 1.85 h (reported ≈1.6 h; v0.4 2.2 h). Peak Q at Kerrville 5,686 m³/s (attenuation 0.64 from 8,892; v0.4 6,828 / 0.77);
reported "more than 134,000 cfs" = 3,794 m³/s, which is a lower bound, so 0.64 is not proven wrong, but 1.5× the only number available is
still high. Mass balance closes to 0.0000 % of inflow volume (in 3.59e7 m³, out 3.66e7, storage 1.06e6 → 0.42e6).

Extent (with the `--pluvial cal` overlay as before): CSI 0.67/0.53/0.57/0.51 (v0.4 0.67/0.51/0.54/0.43), FAR lower in w2–w3. AUC 0.951–0.998 and mark
rank 0.65/0.67/0.75/0.74 (v0.4 0.65/0.69/0.76/0.76) — **AUC is still circular and mark rank is still no better than low elevation (0.72–0.76)**.

## Tuning and robustness
- 144-case grid (`n_ch`×`n_fp`×`h_bf`×`carve`, `results_v05/tune_grid.json`): pooled WSE RMSE spans only **1.77–1.80 m** for the best 8 and the best-J
  case (n_fp 0.07, h_bf 6) is 1.79 m — the same as the a-priori defaults — so the improvement comes from the model structure, not from fitting. Peak-Q attenuation
  does vary (0.45–0.90 across the grid), i.e. discharge remains weakly constrained by anything except the one Kerrville figure. `carve` never helped.
- Split-sample on marks (fit on w0/2/3 → test w1/4, and reverse): held-out RMSE **0.76 m and 2.31 m** (v0.4: 1.31 and 2.64 m).
- Numerics: dt and CFL factor are converged (Q at Kerrville within 0.1 %). Section length is not fully converged: 200/400/800 m gives Kerrville Q 5,510/5,686/5,748 m³/s and lag
  2.10/1.85/1.70 h — treat lag as ±0.2 h.

## What this does not show
- Same single storm, same 60 marks, same hydrograph shape (t0 3 h, tp 1.5 h, shape 12) that v0.4 tuned on those marks; the shape was **not** re-tuned for the new router, which may cost or flatter it.
- The 100 m block-mean DEM still hides the channel. Discharge remains uncertain; the 0.43 vs 0.64 attenuation gap is reduced, not closed. Floodplain storage
  counts all connected wet cells as storage; ineffective-flow (dead-water) zones are not separated.
- w0 is worse than v0.4 (3.04 vs 2.58 m); its marks scatter widely and the router's first sections are influenced by the imposed inflow boundary.
- Gap reaches between windows reuse the adjacent window's section tables (no DEM there).
- No 2026 scenario was run (still no sourced lateral hydrograph); the neural SHPINN surrogate was not retrained; the default-feature (`nn`, Burn) build was **not** re-compiled in this session —
  only `--no-default-features`, where all 14 tests pass (5 v0.4 river, 6 new river1d, 3 v0.3).

## Reproduce
```
cargo build --release -p flood_shpinn --no-default-features --example river_eval
cd eval_tx_hwm
../target/release/examples/river_eval --chain windows/chain_2025.csv --out results_v05/final --tag v05 --rain windows/rain.csv --pluvial cal
python3 scripts/score_river.py results_v05/final v05 --json results_v05/final_score.json
python3 scripts/tune_v05.py              # 144-case study + split-sample
../target/release/examples/river_eval ... --router mc    # v0.4 behaviour
cargo test --release -p flood_shpinn --no-default-features
```

## Next steps
1. Re-tune/replace the assumed hydrograph using the 1D router and add lateral inflow (needed for 2026); 2. dead-water/ineffective-flow zones and a gauge-constrained
Q term in the loss; 3. 1–3 m lidar DEM; 4. only then retrain the SHPINN, as a residual on this solver.
