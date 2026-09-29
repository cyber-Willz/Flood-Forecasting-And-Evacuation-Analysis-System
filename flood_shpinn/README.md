# flood_shpinn v0.3 (experimental additions on v0.2)
Defaults reproduce v0.2. New optional flags: `--k-dir K` (directed downhill routing), `--d`, `--lam`, `--depth-scale auto|X`,
`--reference ref.csv [--calibrate]` (agreement with a flood-prone mask, plus trivial-baseline AUC).
See EVALUATION.md before using any v0.3 option: it currently does not improve, and in tests degrades, the surrogate.

    cargo run --release -p flood_shpinn --example flood_demo -- --gis real_data_inputs/cells.csv --rain real_data_inputs/rain.csv \
      --cell-m 510.4 --epochs 3000 --seed 1 --out out --json m.json     # v0.2 behaviour

Workspace uses the real shpinn / spectral_hypergraph / krylov_ds / nbsc crates (v0.2's SHIMS are not needed).
Limits from v0.2 still apply (reduced-order lumped model; uncalibrated to observations; proxy population/imperviousness; synthetic storm).


## v0.5 — 1D compound-section router
`river1d.rs` adds a 1D local-inertial river router with per-section HAND geometry, compound channel/floodplain roughness and a slope-following peak
water surface; `river_eval --router d1d` (default) uses it, `--router mc` keeps the v0.4 Muskingum–Cunge path. See `EVALUATION_v05.md`
(pooled water-surface RMSE 2.21 → 1.79 m on the 2025 marks; discharge attenuation still not resolved).
