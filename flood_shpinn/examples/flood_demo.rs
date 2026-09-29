//! Flood forecast + evacuation-window demo (v0.2).
//!
//!   cargo run --release -p flood_shpinn --example flood_demo -- \
//!       --gis cells.csv --rain rain.csv --cell-m 510.4 --out out_dir \
//!       [--model v2|legacy] [--epochs N] [--seed S] [--colloc K] [--k-embed K] [--hidden 64,64] \
//!       [--speed-kmh 20] [--json metrics.json]
//!       [--reference reference.csv [--calibrate] [--max-depth-m 2.0]] [--k-dir K] [--d D] [--lam L]
//!
//! v0.3: `--k-dir > 0` turns on directed downhill routing; `--reference` scores the peak-depth map against a
//! known-flood-prone mask (observed data, or the HAND proxy from real_data/make_reference.py); `--calibrate`
//! fits (k_dir, D, lam) to that mask with a 2-fold spatial cross-validation report before training.
//!
//! With no `--gis/--rain` a SYNTHETIC valley + design storm is used (as in v0.1).
//! Ground truth is an independent RK4 solve of the same reduced-order model: results validate the
//! SHPINN *solver*, not the model's fidelity to a real basin.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use flood_shpinn::catchment::Catchment;
use flood_shpinn::evac::{plan, CellPlan, DepthSeries, EvacParams};
use flood_shpinn::floodnet::{train, train_legacy, Model, TrainCfg};
use flood_shpinn::metrics::{depth_skill, evac_skill};
use flood_shpinn::rainfall::Rainfall;
use flood_shpinn::calib::{auc, calibrate, contingency, matched_prevalence_csi, peak_map, subset, RefMask};
use flood_shpinn::truth::{simulate, Params};

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "flood_out".into()));
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let cell_m: f64 = arg(&args, "--cell-m").map(|s| s.parse().unwrap()).unwrap_or(250.0);
    let model = match arg(&args, "--model").as_deref() {
        Some("legacy") => Model::Legacy,
        _ => Model::V2,
    };
    let seed: u64 = arg(&args, "--seed").map(|s| s.parse().unwrap()).unwrap_or(1);
    let speed: f64 = arg(&args, "--speed-kmh").map(|s| s.parse().unwrap()).unwrap_or(20.0);

    let (gis_path, rain_path, synthetic) = match (arg(&args, "--gis"), arg(&args, "--rain")) {
        (Some(g), Some(r)) => (PathBuf::from(g), PathBuf::from(r), false),
        _ => {
            let g = out.join("synthetic_gis.csv");
            let r = out.join("synthetic_rainfall.csv");
            Catchment::synthetic(8, 10).write_csv(&g)?;
            Rainfall::design_storm(130.0, 2.5, 0.5, 5.0, 6.0, 0.25).write_csv(&r)?;
            (g, r, true)
        }
    };
    let mut cat = Catchment::load_csv(&gis_path, cell_m)?;
    cat.cell_m = cell_m;
    let rain = Rainfall::load_csv(&rain_path)?;
    let mut p = Params::default();
    if let Some(v) = arg(&args, "--k-dir") { p.k_dir = v.parse().unwrap(); }
    if let Some(v) = arg(&args, "--d") { p.d = v.parse().unwrap(); }
    if let Some(v) = arg(&args, "--lam") { p.lam = v.parse().unwrap(); }
    let reference = match arg(&args, "--reference") {
        Some(rp) => Some(RefMask::load_csv(&PathBuf::from(rp), &cat)?),
        None => None,
    };
    let t_end = *rain.hours.last().unwrap();

    println!("== inputs ({}) | model = {:?} | seed = {seed} ==", if synthetic { "SYNTHETIC" } else { "user data" }, model);
    println!("grid {}x{} = {} cells, {} outlets, {} shelters, pop {:.0}", cat.nrows, cat.ncols, cat.n(), cat.outlets().len(), cat.shelters().len(), cat.cells.iter().map(|c| c.pop).sum::<f64>());
    println!("storm: peak {:.0} mm/h, total {:.0} mm over {t_end:.1} h", rain.mm_per_h.iter().cloned().fold(0.0, f64::max), rain.total_mm());

    let (hg, lap) = cat.laplacian()?;
    let w = cat.convergence_weights(&lap, 0.6);
    println!("hypergraph: {} vertices, {} hyperedges", hg.num_vertices(), hg.num_hyperedges());
    let ldir = cat.downhill_operator();
    let max_depth: f64 = arg(&args, "--max-depth-m").map(|s| s.parse().unwrap()).unwrap_or(2.0);
    let keep_all: Vec<bool> = cat.cells.iter().map(|c| !c.outlet).collect();
    let p_v02 = Params::default();
    if let (Some(r), true) = (&reference, args.iter().any(|a| a == "--calibrate")) {
        let half = cat.ncols / 2;
        let west: Vec<bool> = cat.cells.iter().map(|c| !c.outlet && c.col < half).collect();
        let east: Vec<bool> = cat.cells.iter().map(|c| !c.outlet && c.col >= half).collect();
        let score = |pp: &Params, keep: &[bool]| { let pk = peak_map(&simulate(&cat, &lap, &ldir, &w, &rain, pp, t_end, 0.25)); let (s, l) = subset(&pk, &r.flood, keep); auc(&s, &l) };
        println!("\n== calibration vs reference mask ({} of {} cells flood-prone) ==", r.flood.iter().filter(|&&x| x).count(), cat.n());
        for (name, train, test) in [("fit west -> test east", &west, &east), ("fit east -> test west", &east, &west)] {
            let f = calibrate(&cat, &lap, &ldir, &w, &rain, &p_v02, t_end, r, train, max_depth);
            println!("  {name}: k_dir={} D={} lam={}  train AUC {:.3}  held-out AUC {:.3}  (v0.2 defaults held-out {:.3})", f.params.k_dir, f.params.d, f.params.lam, f.train_auc, score(&f.params, test), score(&p_v02, test));
        }
        let f = calibrate(&cat, &lap, &ldir, &w, &rain, &p_v02, t_end, r, &keep_all, max_depth);
        println!("  full fit ({} sims): k_dir={} D={} lam={}  AUC {:.3}  (v0.2 defaults {:.3})", f.tried, f.params.k_dir, f.params.d, f.params.lam, f.train_auc, score(&p_v02, &keep_all));
        p = f.params;
    }
    // v0.3: the model is linear, so `depth_scale` only normalises the state (h = s * depth_scale) and leaves the
    // physics unchanged. `auto` picks it from a coarse solve so the network's target peaks at ~2 state units.
    if let Some(v) = arg(&args, "--depth-scale") {
        p.depth_scale = if v == "auto" {
            let mut q = p;
            q.depth_scale = 0.1;
            let pk = peak_map(&flood_shpinn::truth::simulate_sub(&cat, &lap, &ldir, &w, &rain, &q, t_end, 0.25, 2));
            (flood_shpinn::calib::p99_peak(&cat, &pk) / 2.0).max(0.02)
        } else {
            v.parse().unwrap()
        };
    }
    println!("physics params: D={} lam={} k_dir={} depth_scale={:.4}", p.d, p.lam, p.k_dir, p.depth_scale);
    let truth = simulate(&cat, &lap, &ldir, &w, &rain, &p, t_end, 1.0 / 12.0);

    let epochs: usize = arg(&args, "--epochs").map(|s| s.parse().unwrap()).unwrap_or(4000);
    let mut cfg = match model {
        Model::V2 => TrainCfg::v2(t_end, epochs, seed),
        Model::Legacy => TrainCfg { epochs, t_end, seed, ..Default::default() },
    };
    if let Some(k) = arg(&args, "--k-embed") { cfg.k_embed = k.parse().unwrap(); }
    if let Some(k) = arg(&args, "--colloc") { cfg.n_colloc = k.parse().unwrap(); }
    if let Some(h) = arg(&args, "--hidden") { cfg.hidden = h.split(',').map(|x| x.parse().unwrap()).collect(); }

    println!("\n== training SHPINN ({:?}, {epochs} epochs) ==", model);
    let t0 = Instant::now();
    let net = match model {
        Model::V2 => train(&cat, &hg, &ldir, &w, &rain, &p, &cfg, (epochs / 8).max(1))?,
        Model::Legacy => train_legacy(&cat, &hg, &w, &rain, &p, &cfg, (epochs / 8).max(1))?,
    };
    let train_s = t0.elapsed().as_secs_f64();
    println!("training wall time: {train_s:.1} s");

    let times: Vec<f64> = truth.iter().map(|x| x.0).collect();
    let pinn = DepthSeries { times_h: times.clone(), depth_m: times.iter().map(|&t| net.predict(t)).collect() };
    let tru = DepthSeries { times_h: times.clone(), depth_m: truth.iter().map(|x| x.1.clone()).collect() };

    // Held-out: odd multiples of 0.25 h (never on the legacy 0.5 h training grid).
    let held: Vec<usize> = (0..).map(|j| 3 + 6 * j).take_while(|&k| k < times.len()).collect();
    println!("\n== depth vs RK4 ground truth (held-out times, {} of them) ==", held.len());
    for &k in held.iter().filter(|&&k| (k - 3) % 12 == 6) {
        let mse = (0..cat.n()).map(|i| (pinn.depth_m[k][i] - tru.depth_m[k][i]).powi(2)).sum::<f64>() / cat.n() as f64;
        let mx = tru.depth_m[k].iter().cloned().fold(0.0, f64::max);
        println!("  t={:.2} h: RMSE {:.4} m   (max true depth {:.3} m)", times[k], mse.sqrt(), mx);
    }
    let peak = |ds: &DepthSeries, i: usize| ds.depth_m.iter().map(|f| f[i]).fold(0.0, f64::max);
    let peak_t = |ds: &DepthSeries, i: usize| { let mut b = (0.0, 0usize); for (k, f) in ds.depth_m.iter().enumerate() { if f[i] > b.0 { b = (f[i], k); } } ds.times_h[b.1] };
    let pk_p: Vec<f64> = (0..cat.n()).map(|i| peak(&pinn, i)).collect();
    let pk_t: Vec<f64> = (0..cat.n()).map(|i| peak(&tru, i)).collect();
    let ep = EvacParams::for_cell(cell_m, speed, t_end);
    let ds = depth_skill(&held.iter().map(|&k| pinn.depth_m[k].clone()).collect::<Vec<_>>(), &held.iter().map(|&k| tru.depth_m[k].clone()).collect::<Vec<_>>(), &pk_p, &pk_t, ep.thr_m);
    println!("\nheld-out RMSE {:.4} m ({:.1}% of max true depth), NSE {:.3}", ds.rmse_m, 100.0 * ds.rel_rmse, ds.nse);
    println!("peak-depth RMSE {:.4} m, bias {:+.4} m; flood extent (>{:.2} m): truth {} cells, model {} cells, CSI {:.3}, POD {:.3}, FAR {:.3}, accuracy {:.1}%", ds.peak_rmse_m, ds.peak_bias_m, ep.thr_m, ds.wet_truth, ds.wet_pred, ds.csi, ds.pod, ds.far, 100.0 * ds.extent_acc);

    let (mz, mp) = (cat.cells.iter().map(|c| c.elev_m).sum::<f64>() / cat.n() as f64, pk_t.iter().sum::<f64>() / cat.n() as f64);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..cat.n() { let (x, y) = (cat.cells[i].elev_m - mz, pk_t[i] - mp); sxy += x * y; sxx += x * x; syy += y * y; }
    println!("corr(elevation, peak depth) in truth model = {:.2}", sxy / (sxx * syy).sqrt());

    let tp = Instant::now();
    let plan_p = plan(&cat, &pinn, &ep);
    let plan_t = plan(&cat, &tru, &ep);
    let es = evac_skill(&plan_p, &plan_t, 0.25);
    println!("\n== evacuation (depth limit {:.2} m, {:.2} min/cell = {:.0} km/h over {:.0} m cells; routing {:.1}s) ==", ep.thr_m, ep.minutes_per_cell, speed, cat.cell_m, tp.elapsed().as_secs_f64());
    println!("population in scope {:.0}; window closes within horizon for: truth {:.0}, model {:.0}", es.pop_total, es.pop_cut_truth, es.pop_cut_pred);
    println!("window-close agreement within 15 min: {}/{} cells ({:.1}% of population); pop-weighted mean |diff| where both close: {:.1} min", es.cells_agree, es.cells, 100.0 * es.pop_agree_frac, es.mae_min);
    println!("unsafe errors (forecast later/never vs truth closes): {:.0} people; over-cautious errors: {:.0} people", es.pop_false_safe, es.pop_false_alarm);

    let mut worst: Vec<&CellPlan> = plan_p.iter().filter(|c| c.window_closes_h.is_some()).collect();
    worst.sort_by(|a, b| a.window_closes_h.partial_cmp(&b.window_closes_h).unwrap().then(b.pop.partial_cmp(&a.pop).unwrap()));
    println!("\nearliest-closing evacuation windows (model | truth):\n  cell(r,c)   pop   closes_h  truth_h  travel_min@t0");
    for c in worst.iter().take(8) {
        let t = plan_t.iter().find(|x| x.cell == c.cell).unwrap();
        let cl = &cat.cells[c.cell];
        println!("  ({},{})  {:7.0}   {:6.2}   {:>6}   {:>6.1}", cl.row, cl.col, c.pop, c.window_closes_h.unwrap(), t.window_closes_h.map(|x| format!("{x:.2}")).unwrap_or("none".into()), c.travel_min_at_0.unwrap_or(f64::NAN));
    }

    let mut s = String::from("row,col,elev_m,pop,peak_depth_model_m,peak_depth_truth_m,peak_time_model_h,peak_time_truth_h,window_closes_model_h,window_closes_truth_h\n");
    for i in 0..cat.n() {
        let c = &cat.cells[i];
        let f = |o: Option<&CellPlan>| o.and_then(|x| x.window_closes_h).map(|v| format!("{v:.2}")).unwrap_or_default();
        s += &format!("{},{},{:.2},{:.0},{:.3},{:.3},{:.2},{:.2},{},{}\n", c.row, c.col, c.elev_m, c.pop, pk_p[i], pk_t[i], peak_t(&pinn, i), peak_t(&tru, i), f(plan_p.iter().find(|x| x.cell == i)), f(plan_t.iter().find(|x| x.cell == i)));
    }
    let tag = format!("{}_seed{seed}", if model == Model::V2 { "v2" } else { "legacy" });
    fs::write(out.join(format!("flood_evac_results_{tag}.csv")), s).map_err(|e| e.to_string())?;

    let max_true = tru.depth_m.iter().flatten().cloned().fold(0.0, f64::max);
    let nonvacuous = ds.wet_truth > 0 && plan_t.iter().any(|c| c.window_closes_h.is_some());
    let ok = nonvacuous && ds.rmse_m < 0.05 * max_true && ds.extent_acc >= 0.9;
    println!("\n{}: held-out RMSE {:.4} m (tol 5% of max true depth {:.3} m = {:.4}), extent accuracy {:.1}% (tol 90%), scenario floods & cuts routes: {nonvacuous}", if ok { "PASS" } else { "FAIL" }, ds.rmse_m, max_true, 0.05 * max_true, 100.0 * ds.extent_acc);

    let mut extra = format!("\"k_dir\":{},\"d\":{},\"lam\":{}", p.k_dir, p.d, p.lam);
    if let Some(r) = &reference {
        let (st, l) = subset(&pk_t, &r.flood, &keep_all);
        let (sp, _) = subset(&pk_p, &r.flood, &keep_all);
        let pk_v02 = peak_map(&simulate(&cat, &lap, &ldir, &w, &rain, &p_v02, t_end, 0.25));
        let (s0, _) = subset(&pk_v02, &r.flood, &keep_all);
        let (a_t, a_p, a_0) = (auc(&st, &l), auc(&sp, &l), auc(&s0, &l));
        let (ct, cp, c0) = (contingency(&st, &l, ep.thr_m), contingency(&sp, &l, ep.thr_m), contingency(&s0, &l, ep.thr_m));
        let (mt, mp, m0) = (matched_prevalence_csi(&st, &l), matched_prevalence_csi(&sp, &l), matched_prevalence_csi(&s0, &l));
        let zneg: Vec<f64> = cat.cells.iter().map(|c| -c.elev_m).collect();
        let (sz, _) = subset(&zneg, &r.flood, &keep_all);
        let a_z = auc(&sz, &l);
        println!("\n== agreement with reference flood-prone mask ({} cells) ==", l.iter().filter(|&&x| x).count());
        println!("  v0.2 default physics (RK4):  AUC {a_0:.3}  matched-prev CSI {m0:.3}  CSI@{:.2}m {:.3} POD {:.3} FAR {:.3}", ep.thr_m, c0.csi, c0.pod, c0.far);
        println!("  this run's physics   (RK4):  AUC {a_t:.3}  matched-prev CSI {mt:.3}  CSI@{:.2}m {:.3} POD {:.3} FAR {:.3}", ep.thr_m, ct.csi, ct.pod, ct.far);
        println!("  this run's SHPINN surrogate: AUC {a_p:.3}  matched-prev CSI {mp:.3}  CSI@{:.2}m {:.3} POD {:.3} FAR {:.3}", ep.thr_m, cp.csi, cp.pod, cp.far);
        println!("  trivial baseline (rank by low elevation, no physics): AUC {a_z:.3}  <- a model must beat this to claim skill");
        extra += &format!(",\"ref_auc_v02_default\":{a_0:.4},\"ref_auc_truth\":{a_t:.4},\"ref_auc_pinn\":{a_p:.4},\"ref_mpcsi_v02_default\":{m0:.4},\"ref_mpcsi_truth\":{mt:.4},\"ref_mpcsi_pinn\":{mp:.4},\"ref_csi_pinn\":{:.4},\"ref_pod_pinn\":{:.4},\"ref_far_pinn\":{:.4}", cp.csi, cp.pod, cp.far);
    }
    if let Some(jp) = arg(&args, "--json") {
        let j = format!("{{\"model\":\"{:?}\",\"seed\":{seed},\"epochs\":{epochs},\"train_s\":{train_s:.1},\"cells\":{},\"rmse_m\":{:.5},\"rel_rmse\":{:.5},\"nse\":{:.5},\"peak_rmse_m\":{:.5},\"peak_bias_m\":{:.5},\"csi\":{:.4},\"pod\":{:.4},\"far\":{:.4},\"extent_acc\":{:.4},\"wet_truth\":{},\"wet_pred\":{},\"pop_total\":{:.0},\"pop_cut_truth\":{:.0},\"pop_cut_pred\":{:.0},\"pop_agree\":{:.4},\"mae_min\":{:.2},\"pop_false_safe\":{:.0},\"pop_false_alarm\":{:.0},\"pass\":{ok}}}\n",
            model, cat.n(), ds.rmse_m, ds.rel_rmse, ds.nse, ds.peak_rmse_m, ds.peak_bias_m, ds.csi, ds.pod, ds.far, ds.extent_acc, ds.wet_truth, ds.wet_pred, es.pop_total, es.pop_cut_truth, es.pop_cut_pred, es.pop_agree_frac, es.mae_min, es.pop_false_safe, es.pop_false_alarm);
        let j = format!("{},{}}}\n", j.trim_end().trim_end_matches('}'), extra);
        fs::write(jp, j).map_err(|e| e.to_string())?;
    }
    if !ok { std::process::exit(1); }
    Ok(())
}
