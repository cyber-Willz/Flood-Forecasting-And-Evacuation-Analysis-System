//! v0.5 fluvial evaluation driver (physics only; needs no neural network / Burn).
//!
//!   --router d1d (default)  1D compound-section local-inertial router (river1d.rs): dynamic floodplain storage,
//!                           slope-following water surface. Extra flags: --n-ch (default = --n) --n-fp 0.10 --hbf 3.0
//!                           --carve 0 --seg 400 --alpha 0.7
//!   --router mc             the v0.4 Muskingum-Cunge + uniform-stage path, unchanged (for comparison)
//!
//!   river_eval --chain chain.csv --out out_dir --tag v04 \
//!       [--q-peak 8892 --q-base 20 --t0 3 --tp 1.5 --shape 12 | --inflow hydro.csv] \
//!       [--n 0.05] [--mc const|variable] [--dt 0.05] [--tend 14] [--sinuosity auto|1.3] [--k-scale 1] \
//!       [--pluvial none|v02|cal --rain rain.csv] [--lateral ID:hydro.csv ...]
//!
//! `chain.csv` (header required): `id,cells_csv,x0,y1` - windows listed upstream -> downstream; `x0,y1` are the
//! window's top-left corner in a metric CRS (used only to measure the river gap between consecutive windows).
//! The first window receives the inflow hydrograph; every other window receives it routed (Muskingum-Cunge) through
//! half of the upstream window, the gap, and half of its own reach. Per window it writes
//! `<tag>_w<id>/flood_evac_results_v2_seed1.csv` (same columns as `flood_demo`, so eval_tx_hwm/scripts/score.py
//! works unchanged), `<tag>_w<id>_hydro.csv`, `reach_w<id>.csv`, `rating_w<id>.csv`, and `<tag>_summary.json`.

use std::fs;
use std::path::{Path, PathBuf};

use flood_shpinn::catchment::Catchment;
use flood_shpinn::rainfall::Rainfall;
use flood_shpinn::river1d::{connected_depth, series_peak, simulate, window_geometry, window_wse, Chain, ChannelParams, Lateral, Solver1D};
use flood_shpinn::river::{route_mc, write_reach_csv, Hydrograph, Reach};
use flood_shpinn::truth::Params;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}
fn argf(args: &[String], key: &str, d: f64) -> f64 {
    arg(args, key).map(|s| s.parse().unwrap()).unwrap_or(d)
}

struct Win {
    id: usize,
    cat: Catchment,
    reach: Reach,
    x0: f64,
    y1: f64,
}

fn xy(w: &Win, i: usize) -> (f64, f64) {
    let c = &w.cat.cells[i];
    (w.x0 + (c.col as f64 + 0.5) * w.cat.cell_m, w.y1 - (c.row as f64 + 0.5) * w.cat.cell_m)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let chain_path = PathBuf::from(arg(&args, "--chain").ok_or("--chain chain.csv required")?);
    let out = PathBuf::from(arg(&args, "--out").unwrap_or_else(|| "river_out".into()));
    let tag = arg(&args, "--tag").unwrap_or_else(|| "v04".into());
    let n_man = argf(&args, "--n", 0.05);
    let dt = argf(&args, "--dt", 0.05);
    let t_end = argf(&args, "--tend", 14.0);
    let k_scale = argf(&args, "--k-scale", 1.0);
    let variable = arg(&args, "--mc").as_deref() == Some("variable");
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;

    // ---- load windows
    let base = chain_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let text = fs::read_to_string(&chain_path).map_err(|e| e.to_string())?;
    let mut wins: Vec<Win> = Vec::new();
    for (k, line) in text.lines().enumerate().filter(|(k, l)| *k > 0 && !l.trim().is_empty() && !l.starts_with('#')) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if f.len() < 4 {
            return Err(format!("chain line {}: need id,cells_csv,x0,y1", k + 1));
        }
        let cat = Catchment::load_csv(&base.join(f[1]), 100.0)?;
        let reach = Reach::extract(&cat, 0.625)?;
        wins.push(Win { id: f[0].parse().unwrap(), cat, reach, x0: f[2].parse().unwrap(), y1: f[3].parse().unwrap() });
    }
    if wins.is_empty() {
        return Err("empty chain".into());
    }
    let sinu = match arg(&args, "--sinuosity").as_deref() {
        None | Some("auto") => wins.iter().map(|w| w.reach.sinuosity()).sum::<f64>() / wins.len() as f64,
        Some(v) => v.parse().unwrap(),
    };
    let ratings: Vec<_> = wins.iter().map(|w| w.reach.rating(n_man, 30.0, 0.1)).collect();

    // ---- upstream hydrograph at the first window
    let hunt = match arg(&args, "--inflow") {
        Some(p) => Hydrograph::load_csv(&PathBuf::from(p))?,
        None => Hydrograph::nash(argf(&args, "--q-base", 20.0), argf(&args, "--q-peak", 8892.0), argf(&args, "--t0", 3.0), argf(&args, "--tp", 1.5), argf(&args, "--shape", 12.0), t_end, dt),
    };
    let mut laterals: Vec<(usize, Hydrograph)> = Vec::new();
    let mut ai = 0;
    while ai < args.len() {
        if args[ai] == "--lateral" {
            let spec = &args[ai + 1];
            let (id, p) = spec.split_once(':').ok_or("--lateral ID:file.csv")?;
            laterals.push((id.parse().unwrap(), Hydrograph::load_csv(&PathBuf::from(p))?));
        }
        ai += 1;
    }

    // ---- optional pluvial layer (v0.3 ponding model), combined by max()
    let pluvial = arg(&args, "--pluvial").unwrap_or_else(|| "none".into());
    let rain = arg(&args, "--rain").map(|p| Rainfall::load_csv(&PathBuf::from(p))).transpose()?;

    let router = arg(&args, "--router").unwrap_or_else(|| "d1d".into());
    let sinu_gap = |a: &Win, b: &Win| {
        let (ea, ib) = (xy(a, a.reach.outlet()), xy(b, b.reach.inlet()));
        ((ea.0 - ib.0).powi(2) + (ea.1 - ib.1).powi(2)).sqrt() * sinu
    };
    let mut gaps: Vec<f64> = vec![0.0];
    for j in 1..wins.len() {
        gaps.push(sinu_gap(&wins[j - 1], &wins[j]));
    }

    // per-window products, filled by whichever router runs
    let mut q_at: Vec<Hydrograph> = Vec::new();
    let mut stage_t: Vec<Vec<f64>> = Vec::new(); // stage (m above the window's reference bed) at mid-window
    let mut wse_cells: Vec<Vec<f64>> = Vec::new();
    let mut depth_cells: Vec<Vec<f64>> = Vec::new();
    let mut rating_out: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::new();
    let mut extra = String::new();

    if router == "mc" {
        let mut qs: Vec<Hydrograph> = vec![hunt.clone()];
        for j in 1..wins.len() {
            let (a, b) = (&wins[j - 1], &wins[j]);
            let s_gap = 0.5 * (a.reach.slope + b.reach.slope);
            let lat = laterals.iter().find(|(id, _)| *id == b.id).map(|x| &x.1);
            let mut h = route_mc(&qs[j - 1], None, &ratings[j - 1], 0.5 * a.reach.length_m, a.reach.slope, dt, t_end, k_scale, variable);
            h = route_mc(&h, lat, &ratings[j], gaps[j].max(1.0), s_gap, dt, t_end, k_scale, variable);
            h = route_mc(&h, None, &ratings[j], 0.5 * b.reach.length_m, b.reach.slope, dt, t_end, k_scale, variable);
            qs.push(h);
        }
        for (j, w) in wins.iter().enumerate() {
            let rt = &ratings[j];
            let (qp, _) = qs[j].peak();
            let hp = rt.stage(qp);
            let (d, wse) = w.reach.inundation(hp);
            stage_t.push(qs[j].q.iter().map(|&q| rt.stage(q)).collect());
            wse_cells.push(wse);
            depth_cells.push(d);
            rating_out.push((rt.h.clone(), rt.a.clone(), rt.b.clone(), rt.q.clone()));
        }
        q_at = qs;
    } else {
        let cp = ChannelParams {
            n_ch: argf(&args, "--n-ch", n_man),
            n_fp: argf(&args, "--n-fp", 0.10),
            h_bf: argf(&args, "--hbf", 3.0),
            carve: argf(&args, "--carve", 0.0),
            seg_len_m: argf(&args, "--seg", 400.0),
            ..ChannelParams::default()
        };
        let sol = Solver1D { alpha: argf(&args, "--alpha", 0.7), ..Solver1D::default() };
        let geoms: Vec<_> = wins.iter().map(|w| window_geometry(&w.reach, &cp)).collect();
        let gref: Vec<&_> = geoms.iter().collect();
        let chain = Chain::build(&gref, &gaps, cp.seg_len_m);
        let lats: Vec<Lateral> = laterals
            .iter()
            .filter_map(|(id, h)| wins.iter().position(|w| w.id == *id).map(|j| Lateral { n0: chain.win_range[j].0, cnt: chain.win_range[j].1, hydro: h.clone() }))
            .collect();
        let sim = simulate(&chain, &hunt, &lats, argf(&args, "--q-base", 20.0), t_end, dt, &sol);
        println!(
            "1D router: {} nodes, {} steps, mass-balance closure {:+.4}% of inflow volume; adverse bed steps between windows (m): {:?}",
            chain.n(), sim.steps, 100.0 * sim.mass_error(), chain.adverse_steps_m.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>()
        );
        println!("  volumes: in {:.3e} out {:.3e} lat {:.3e} S0 {:.3e} S1 {:.3e} m3", sim.vol_in_m3, sim.vol_out_m3, sim.vol_lat_m3, sim.storage0_m3, sim.storage1_m3);
        let mut prof = String::from("node,window,section,dx_m,bed_m,eta_peak_m,q_peak_m3s\n");
        for (i, nd) in chain.nodes.iter().enumerate() {
            let (wl, sl) = nd.win.map(|(a, b)| (wins[a].id as i64, b as i64)).unwrap_or((-1, -1));
            prof += &format!("{i},{wl},{sl},{:.0},{:.3},{:.3},{:.1}\n", nd.dx, nd.bed, sim.eta_max[i], sim.q_max[i]);
        }
        fs::write(out.join(format!("{tag}_profile.csv")), prof).map_err(|e| e.to_string())?;
        for (j, w) in wins.iter().enumerate() {
            let (n0, cnt) = chain.win_range[j];
            let mid = n0 + cnt / 2;
            let qser: Vec<f64> = sim.q.iter().map(|r| r[mid]).collect();
            let hser: Vec<f64> = sim.eta.iter().map(|r| r[mid] - chain.nodes[mid].bed).collect();
            q_at.push(Hydrograph { t_h: sim.t_h.clone(), q: qser });
            stage_t.push(hser);
            let wse = window_wse(&chain, j, &geoms[j], &sim.eta_max);
            let elev_eff: Vec<f64> = (0..w.cat.n()).map(|i| w.reach.elev[i]).collect();
            depth_cells.push(connected_depth(&w.reach, &elev_eff, &wse));
            wse_cells.push(wse);
            let t = &chain.tables[chain.nodes[mid].table];
            let s = chain.slope(mid).sqrt();
            let hs: Vec<f64> = (0..t.a.len()).map(|k| k as f64 * t.dh).collect();
            rating_out.push((hs.clone(), t.a.clone(), t.b.clone(), t.k.iter().map(|k| k * s).collect()));
        }
        extra = format!(",\"router\":\"d1d\",\"mass_closure\":{:.6}", sim.mass_error());
    }

    // ---- per-window outputs
    let mut summary = String::from("[\n");
    for (j, w) in wins.iter().enumerate() {
        let (qp, tp) = q_at[j].peak();
        let (hp, _) = series_peak(&q_at[j].t_h, &stage_t[j]);
        let mut depth = depth_cells[j].clone();
        let wse = &wse_cells[j];
        let mut pluv_max = 0.0f64;
        if let (Some(r), true) = (&rain, pluvial != "none") {
            let (_hg, lap) = w.cat.laplacian()?;
            let wt = w.cat.convergence_weights(&lap, 0.6);
            let ldir = w.cat.downhill_operator();
            let mut p = Params::default();
            if pluvial == "cal" {
                p.k_dir = 1.0;
                p.d = 1.6;
                p.lam = 0.01;
            }
            let sim = simulate_pluvial(&w.cat, &lap, &ldir, &wt, r, &p);
            for i in 0..w.cat.n() {
                let pk = sim.iter().map(|s| s.1[i]).fold(0.0, f64::max);
                pluv_max = pluv_max.max(pk);
                depth[i] = depth[i].max(pk);
            }
        }
        let dir = out.join(format!("{tag}_w{}", w.id));
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut s = String::from("row,col,elev_m,pop,peak_depth_model_m,peak_depth_truth_m,peak_time_model_h,peak_time_truth_h,window_closes_model_h,window_closes_truth_h\n");
        for i in 0..w.cat.n() {
            let c = &w.cat.cells[i];
            s += &format!("{},{},{:.2},{:.0},{:.3},{:.3},{:.2},{:.2},,\n", c.row, c.col, c.elev_m, c.pop, depth[i], depth[i], tp, tp);
        }
        fs::write(dir.join("flood_evac_results_v2_seed1.csv"), s).map_err(|e| e.to_string())?;
        let mut ws = String::from("row,col,wse_m\n");
        for i in 0..w.cat.n() {
            ws += &format!("{},{},{:.3}\n", w.cat.cells[i].row, w.cat.cells[i].col, wse[i]);
        }
        fs::write(out.join(format!("{tag}_w{}_wse.csv", w.id)), ws).map_err(|e| e.to_string())?;
        let mut hs = String::from("hour,q_m3s,stage_m\n");
        for ((t, q), st) in q_at[j].t_h.iter().zip(&q_at[j].q).zip(&stage_t[j]) {
            hs += &format!("{t:.4},{q:.3},{st:.3}\n");
        }
        fs::write(out.join(format!("{tag}_w{}_hydro.csv", w.id)), hs).map_err(|e| e.to_string())?;
        write_reach_csv(&w.cat, &w.reach, &out.join(format!("reach_w{}.csv", w.id)))?;
        let (rh, ra, rb, rq) = &rating_out[j];
        let mut rs = String::from("stage_m,area_m2,top_width_m,q_m3s\n");
        for k in 0..rh.len() {
            rs += &format!("{:.3},{:.2},{:.1},{:.2}\n", rh[k], ra[k], rb[k], rq[k]);
        }
        fs::write(out.join(format!("rating_w{}.csv", w.id)), rs).map_err(|e| e.to_string())?;
        let wet = depth.iter().filter(|&&d| d > 0.10).count();
        let wse_thal = wse[w.reach.path[w.reach.path.len() / 2]];
        summary += &format!(
            "  {{\"id\":{},\"length_m\":{:.0},\"slope\":{:.5},\"sinuosity\":{:.3},\"gap_from_prev_m\":{:.0},\"q_peak_m3s\":{:.1},\"t_peak_h\":{:.2},\"stage_peak_m\":{:.3},\"wse_mid_thalweg_m\":{:.2},\"wet_cells_gt0p1\":{},\"pluvial_peak_max_m\":{:.3},\"volume_m3\":{:.0}{}}}{}\n",
            w.id, w.reach.length_m, w.reach.slope, w.reach.sinuosity(), gaps[j], qp, tp, hp, wse_thal, wet, pluv_max, q_at[j].volume_m3(), extra, if j + 1 < wins.len() { "," } else { "" }
        );
        println!(
            "w{}: L={:.0} m S={:.5} sinu={:.2} gap={:.0} m | Q_peak {:.0} m3/s at {:.2} h -> stage {:.2} m (WSE@mid-thalweg {:.1} m), {} cells >0.1 m",
            w.id, w.reach.length_m, w.reach.slope, w.reach.sinuosity(), gaps[j], qp, tp, hp, wse_thal, wet
        );
    }
    summary += "]\n";
    fs::write(out.join(format!("{tag}_summary.json")), summary).map_err(|e| e.to_string())?;
    println!("router {router}; in-window sinuosity used for gaps: {sinu:.3}; n_manning {n_man}; dt {dt} h; k_scale {k_scale}");
    Ok(())
}

fn simulate_pluvial(cat: &Catchment, lap: &nalgebra::DMatrix<f64>, ldir: &nalgebra::DMatrix<f64>, wt: &[f64], r: &Rainfall, p: &Params) -> Vec<(f64, Vec<f64>)> {
    flood_shpinn::truth::simulate(cat, lap, ldir, wt, r, p, *r.hours.last().unwrap(), 0.25)
}
