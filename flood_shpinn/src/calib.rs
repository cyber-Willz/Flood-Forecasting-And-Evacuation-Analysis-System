//! Reference-mask agreement + parameter calibration (v0.3).
//!
//! A *reference mask* is a per-cell boolean "known flood-prone" layer on the same grid as `cells.csv`:
//! `row,col,...,ref_flood` (or `flooded`) with 0/1 in that column. It can be observed data (FEMA NFHL
//! SFHA rasterised with `real_data/prep_observed_fema.py`, USGS high-water-mark cells, a satellite
//! flood extent) or, when none is available offline, the terrain-derived HAND proxy from
//! `real_data/make_reference.py`. The code does not care which; the *interpretation* of the numbers does.
//!
//! Skill is measured by ROC-AUC of the peak-depth map against the mask (threshold-free: "do the model's
//! deepest cells fall in the known flood-prone cells?") plus CSI/POD/FAR at the evacuation depth
//! threshold and at matched prevalence (threshold set so the model marks as many cells as the mask).
//! Outlet cells are excluded (their depth is pinned to 0 by the boundary condition).

use std::fs;
use std::path::Path;

use nalgebra::DMatrix;

use crate::catchment::Catchment;
use crate::rainfall::Rainfall;
use crate::truth::{simulate_sub, Params};

#[derive(Clone, Debug)]
pub struct RefMask {
    pub flood: Vec<bool>,
}

impl RefMask {
    pub fn load_csv(path: &Path, cat: &Catchment) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#'));
        let header: Vec<&str> = lines.next().ok_or("reference csv empty")?.split(',').map(str::trim).collect();
        let col = header.iter().position(|h| *h == "ref_flood" || *h == "flooded").ok_or("reference csv needs a ref_flood or flooded column")?;
        let mut flood = vec![false; cat.n()];
        let mut seen = 0usize;
        for l in lines {
            let f: Vec<&str> = l.split(',').map(str::trim).collect();
            let (r, c): (usize, usize) = (f[0].parse().map_err(|e| format!("{e}"))?, f[1].parse().map_err(|e| format!("{e}"))?);
            if r < cat.nrows && c < cat.ncols {
                flood[cat.idx(r, c)] = f[col].parse::<f64>().map_err(|e| format!("{e}"))? > 0.5;
                seen += 1;
            }
        }
        if seen != cat.n() {
            return Err(format!("reference csv covers {seen} cells, grid has {}", cat.n()));
        }
        Ok(RefMask { flood })
    }
}

pub fn peak_map(series: &[(f64, Vec<f64>)]) -> Vec<f64> {
    let n = series[0].1.len();
    (0..n).map(|i| series.iter().map(|s| s.1[i]).fold(0.0, f64::max)).collect()
}

/// ROC-AUC via the Mann-Whitney U statistic with mid-ranks for ties.
pub fn auc(score: &[f64], label: &[bool]) -> f64 {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap());
    let mut rank = vec![0.0; score.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && score[idx[j + 1]] == score[idx[i]] {
            j += 1;
        }
        let mid = 0.5 * (i + j) as f64 + 1.0;
        for k in i..=j {
            rank[idx[k]] = mid;
        }
        i = j + 1;
    }
    let np = label.iter().filter(|&&l| l).count() as f64;
    let nn = label.len() as f64 - np;
    if np == 0.0 || nn == 0.0 {
        return f64::NAN;
    }
    let sum_pos: f64 = label.iter().enumerate().filter(|(_, &l)| l).map(|(i, _)| rank[i]).sum();
    (sum_pos - np * (np + 1.0) / 2.0) / (np * nn)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Contingency {
    pub csi: f64,
    pub pod: f64,
    pub far: f64,
}

pub fn contingency(score: &[f64], label: &[bool], thr: f64) -> Contingency {
    let (mut tp, mut fp, mut fnn) = (0usize, 0usize, 0usize);
    for (s, &l) in score.iter().zip(label) {
        match (*s > thr, l) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fnn += 1,
            _ => {}
        }
    }
    let d = |a: usize, b: usize| if b == 0 { f64::NAN } else { a as f64 / b as f64 };
    Contingency { csi: d(tp, tp + fp + fnn), pod: d(tp, tp + fnn), far: d(fp, tp + fp) }
}

/// CSI when the model marks exactly as many cells (the deepest ones) as the reference has.
pub fn matched_prevalence_csi(score: &[f64], label: &[bool]) -> f64 {
    let k = label.iter().filter(|&&l| l).count();
    if k == 0 || k >= score.len() {
        return f64::NAN;
    }
    let mut s: Vec<f64> = score.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    contingency(score, label, s[k] ).csi
}

/// Restrict `score`/`label` to the cells selected by `keep`.
pub fn subset(score: &[f64], label: &[bool], keep: &[bool]) -> (Vec<f64>, Vec<bool>) {
    let mut s = Vec::new();
    let mut l = Vec::new();
    for i in 0..score.len() {
        if keep[i] {
            s.push(score[i]);
            l.push(label[i]);
        }
    }
    (s, l)
}

#[derive(Clone, Debug)]
pub struct CalibResult {
    pub params: Params,
    pub train_auc: f64,
    pub tried: usize,
}

/// 99th percentile of peak depth over non-outlet cells (the plausibility guard's statistic).
pub fn p99_peak(cat: &Catchment, peak: &[f64]) -> f64 {
    let mut v: Vec<f64> = (0..cat.n()).filter(|&i| !cat.cells[i].outlet).map(|i| peak[i]).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 * 0.99) as usize).min(v.len() - 1)]
}

fn eval_params(cat: &Catchment, lap: &DMatrix<f64>, ldir: &DMatrix<f64>, w: &[f64], rain: &Rainfall, p: &Params, t_end: f64, r: &RefMask, keep: &[bool]) -> (f64, f64) {
    let sim = simulate_sub(cat, lap, ldir, w, rain, p, t_end, 0.25, 4);
    let peak = peak_map(&sim);
    let (s, l) = subset(&peak, &r.flood, keep);
    (auc(&s, &l), p99_peak(cat, &peak))
}

/// Exhaustive grid search over (k_dir, D, lam) maximising peak-depth AUC on the cells in `keep`.
/// The grid is deliberately small and physical (routing time ~ 1 cell / 0.2-2 h; loss 2-30 %/h);
/// ties are broken toward the default parameters so the fit never wanders without evidence.
///
/// **Plausibility guard**: AUC only ranks cells, so on its own it rewards parameter sets that pile the
/// whole catchment's runoff into a few valley cells (cell-mean depths of many metres). Candidates whose
/// 99th-percentile peak depth exceeds `max_depth_m` are rejected. The cap is a sanity bound, not an
/// observation; replace it with measured depths (USGS high-water marks, gauge stage) when you have them.
pub fn calibrate(cat: &Catchment, lap: &DMatrix<f64>, ldir: &DMatrix<f64>, w: &[f64], rain: &Rainfall, base: &Params, t_end: f64, r: &RefMask, keep: &[bool], max_depth_m: f64) -> CalibResult {
    let (mut best, mut best_auc, mut tried) = (*base, f64::MIN, 0usize);
    let dist = |p: &Params| ((p.k_dir - base.k_dir) / 3.0).abs() + ((p.d - base.d) / 0.5).abs() + ((p.lam - base.lam) / 0.15).abs();
    for &k in &[0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0] {
        for &d in &[0.1, 0.2, 0.35, 0.6, 1.0, 1.6] {
            for &lam in &[0.01, 0.02, 0.06, 0.15, 0.3] {
                let p = Params { k_dir: k, d, lam, ..*base };
                let (a, p99) = eval_params(cat, lap, ldir, w, rain, &p, t_end, r, keep);
                tried += 1;
                if p99 > max_depth_m {
                    continue;
                }
                if a.is_finite() && (a > best_auc + 1e-4 || ((a - best_auc).abs() <= 1e-4 && dist(&p) < dist(&best))) {
                    best_auc = a;
                    best = p;
                }
            }
        }
    }
    CalibResult { params: best, train_auc: best_auc, tried }
}
