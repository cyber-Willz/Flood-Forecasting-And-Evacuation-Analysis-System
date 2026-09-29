//! Reduced-order runoff model and its independent RK4 solution.
//!
//! State `s = h / depth_scale` (depth in units of `depth_scale` metres):
//!
//! ```text
//! ds/dt = w_i * q_i(t) - lam * s - D * (Delta s)_i - k_dir * (L s)_i   (interior cells)
//! s_i   = 0                                                             (outlet cells)
//! ```
//!
//! v0.3 adds `k_dir * L s`, a conservative *downhill routing* operator (see
//! [`crate::catchment::Catchment::downhill_operator`]). `Delta` is symmetric and cannot push water
//! in a preferred direction; `L` moves each cell's water to its strictly-lower neighbours in
//! proportion to elevation drop, so runoff accumulates in valleys and floodplains. `k_dir = 0`
//! reproduces the v0.2 model exactly.
//!
//! `q_i(t) = rain_mm_h(t)/1000 * (0.2 + 0.8*imperv_i) / depth_scale` is runoff
//! generation, `lam` is infiltration/storm-drain loss, `D * Delta` is lateral
//! redistribution along the hypergraph. `-Delta` has non-positive
//! off-diagonals, so the system is Metzler: with `s(0)=0` and `q>=0`, depths
//! stay non-negative without clamping.

use nalgebra::{DMatrix, DVector};

use crate::catchment::Catchment;
use crate::rainfall::Rainfall;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Lateral redistribution rate (1/h).
    pub d: f64,
    /// Infiltration / drain loss rate (1/h).
    pub lam: f64,
    /// Metres per state unit.
    pub depth_scale: f64,
    /// Downhill routing rate (1/h). 0 = v0.2 (symmetric diffusion only).
    pub k_dir: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params { d: 0.35, lam: 0.06, depth_scale: 0.1, k_dir: 0.0 }
    }
}

pub fn source(cat: &Catchment, w: &[f64], rain: &Rainfall, t: f64, p: &Params) -> DVector<f64> {
    let r = rain.at(t) / 1000.0;
    DVector::from_iterator(
        cat.n(),
        cat.cells.iter().zip(w).map(|(c, &wi)| if c.outlet { 0.0 } else { wi * r * (0.2 + 0.8 * c.imperv) / p.depth_scale }),
    )
}

/// Row-compressed sparse matrix (the Laplacian and routing operator are ~99% zeros).
struct Csr(Vec<Vec<(usize, f64)>>);
impl Csr {
    fn from_dense(m: &DMatrix<f64>) -> Self {
        Csr((0..m.nrows()).map(|i| (0..m.ncols()).filter_map(|j| { let v = m[(i, j)]; if v.abs() > 1e-14 { Some((j, v)) } else { None } }).collect()).collect())
    }
    fn mul(&self, x: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(self.0.len(), self.0.iter().map(|row| row.iter().map(|&(j, v)| v * x[j]).sum::<f64>()))
    }
}

fn deriv(cat: &Catchment, lap: &Csr, ldir: &Csr, w: &[f64], rain: &Rainfall, p: &Params, t: f64, s: &DVector<f64>) -> DVector<f64> {
    let mut ds = source(cat, w, rain, t, p) - s * p.lam - lap.mul(s) * p.d - ldir.mul(s) * p.k_dir;
    for (i, c) in cat.cells.iter().enumerate() {
        if c.outlet {
            ds[i] = 0.0;
        }
    }
    ds
}

/// RK4 from `s(0)=0`; returns `(t_hours, depth_metres_per_cell)` every `out_h`.
pub fn simulate(cat: &Catchment, lap: &DMatrix<f64>, ldir: &DMatrix<f64>, w: &[f64], rain: &Rainfall, p: &Params, t_end: f64, out_h: f64) -> Vec<(f64, Vec<f64>)> {
    simulate_sub(cat, lap, ldir, w, rain, p, t_end, out_h, 10)
}

/// As [`simulate`] with `sub` RK4 sub-steps per output interval (calibration uses fewer).
pub fn simulate_sub(cat: &Catchment, lap: &DMatrix<f64>, ldir: &DMatrix<f64>, w: &[f64], rain: &Rainfall, p: &Params, t_end: f64, out_h: f64, sub: usize) -> Vec<(f64, Vec<f64>)> {
    let (lap, ldir) = (&Csr::from_dense(lap), &Csr::from_dense(ldir));
    let dt = out_h / sub as f64;
    let nout = (t_end / out_h).round() as usize;
    let mut s = DVector::<f64>::zeros(cat.n());
    let mut out = vec![(0.0, s.iter().map(|v| v * p.depth_scale).collect())];
    let mut t = 0.0;
    for _ in 0..nout {
        for _ in 0..sub {
            let k1 = deriv(cat, lap, ldir, w, rain, p, t, &s);
            let k2 = deriv(cat, lap, ldir, w, rain, p, t + dt / 2.0, &(&s + &k1 * (dt / 2.0)));
            let k3 = deriv(cat, lap, ldir, w, rain, p, t + dt / 2.0, &(&s + &k2 * (dt / 2.0)));
            let k4 = deriv(cat, lap, ldir, w, rain, p, t + dt, &(&s + &k3 * dt));
            s += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
            t += dt;
        }
        out.push((t, s.iter().map(|v| v * p.depth_scale).collect()));
    }
    out
}
