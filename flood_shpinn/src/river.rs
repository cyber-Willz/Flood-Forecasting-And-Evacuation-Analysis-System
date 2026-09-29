//! Fluvial layer (v0.4): upstream-inflow / river-routing component.
//!
//! v0.3 is a rain-driven *ponding* model: every cell is fed only by rain that falls on it, so a river flood that is
//! generated upstream and arrives through the window boundary (the July-2025 Guadalupe flood at Hunt/Kerrville) cannot
//! appear. This module adds the missing physics without touching the ponding model:
//!
//! 1. [`Reach::extract`] finds the river thalweg through a window (least-cost path over the DEM from the lowest
//!    boundary cell far from the outlet to the outlet) and a non-increasing bed profile along it.
//! 2. HAND (height above nearest drainage) is taken against that thalweg, giving a *synthetic rating curve*
//!    ([`Rating`]): for stage `h` the connected inundated volume gives area `A = V/L`, hydraulic radius `R = A/B`,
//!    and Manning's `Q = A R^(2/3) S^(1/2) / n`. It is inverted to turn a discharge into a stage.
//! 3. [`route_mc`] routes an upstream inflow [`Hydrograph`] through a reach with the variable-parameter
//!    Muskingum-Cunge scheme (celerity and top width from the rating), optionally adding lateral inflow.
//! 4. [`Reach::inundation`] maps a stage to a depth field: `WSE_i = bed(nearest thalweg cell) + h`, depth
//!    `max(WSE_i - z_i, 0)`, kept only where hydraulically connected (8-neighbour) to the channel.
//!
//! Limits (see EVALUATION_v04.md): the DEM is a 100 m block mean (the true channel is under-resolved), stage is uniform
//! along a window (uniform-flow assumption; checked against the HWM water-surface slope), Manning `n` is a single
//! scalar, and the fluvial and pluvial layers are decoupled (max of the two) because the linear ponding model has no
//! momentum and cannot carry thousands of m3/s.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs;
use std::path::Path;

use crate::catchment::Catchment;

const NB8: [(isize, isize); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

#[derive(PartialEq)]
struct Node(f64, usize);
impl Eq for Node {}
impl Ord for Node {
    // reversed: BinaryHeap is a max-heap, we want the smallest cost first
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal).then(o.1.cmp(&self.1))
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// River channel inside one window plus HAND against it.
#[derive(Clone, Debug)]
pub struct Reach {
    pub nrows: usize,
    pub ncols: usize,
    pub cell_m: f64,
    pub elev: Vec<f64>,
    /// Thalweg cells, inlet -> outlet.
    pub path: Vec<usize>,
    pub thal: Vec<bool>,
    /// Bed elevation on thalweg cells (non-increasing downstream); NaN elsewhere.
    pub bed: Vec<f64>,
    /// Nearest thalweg cell of every cell (Euclidean, cell units).
    pub near: Vec<usize>,
    pub bed_near: Vec<f64>,
    /// `elev - bed_near`.
    pub hand: Vec<f64>,
    /// Channel length inside the window (m).
    pub length_m: f64,
    /// Bed slope over the window (m/m), floored at 1e-4.
    pub slope: f64,
    /// Straight-line inlet -> outlet distance (m); `length_m / straight_m` is the in-window sinuosity.
    pub straight_m: f64,
}

impl Reach {
    /// Thalweg extraction. Outlet = lowest cell with `role=outlet`; inlet = lowest boundary cell at least
    /// `min_sep_frac * max(nrows, ncols)` cells from the outlet. Cost = step length * (1 + ((z - zmin)/10 m)^2).
    pub fn extract(cat: &Catchment, min_sep_frac: f64) -> Result<Reach, String> {
        let (nr, nc) = (cat.nrows, cat.ncols);
        let elev: Vec<f64> = cat.cells.iter().map(|c| c.elev_m).collect();
        let outlet = cat
            .outlets()
            .into_iter()
            .min_by(|&a, &b| elev[a].partial_cmp(&elev[b]).unwrap())
            .ok_or("river: window has no outlet cell")?;
        let (orow, ocol) = (cat.cells[outlet].row as f64, cat.cells[outlet].col as f64);
        let min_sep = min_sep_frac * nr.max(nc) as f64;
        let inlet = (0..cat.n())
            .filter(|&i| {
                let c = &cat.cells[i];
                (c.row == 0 || c.col == 0 || c.row == nr - 1 || c.col == nc - 1)
                    && ((c.row as f64 - orow).powi(2) + (c.col as f64 - ocol).powi(2)).sqrt() >= min_sep
            })
            .min_by(|&a, &b| elev[a].partial_cmp(&elev[b]).unwrap())
            .ok_or("river: no boundary cell far enough from the outlet")?;
        let zmin = elev.iter().cloned().fold(f64::INFINITY, f64::min);
        let cost: Vec<f64> = elev.iter().map(|z| 1.0 + ((z - zmin) / 10.0).powi(2)).collect();
        let mut dist = vec![f64::INFINITY; cat.n()];
        let mut prev = vec![usize::MAX; cat.n()];
        let mut pq = BinaryHeap::new();
        dist[inlet] = 0.0;
        pq.push(Node(0.0, inlet));
        while let Some(Node(d, i)) = pq.pop() {
            if i == outlet {
                break;
            }
            if d > dist[i] {
                continue;
            }
            let (r, c) = (cat.cells[i].row as isize, cat.cells[i].col as isize);
            for (dr, dc) in NB8 {
                let (rr, cc) = (r + dr, c + dc);
                if rr < 0 || cc < 0 || rr >= nr as isize || cc >= nc as isize {
                    continue;
                }
                let j = cat.idx(rr as usize, cc as usize);
                let len = if dr != 0 && dc != 0 { std::f64::consts::SQRT_2 } else { 1.0 };
                let nd = d + len * 0.5 * (cost[i] + cost[j]);
                if nd < dist[j] {
                    dist[j] = nd;
                    prev[j] = i;
                    pq.push(Node(nd, j));
                }
            }
        }
        if !dist[outlet].is_finite() {
            return Err("river: outlet unreachable".into());
        }
        let mut path = vec![outlet];
        while *path.last().unwrap() != inlet {
            path.push(prev[*path.last().unwrap()]);
        }
        path.reverse();
        let mut thal = vec![false; cat.n()];
        let mut bed = vec![f64::NAN; cat.n()];
        let mut zrun = f64::INFINITY;
        let mut length = 0.0;
        for (k, &i) in path.iter().enumerate() {
            thal[i] = true;
            zrun = zrun.min(elev[i]);
            bed[i] = zrun;
            if k > 0 {
                let (a, b) = (&cat.cells[path[k - 1]], &cat.cells[i]);
                length += ((a.row as f64 - b.row as f64).powi(2) + (a.col as f64 - b.col as f64).powi(2)).sqrt() * cat.cell_m;
            }
        }
        let mut near = vec![0usize; cat.n()];
        let mut bed_near = vec![0.0; cat.n()];
        for i in 0..cat.n() {
            let (r, c) = (cat.cells[i].row as f64, cat.cells[i].col as f64);
            let mut best = (f64::INFINITY, path[0]);
            for &k in &path {
                let d2 = (cat.cells[k].row as f64 - r).powi(2) + (cat.cells[k].col as f64 - c).powi(2);
                if d2 < best.0 {
                    best = (d2, k);
                }
            }
            near[i] = best.1;
            bed_near[i] = bed[best.1];
        }
        let hand: Vec<f64> = (0..cat.n()).map(|i| elev[i] - bed_near[i]).collect();
        let (a, b) = (&cat.cells[inlet], &cat.cells[outlet]);
        let straight = ((a.row as f64 - b.row as f64).powi(2) + (a.col as f64 - b.col as f64).powi(2)).sqrt() * cat.cell_m;
        let slope = ((bed[inlet] - bed[outlet]) / length.max(1.0)).max(1e-4);
        Ok(Reach { nrows: nr, ncols: nc, cell_m: cat.cell_m, elev, path, thal, bed, near, bed_near, hand, length_m: length, slope, straight_m: straight })
    }

    pub fn inlet(&self) -> usize {
        self.path[0]
    }
    pub fn outlet(&self) -> usize {
        *self.path.last().unwrap()
    }
    pub fn sinuosity(&self) -> f64 {
        (self.length_m / self.straight_m.max(1.0)).max(1.0)
    }

    /// Depth and water-surface fields for stage `h` (m above the local thalweg bed), connected to the channel.
    pub fn inundation(&self, h: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.elev.len();
        let wse: Vec<f64> = (0..n).map(|i| self.bed_near[i] + h).collect();
        let d0: Vec<f64> = (0..n).map(|i| (wse[i] - self.elev[i]).max(0.0)).collect();
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        for i in 0..n {
            if self.thal[i] && d0[i] > 0.0 {
                seen[i] = true;
                stack.push(i);
            }
        }
        while let Some(i) = stack.pop() {
            let (r, c) = ((i / self.ncols) as isize, (i % self.ncols) as isize);
            for (dr, dc) in NB8 {
                let (rr, cc) = (r + dr, c + dc);
                if rr < 0 || cc < 0 || rr >= self.nrows as isize || cc >= self.ncols as isize {
                    continue;
                }
                let j = rr as usize * self.ncols + cc as usize;
                if !seen[j] && d0[j] > 0.0 {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        let depth: Vec<f64> = (0..n).map(|i| if seen[i] { d0[i] } else { 0.0 }).collect();
        (depth, wse)
    }

    /// Synthetic HAND rating curve with Manning roughness `n_man` (uniform flow at the reach-mean bed slope).
    pub fn rating(&self, n_man: f64, hmax: f64, dh: f64) -> Rating {
        let cell2 = self.cell_m * self.cell_m;
        let m = (hmax / dh).round() as usize;
        let (mut h, mut a, mut b, mut q) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut qmax: f64 = 0.0;
        for k in 1..=m {
            let hk = k as f64 * dh;
            let (d, _) = self.inundation(hk);
            let vol: f64 = d.iter().sum::<f64>() * cell2;
            let wet: f64 = d.iter().filter(|&&x| x > 0.0).count() as f64 * cell2;
            let area = vol / self.length_m;
            let width = (wet / self.length_m).max(1e-6);
            let rad = area / width;
            let qk = area * rad.powf(2.0 / 3.0) * self.slope.sqrt() / n_man;
            qmax = qmax.max(qk); // enforce a monotone table (connectivity can make Q(h) jump)
            h.push(hk);
            a.push(area);
            b.push(width);
            q.push(qmax);
        }
        Rating { h, a, b, q, n_man, slope: self.slope }
    }
}

#[derive(Clone, Debug)]
pub struct Rating {
    pub h: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub q: Vec<f64>,
    pub n_man: f64,
    pub slope: f64,
}

fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    let last = xs.len() - 1;
    if x >= xs[last] {
        return ys[last];
    }
    let k = xs.partition_point(|&v| v <= x) - 1;
    let f = (x - xs[k]) / (xs[k + 1] - xs[k]).max(1e-300);
    ys[k] * (1.0 - f) + ys[k + 1] * f
}

impl Rating {
    /// Stage (m above bed) for discharge `q` (m3/s); clamped at the top of the table.
    pub fn stage(&self, q: f64) -> f64 {
        if q <= 0.0 {
            return 0.0;
        }
        if q >= *self.q.last().unwrap() {
            return *self.h.last().unwrap();
        }
        // q table is non-decreasing but may have flats: use the first index reaching q
        let k = self.q.partition_point(|&v| v < q);
        if k == 0 {
            return self.h[0] * q / self.q[0].max(1e-300);
        }
        let f = (q - self.q[k - 1]) / (self.q[k] - self.q[k - 1]).max(1e-300);
        self.h[k - 1] * (1.0 - f) + self.h[k] * f
    }
    pub fn discharge(&self, h: f64) -> f64 {
        if h <= 0.0 {
            return 0.0;
        }
        if h < self.h[0] {
            return self.q[0] * h / self.h[0];
        }
        interp(&self.h, &self.q, h)
    }
    pub fn top_width(&self, q: f64) -> f64 {
        interp(&self.h, &self.b, self.stage(q))
    }
    /// Kinematic celerity `dQ/dA` (m/s) at discharge `q`, floored at 0.3 m/s.
    pub fn celerity(&self, q: f64) -> f64 {
        let h = self.stage(q);
        let k = self.h.partition_point(|&v| v < h).clamp(1, self.h.len() - 2);
        let (dq, da) = (self.q[k + 1] - self.q[k - 1], self.a[k + 1] - self.a[k - 1]);
        if da <= 0.0 || dq <= 0.0 {
            return 0.3;
        }
        (dq / da).max(0.3)
    }
}

/// Discharge time series (hours, m3/s), piecewise linear.
#[derive(Clone, Debug)]
pub struct Hydrograph {
    pub t_h: Vec<f64>,
    pub q: Vec<f64>,
}

impl Hydrograph {
    pub fn at(&self, t: f64) -> f64 {
        if t <= self.t_h[0] {
            return self.q[0];
        }
        let last = self.t_h.len() - 1;
        if t >= self.t_h[last] {
            return self.q[last];
        }
        let k = self.t_h.partition_point(|&v| v <= t) - 1;
        let f = (t - self.t_h[k]) / (self.t_h[k + 1] - self.t_h[k]);
        self.q[k] * (1.0 - f) + self.q[k + 1] * f
    }
    pub fn peak(&self) -> (f64, f64) {
        let mut b = (f64::NEG_INFINITY, 0.0);
        for (t, q) in self.t_h.iter().zip(&self.q) {
            if *q > b.0 {
                b = (*q, *t);
            }
        }
        b
    }
    /// Trapezoid volume (m3).
    pub fn volume_m3(&self) -> f64 {
        self.t_h.windows(2).zip(self.q.windows(2)).map(|(t, q)| 0.5 * (q[0] + q[1]) * (t[1] - t[0]) * 3600.0).sum()
    }
    /// Nash-type flood wave `q = base + (peak-base) * (x e^(1-x))^shape`, `x = (t - t0)/tp` for `t >= t0`.
    pub fn nash(q_base: f64, q_peak: f64, t0_h: f64, tp_h: f64, shape: f64, t_end_h: f64, dt_h: f64) -> Hydrograph {
        let n = (t_end_h / dt_h).round() as usize;
        let (mut t_h, mut q) = (Vec::new(), Vec::new());
        for k in 0..=n {
            let t = k as f64 * dt_h;
            let v = if t <= t0_h {
                q_base
            } else {
                let x = (t - t0_h) / tp_h;
                q_base + (q_peak - q_base) * (x * (1.0 - x).exp()).powf(shape)
            };
            t_h.push(t);
            q.push(v);
        }
        Hydrograph { t_h, q }
    }
    pub fn load_csv(path: &Path) -> Result<Hydrograph, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (mut t, mut q) = (Vec::new(), Vec::new());
        let mut header = false;
        for (ln, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if !header {
                header = true;
                continue;
            }
            let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if f.len() < 2 {
                return Err(format!("inflow csv line {}: need hour,q_m3s", ln + 1));
            }
            t.push(f[0].parse::<f64>().map_err(|e| format!("line {}: {e}", ln + 1))?);
            q.push(f[1].parse::<f64>().map_err(|e| format!("line {}: {e}", ln + 1))?.max(0.0));
        }
        if t.len() < 2 || t.windows(2).any(|w| w[1] <= w[0]) {
            return Err("inflow csv: need >=2 strictly increasing hours".into());
        }
        Ok(Hydrograph { t_h: t, q })
    }
    pub fn write_csv(&self, path: &Path) -> Result<(), String> {
        let mut s = String::from("hour,q_m3s\n");
        for (t, q) in self.t_h.iter().zip(&self.q) {
            s += &format!("{t:.4},{q:.3}\n");
        }
        fs::write(path, s).map_err(|e| e.to_string())
    }
}

/// Muskingum-Cunge routing of `inflow` through a reach of `length_m` and bed `slope`.
/// The reach is split into sub-reaches no shorter than `c * dt` so that `K >= dt`; `K = dx/c`,
/// `X = 0.5 (1 - Q/(B c S dx))` clamped to `[0, min(0.5, dt/2K)]` (the standard non-negative-coefficient limit).
/// `k_scale` multiplies `K` (1 = pure rating-derived storage; used only for sensitivity). `lateral` (m3/s, total over
/// the reach) is added uniformly along the sub-reaches.
///
/// `variable = false` (default in `river_eval`): constant coefficients per sub-reach, evaluated at half the peak
/// discharge of that sub-reach's inflow. C1+C2+C3 = 1 with fixed C's, so volume is conserved to numerical precision.
/// `variable = true`: coefficients re-evaluated every step from the local discharge. This follows the nonlinear
/// celerity better but is NOT mass conserving (tests measured +8 % volume on a 1,500 m3/s wave); use only for
/// sensitivity runs.
pub fn route_mc(inflow: &Hydrograph, lateral: Option<&Hydrograph>, rating: &Rating, length_m: f64, slope: f64, dt_h: f64, t_end_h: f64, k_scale: f64, variable: bool) -> Hydrograph {
    let nt = (t_end_h / dt_h).round() as usize;
    let dt_s = dt_h * 3600.0;
    let t_grid: Vec<f64> = (0..=nt).map(|k| k as f64 * dt_h).collect();
    let mut cur: Vec<f64> = t_grid.iter().map(|&t| inflow.at(t)).collect();
    let qpk = cur.iter().cloned().fold(0.0, f64::max);
    let c_ref = rating.celerity((0.5 * qpk).max(1.0));
    let n_sub = ((length_m / (c_ref * dt_s)).floor() as usize).max(1);
    let dx = length_m / n_sub as f64;
    for _ in 0..n_sub {
        let mut out = vec![0.0; nt + 1];
        out[0] = cur[0];
        let q_half = (0.5 * cur.iter().cloned().fold(0.0, f64::max)).max(1.0);
        for k in 0..nt {
            let qref = if variable { ((cur[k] + cur[k + 1] + out[k]) / 3.0).max(1e-3) } else { q_half };
            let c = rating.celerity(qref);
            let b = rating.top_width(qref).max(1.0);
            let kk = (k_scale * dx / c / 3600.0).max(1e-6); // hours
            let x = (0.5 * (1.0 - qref / (b * c * slope.max(1e-5) * dx))).clamp(0.0, (0.5f64).min(dt_h / (2.0 * kk)));
            let dd = 2.0 * kk * (1.0 - x) + dt_h;
            let (c1, c2, mut c3) = ((dt_h - 2.0 * kk * x) / dd, (dt_h + 2.0 * kk * x) / dd, (2.0 * kk * (1.0 - x) - dt_h) / dd);
            let mut c1 = c1;
            let mut c2 = c2;
            if c3 < 0.0 {
                // K < dt/2(1-X): fall back to pure translation-with-averaging (still conservative)
                c3 = 0.0;
                let s = c1 + c2;
                c1 /= s;
                c2 /= s;
            }
            out[k + 1] = (c1 * cur[k + 1] + c2 * cur[k] + c3 * out[k]).max(0.0);
        }
        if let Some(l) = lateral {
            for k in 0..=nt {
                out[k] += l.at(t_grid[k]) / n_sub as f64;
            }
        }
        cur = out;
    }
    Hydrograph { t_h: t_grid, q: cur }
}

/// Write a per-cell reach table (`row,col,elev_m,thalweg,bed_near_m,hand_m`) for external analysis.
pub fn write_reach_csv(cat: &Catchment, r: &Reach, path: &Path) -> Result<(), String> {
    let mut s = String::from("row,col,elev_m,thalweg,bed_near_m,hand_m\n");
    for i in 0..cat.n() {
        let c = &cat.cells[i];
        s += &format!("{},{},{:.3},{},{:.3},{:.3}\n", c.row, c.col, c.elev_m, r.thal[i] as u8, r.bed_near[i], r.hand[i]);
    }
    fs::write(path, s).map_err(|e| e.to_string())
}

pub fn write_rating_csv(rt: &Rating, path: &Path) -> Result<(), String> {
    let mut s = String::from("stage_m,area_m2,top_width_m,q_m3s\n");
    for k in 0..rt.h.len() {
        s += &format!("{:.3},{:.2},{:.1},{:.2}\n", rt.h[k], rt.a[k], rt.b[k], rt.q[k]);
    }
    fs::write(path, s).map_err(|e| e.to_string())
}
