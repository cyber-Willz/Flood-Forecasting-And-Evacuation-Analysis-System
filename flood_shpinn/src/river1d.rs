//! v0.5 fluvial router: 1D local-inertial (Bates et al. 2010) flow along the thalweg chain, with per-section
//! *compound* (channel + floodplain) HAND geometry.
//!
//! What it changes relative to v0.4 (`river.rs`, still available as `--router mc`):
//!
//! * **Off-channel storage is dynamic.** v0.4 used one reach-averaged, uniform-flow rating per window and a
//!   constant-parameter Muskingum-Cunge router, which cannot attenuate a flood wave by more than a few percent without
//!   leaving its range of validity (v0.4 attenuated to 0.77 where the reported Hunt->Kerrville ratio is ~0.43). Here each
//!   ~400 m section carries its own area/width/conveyance tables built from the DEM's HAND field, the mass balance uses
//!   the full flooded width (channel + floodplain storage), and momentum is solved, so the wave both lags and attenuates
//!   from the same geometry.
//! * **Compound roughness.** Conveyance is the sum over a channel part (`HAND <= h_bf`, Manning `n_ch`) and a
//!   floodplain part (`HAND > h_bf`, Manning `n_fp`), so the stage-discharge relation changes slope at bankfull rather
//!   than using one scalar `n` (v0.4's rating was 0.5x low-flow / 1.6x flood error).
//! * **Slope-following water surface.** The solver returns a peak water-surface elevation per node; every cell is
//!   flooded to the surface interpolated along the channel, instead of one uniform stage per window (v0.4's
//!   "best uniform stage" floor of 0.85-2.6 m).
//! * Optional bathymetry carve (`carve`): lowers thalweg cells by a bankfull-depth estimate when building geometry
//!   (the DEM is a 100 m block mean and hides the channel), in the spirit of bathymetry-adjusted HAND.
//!
//! Numerics: explicit in time with `dt = alpha * dx / sqrt(g h)`; semi-implicit friction from the compound conveyance
//! (`S_f = Q|Q|/K^2`); the node update inverts the section's own area table (`A_new = A + dt * net / dx`), so volume
//! is conserved to round-off even where the flooded width jumps with stage. Boundaries: prescribed upstream discharge, normal-depth (open) downstream.

use crate::river::{Hydrograph, Reach};

pub const G: f64 = 9.81;
const NB8: [(isize, isize); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
const B_FLOOR: f64 = 5.0;

/// Geometry / conveyance parameters for building sections.
#[derive(Clone, Debug)]
pub struct ChannelParams {
    /// Manning n of the channel part (cells with HAND <= h_bf).
    pub n_ch: f64,
    /// Manning n of the floodplain part (cells with HAND > h_bf).
    pub n_fp: f64,
    /// Bankfull height above the local bed (m) separating channel from floodplain cells.
    pub h_bf: f64,
    /// Depth (m) subtracted from thalweg cells when building geometry only (bathymetry adjustment; 0 = none).
    pub carve: f64,
    /// Target section length (m).
    pub seg_len_m: f64,
    /// Table extent above the section bed (m) and spacing (m).
    pub hmax: f64,
    pub dh: f64,
}

impl Default for ChannelParams {
    fn default() -> Self {
        ChannelParams { n_ch: 0.05, n_fp: 0.10, h_bf: 3.0, carve: 0.0, seg_len_m: 400.0, hmax: 30.0, dh: 0.1 }
    }
}

/// Stage tables of one section on a uniform grid `h = i * dh` (above the section reference bed).
#[derive(Clone, Debug)]
pub struct SectionTable {
    pub dh: f64,
    /// Flow / storage area per unit channel length (m^2) = integral of `b`.
    pub a: Vec<f64>,
    /// Flooded top width per unit channel length (m).
    pub b: Vec<f64>,
    /// Compound conveyance (m^3/s per unit sqrt(slope)).
    pub k: Vec<f64>,
}

fn lin(v: &[f64], dh: f64, h: f64) -> f64 {
    if h <= 0.0 {
        return v[0];
    }
    let n = v.len();
    let x = h / dh;
    let i = x.floor() as usize;
    if i + 1 >= n {
        return v[n - 1] + (h - (n - 1) as f64 * dh) * (v[n - 1] - v[n - 2]) / dh;
    }
    let f = x - i as f64;
    v[i] * (1.0 - f) + v[i + 1] * f
}

impl SectionTable {
    pub fn area(&self, h: f64) -> f64 {
        lin(&self.a, self.dh, h)
    }
    pub fn conv(&self, h: f64) -> f64 {
        lin(&self.k, self.dh, h)
    }
    /// Exact inverse of `area` (the table is strictly increasing because widths are floored), so updating the stage
    /// through `A_new = A + dt * net_inflow / dx` conserves volume even where the width jumps with stage.
    pub fn stage_for_area(&self, a: f64) -> f64 {
        if a <= 0.0 {
            return 0.0;
        }
        let n = self.a.len();
        if a >= self.a[n - 1] {
            return (n - 1) as f64 * self.dh + (a - self.a[n - 1]) / self.b[n - 1].max(B_FLOOR);
        }
        let k = self.a.partition_point(|&v| v < a).max(1);
        let f = (a - self.a[k - 1]) / (self.a[k] - self.a[k - 1]).max(1e-300);
        ((k - 1) as f64 + f) * self.dh
    }
    /// Normal-depth stage (m above bed) for discharge `q` at bed slope `s`.
    pub fn stage_for_q(&self, q: f64, s: f64) -> f64 {
        let sq = s.max(1e-6).sqrt();
        let top = (self.a.len() - 1) as f64 * self.dh;
        if q <= 0.0 {
            return 0.0;
        }
        if self.conv(top) * sq <= q {
            return top;
        }
        let (mut lo, mut hi) = (0.0, top);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if self.conv(mid) * sq < q {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

/// Sections of one window plus the map from DEM cells to sections.
#[derive(Clone, Debug)]
pub struct WindowGeom {
    pub tables: Vec<SectionTable>,
    /// Reference bed of every section (non-increasing downstream).
    pub bed: Vec<f64>,
    /// Section length (m), uniform within the window.
    pub dx: f64,
    /// Section of the nearest thalweg cell, for every cell.
    pub seg_of_cell: Vec<usize>,
    /// Along-channel distance (m) of the nearest thalweg cell, for every cell.
    pub cell_x: Vec<f64>,
}

/// Connected depth field for a given water-surface field (flooded cells must connect to the thalweg).
pub fn connected_depth(r: &Reach, elev_eff: &[f64], wse: &[f64]) -> Vec<f64> {
    let n = r.elev.len();
    let d0: Vec<f64> = (0..n).map(|i| (wse[i] - elev_eff[i]).max(0.0)).collect();
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..n {
        if r.thal[i] && d0[i] > 0.0 {
            seen[i] = true;
            stack.push(i);
        }
    }
    while let Some(i) = stack.pop() {
        let (rr0, cc0) = ((i / r.ncols) as isize, (i % r.ncols) as isize);
        for (dr, dc) in NB8 {
            let (rr, cc) = (rr0 + dr, cc0 + dc);
            if rr < 0 || cc < 0 || rr >= r.nrows as isize || cc >= r.ncols as isize {
                continue;
            }
            let j = rr as usize * r.ncols + cc as usize;
            if !seen[j] && d0[j] > 0.0 {
                seen[j] = true;
                stack.push(j);
            }
        }
    }
    (0..n).map(|i| if seen[i] { d0[i] } else { 0.0 }).collect()
}

/// Build the compound sections of one window from its HAND field.
pub fn window_geometry(r: &Reach, p: &ChannelParams) -> WindowGeom {
    let n = r.elev.len();
    let cell2 = r.cell_m * r.cell_m;
    // cumulative path length (m)
    let mut cum = vec![0.0; r.path.len()];
    for k in 1..r.path.len() {
        let (a, b) = (r.path[k - 1], r.path[k]);
        let (ar, ac) = ((a / r.ncols) as f64, (a % r.ncols) as f64);
        let (br, bc) = ((b / r.ncols) as f64, (b % r.ncols) as f64);
        cum[k] = cum[k - 1] + ((ar - br).powi(2) + (ac - bc).powi(2)).sqrt() * r.cell_m;
    }
    let total = cum.last().cloned().unwrap_or(1.0).max(1.0);
    let nseg = ((total / p.seg_len_m).round() as usize).max(1);
    let dx = total / nseg as f64;
    let seg_of_pos: Vec<usize> = cum.iter().map(|&c| (((c / total) * nseg as f64).floor() as usize).min(nseg - 1)).collect();
    let mut pos = vec![usize::MAX; n];
    for (k, &i) in r.path.iter().enumerate() {
        pos[i] = k;
    }
    // reference bed = mean thalweg bed of the section, forced non-increasing
    let mut bed = vec![0.0; nseg];
    let mut cnt = vec![0usize; nseg];
    for (k, &i) in r.path.iter().enumerate() {
        bed[seg_of_pos[k]] += r.bed[i];
        cnt[seg_of_pos[k]] += 1;
    }
    for s in 0..nseg {
        bed[s] /= cnt[s].max(1) as f64;
    }
    for s in 1..nseg {
        bed[s] = bed[s].min(bed[s - 1]);
    }
    let seg_of_cell: Vec<usize> = (0..n).map(|i| seg_of_pos[pos[r.near[i]]]).collect();
    let cell_x: Vec<f64> = (0..n).map(|i| cum[pos[r.near[i]]]).collect();
    let elev_eff: Vec<f64> = (0..n).map(|i| if r.thal[i] { r.elev[i] - p.carve } else { r.elev[i] }).collect();
    let m = (p.hmax / p.dh).round() as usize;
    // per-stage accumulators
    let mut b_tab = vec![vec![0.0; m + 1]; nseg];
    let mut k_tab = vec![vec![0.0; m + 1]; nseg];
    for kk in 1..=m {
        let h = kk as f64 * p.dh;
        let wse: Vec<f64> = (0..n).map(|i| bed[seg_of_cell[i]] + h).collect();
        let d = connected_depth(r, &elev_eff, &wse);
        let mut sd = vec![[0.0f64; 2]; nseg];
        let mut sc = vec![[0.0f64; 2]; nseg];
        for i in 0..n {
            if d[i] > 0.0 {
                let cls = if r.hand[i] <= p.h_bf { 0 } else { 1 };
                sd[seg_of_cell[i]][cls] += d[i];
                sc[seg_of_cell[i]][cls] += 1.0;
            }
        }
        for s in 0..nseg {
            let mut kc = 0.0;
            for (cls, nman) in [(0usize, p.n_ch), (1usize, p.n_fp)] {
                let ac = sd[s][cls] * cell2 / dx;
                let bc = sc[s][cls] * cell2 / dx;
                if bc > 0.0 && ac > 0.0 {
                    kc += ac * (ac / bc).powf(2.0 / 3.0) / nman;
                }
            }
            b_tab[s][kk] = ((sc[s][0] + sc[s][1]) * cell2 / dx).max(B_FLOOR);
            k_tab[s][kk] = kc;
        }
    }
    let mut tables = Vec::with_capacity(nseg);
    for s in 0..nseg {
        let (mut b, mut k) = (b_tab[s].clone(), k_tab[s].clone());
        for kk in 1..=m {
            b[kk] = b[kk].max(b[kk - 1]);
            k[kk] = k[kk].max(k[kk - 1]);
        }
        b[0] = b[1];
        let mut a = vec![0.0; m + 1];
        for kk in 1..=m {
            a[kk] = a[kk - 1] + 0.5 * (b[kk - 1] + b[kk]) * p.dh;
        }
        tables.push(SectionTable { dh: p.dh, a, b, k });
    }
    WindowGeom { tables, bed, dx, seg_of_cell, cell_x }
}

/// One computational node (a section of the chain, or a gap section that borrows an adjacent table).
#[derive(Clone, Debug)]
pub struct ChainNode {
    pub table: usize,
    pub bed: f64,
    pub dx: f64,
    /// (window index, section index) for nodes that belong to a window, `None` for gap nodes.
    pub win: Option<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct Chain {
    pub tables: Vec<SectionTable>,
    pub nodes: Vec<ChainNode>,
    /// First node index and node count of each window.
    pub win_range: Vec<(usize, usize)>,
    /// Bed drop that had to be tolerated (m) where a window's first bed lies above the previous window's last bed.
    pub adverse_steps_m: Vec<f64>,
}

impl Chain {
    /// `gaps[j]` = river length (m) between window `j-1` and `j` (`gaps[0]` ignored).
    pub fn build(geoms: &[&WindowGeom], gaps: &[f64], target_dx: f64) -> Chain {
        let mut tables: Vec<SectionTable> = Vec::new();
        let mut nodes: Vec<ChainNode> = Vec::new();
        let mut win_range = Vec::new();
        let mut adverse = Vec::new();
        let mut tab_off = Vec::new();
        for g in geoms {
            tab_off.push(tables.len());
            tables.extend(g.tables.iter().cloned());
        }
        for (j, g) in geoms.iter().enumerate() {
            if j > 0 && gaps[j] > 1.0 {
                let prev = &nodes[nodes.len() - 1];
                let (bed_a, tab_a) = (prev.bed, prev.table);
                let bed_b = g.bed[0];
                let tab_b = tab_off[j];
                let ng = ((gaps[j] / target_dx).round() as usize).max(1);
                let dxg = gaps[j] / ng as f64;
                let s = ((bed_a - bed_b) / gaps[j]).max(1e-4);
                for k in 0..ng {
                    let bed = bed_a - s * (k as f64 + 1.0) * dxg;
                    nodes.push(ChainNode { table: if k < ng / 2 { tab_a } else { tab_b }, bed, dx: dxg, win: None });
                }
                let last = nodes[nodes.len() - 1].bed;
                adverse.push((bed_b - last).max(0.0));
            } else {
                adverse.push(0.0);
            }
            let start = nodes.len();
            for s in 0..g.tables.len() {
                nodes.push(ChainNode { table: tab_off[j] + s, bed: g.bed[s], dx: g.dx, win: Some((j, s)) });
            }
            win_range.push((start, g.tables.len()));
        }
        Chain { tables, nodes, win_range, adverse_steps_m: adverse }
    }
    pub fn n(&self) -> usize {
        self.nodes.len()
    }
    /// Bed slope at node `i` (m/m), floored at 1e-4 (central difference in the interior).
    pub fn slope(&self, i: usize) -> f64 {
        let n = self.n();
        if n < 2 {
            return 1e-3;
        }
        let nd = &self.nodes;
        let (drop, len) = if i == 0 {
            (nd[0].bed - nd[1].bed, 0.5 * (nd[0].dx + nd[1].dx))
        } else if i == n - 1 {
            (nd[n - 2].bed - nd[n - 1].bed, 0.5 * (nd[n - 2].dx + nd[n - 1].dx))
        } else {
            (nd[i - 1].bed - nd[i + 1].bed, 0.5 * nd[i - 1].dx + nd[i].dx + 0.5 * nd[i + 1].dx)
        };
        (drop / len.max(1.0)).max(1e-4)
    }
}

#[derive(Clone, Debug)]
pub struct Solver1D {
    /// Courant-like factor for `dt = alpha dx / sqrt(g h)`.
    pub alpha: f64,
    pub dt_max_s: f64,
}
impl Default for Solver1D {
    fn default() -> Self {
        Solver1D { alpha: 0.7, dt_max_s: 30.0 }
    }
}

/// Lateral inflow: total discharge (m3/s) spread over nodes `[n0, n0+cnt)` in proportion to their length.
#[derive(Clone, Debug)]
pub struct Lateral {
    pub n0: usize,
    pub cnt: usize,
    pub hydro: Hydrograph,
}

#[derive(Clone, Debug)]
pub struct Sim1D {
    pub t_h: Vec<f64>,
    /// `eta[s][node]` water-surface elevation at sample `s`.
    pub eta: Vec<Vec<f64>>,
    /// `q[s][node]` discharge at node centres (mean of adjacent interfaces).
    pub q: Vec<Vec<f64>>,
    pub eta_max: Vec<f64>,
    pub q_max: Vec<f64>,
    pub vol_in_m3: f64,
    pub vol_out_m3: f64,
    pub vol_lat_m3: f64,
    pub storage0_m3: f64,
    pub storage1_m3: f64,
    pub steps: usize,
}

impl Sim1D {
    /// Relative closure of the mass balance `(in + lat - out - dStorage) / max(in, 1)`.
    pub fn mass_error(&self) -> f64 {
        let res = self.vol_in_m3 + self.vol_lat_m3 - self.vol_out_m3 - (self.storage1_m3 - self.storage0_m3);
        res / self.vol_in_m3.max(1.0)
    }
}

pub fn simulate(chain: &Chain, inflow: &Hydrograph, laterals: &[Lateral], q_base: f64, t_end_h: f64, sample_dt_h: f64, sol: &Solver1D) -> Sim1D {
    let n = chain.n();
    let tab = |i: usize| &chain.tables[chain.nodes[i].table];
    let bed: Vec<f64> = chain.nodes.iter().map(|x| x.bed).collect();
    let dx: Vec<f64> = chain.nodes.iter().map(|x| x.dx).collect();
    let slope: Vec<f64> = (0..n).map(|i| chain.slope(i)).collect();
    // initial condition: steady normal depth at q_base everywhere
    let mut h: Vec<f64> = (0..n).map(|i| tab(i).stage_for_q(q_base, slope[i]).max(0.05)).collect();
    let mut qi = vec![q_base; n + 1]; // qi[0] upstream boundary, qi[n] downstream boundary, qi[i+1] between node i and i+1
    let storage = |h: &Vec<f64>| -> f64 { (0..n).map(|i| tab(i).area(h[i]) * dx[i]).sum() };
    let storage0 = storage(&h);
    let mut lat_share: Vec<Vec<(usize, f64)>> = Vec::new(); // per lateral: (node, fraction)
    for l in laterals {
        let tot: f64 = (l.n0..l.n0 + l.cnt).map(|i| dx[i]).sum();
        lat_share.push((l.n0..l.n0 + l.cnt).map(|i| (i, dx[i] / tot)).collect());
    }
    let t_end = t_end_h * 3600.0;
    let mut t = 0.0;
    let mut next_sample = 0.0;
    let mut out = Sim1D { t_h: vec![], eta: vec![], q: vec![], eta_max: vec![f64::NEG_INFINITY; n], q_max: vec![0.0; n], vol_in_m3: 0.0, vol_out_m3: 0.0, vol_lat_m3: 0.0, storage0_m3: storage0, storage1_m3: 0.0, steps: 0 };
    let dxmin = dx.iter().cloned().fold(f64::INFINITY, f64::min);
    loop {
        if t >= next_sample - 1e-9 {
            let eta: Vec<f64> = (0..n).map(|i| bed[i] + h[i]).collect();
            let q: Vec<f64> = (0..n).map(|i| 0.5 * (qi[i] + qi[i + 1])).collect();
            for i in 0..n {
                out.eta_max[i] = out.eta_max[i].max(eta[i]);
                out.q_max[i] = out.q_max[i].max(q[i]);
            }
            out.t_h.push(t / 3600.0);
            out.eta.push(eta);
            out.q.push(q);
            next_sample += sample_dt_h * 3600.0;
        }
        if t >= t_end - 1e-9 {
            break;
        }
        let hmax = h.iter().cloned().fold(1.0, f64::max);
        let mut dt = (sol.alpha * dxmin / (G * hmax).sqrt()).min(sol.dt_max_s);
        dt = dt.min(t_end - t).min((next_sample - t).max(1e-3));
        let eta: Vec<f64> = (0..n).map(|i| bed[i] + h[i]).collect();
        // interface momentum update
        let mut qn = qi.clone();
        for i in 0..n - 1 {
            let ef = eta[i].max(eta[i + 1]);
            let (h1, h2) = ((ef - bed[i]).max(0.0), (ef - bed[i + 1]).max(0.0));
            let af = 0.5 * (tab(i).area(h1) + tab(i + 1).area(h2));
            let kf = (0.5 * (tab(i).conv(h1) + tab(i + 1).conv(h2))).max(1e-3);
            let grad = (eta[i + 1] - eta[i]) / (0.5 * (dx[i] + dx[i + 1]));
            let q0 = qi[i + 1];
            qn[i + 1] = (q0 - G * af * dt * grad) / (1.0 + G * dt * q0.abs() * af / (kf * kf));
        }
        // boundaries at time t+dt
        qn[0] = inflow.at((t + dt) / 3600.0);
        qn[n] = tab(n - 1).conv(h[n - 1]) * slope[n - 1].sqrt();
        // node continuity
        let mut qlat = vec![0.0; n];
        for (l, sh) in laterals.iter().zip(&lat_share) {
            let ql = l.hydro.at((t + dt) / 3600.0);
            for &(i, f) in sh {
                qlat[i] += ql * f;
            }
            out.vol_lat_m3 += ql * dt;
        }
        // Positivity limiter: a node cannot pass on more water than it holds plus what arrives this step. Without it a
        // node that runs dry is clamped at zero depth, which silently creates volume. Upstream-first so that the limited
        // inflow of node i is what bounds its outflow.
        for i in 0..n {
            let avail = tab(i).area(h[i]) * dx[i] / dt + qn[i].max(0.0) + qlat[i].max(0.0);
            if qn[i + 1] > avail {
                qn[i + 1] = avail;
            }
            if i + 1 < n {
                // reverse flow leaves node i+1: bounded by what node i+1 holds
                let back = tab(i + 1).area(h[i + 1]) * dx[i + 1] / dt;
                if qn[i + 1] < -back {
                    qn[i + 1] = -back;
                }
            }
        }
        for i in 0..n {
            let a_new = tab(i).area(h[i]) + dt * (qn[i] - qn[i + 1] + qlat[i]) / dx[i];
            h[i] = tab(i).stage_for_area(a_new);
        }
        out.vol_in_m3 += qn[0] * dt;
        out.vol_out_m3 += qn[n] * dt;
        qi = qn;
        t += dt;
        out.steps += 1;
    }
    out.storage1_m3 = storage(&h);
    out
}

/// Peak water-surface envelope interpolated to every cell of a window along the channel.
pub fn window_wse(chain: &Chain, wi: usize, geom: &WindowGeom, eta_max: &[f64]) -> Vec<f64> {
    let (n0, cnt) = chain.win_range[wi];
    geom.cell_x
        .iter()
        .map(|&x| {
            let f = x / geom.dx - 0.5;
            let i0 = (f.floor().max(0.0) as usize).min(cnt - 1);
            let i1 = (i0 + 1).min(cnt - 1);
            let w = (f - i0 as f64).clamp(0.0, 1.0);
            eta_max[n0 + i0] * (1.0 - w) + eta_max[n0 + i1] * w
        })
        .collect()
}

/// Peak (value, time in hours) of a series sampled at `t_h`.
pub fn series_peak(t_h: &[f64], v: &[f64]) -> (f64, f64) {
    let mut b = (f64::NEG_INFINITY, 0.0);
    for (t, x) in t_h.iter().zip(v) {
        if *x > b.0 {
            b = (*x, *t);
        }
    }
    b
}
