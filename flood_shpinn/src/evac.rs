//! Time-dependent evacuation routing over forecast depth fields.
//!
//! Roads = 4-connected cell adjacency. A cell can be *entered* at time `t`
//! only if its depth at `t` is <= `thr_m`. Travel time per cell grows with
//! depth. For each populated cell we scan departure times and report when
//! the window to reach a shelter first closes.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::catchment::Catchment;

pub struct DepthSeries {
    pub times_h: Vec<f64>,
    pub depth_m: Vec<Vec<f64>>,
}

impl DepthSeries {
    pub fn at(&self, t: f64, i: usize) -> f64 {
        let ts = &self.times_h;
        if t <= ts[0] {
            return self.depth_m[0][i];
        }
        if t >= *ts.last().unwrap() {
            return self.depth_m[ts.len() - 1][i];
        }
        let k = ts.partition_point(|&x| x <= t) - 1;
        let f = (t - ts[k]) / (ts[k + 1] - ts[k]);
        self.depth_m[k][i] * (1.0 - f) + self.depth_m[k + 1][i] * f
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EvacParams {
    /// Max passable depth (m). 0.15 m is a common vehicle/pedestrian limit.
    pub thr_m: f64,
    pub minutes_per_cell: f64,
    pub step_min: f64,
    pub horizon_h: f64,
}

impl EvacParams {
    /// Travel time per cell from the cell edge length and a convoy speed (km/h),
    /// instead of a fixed 2 min/cell that silently assumes 250 m cells.
    pub fn for_cell(cell_m: f64, speed_kmh: f64, horizon_h: f64) -> Self {
        EvacParams { minutes_per_cell: cell_m / (speed_kmh * 1000.0 / 60.0), horizon_h, ..Default::default() }
    }
}

impl Default for EvacParams {
    fn default() -> Self {
        EvacParams { thr_m: 0.15, minutes_per_cell: 2.0, step_min: 5.0, horizon_h: 6.0 }
    }
}

#[derive(Clone, Debug)]
pub struct CellPlan {
    pub cell: usize,
    pub pop: f64,
    /// First departure time (h) at which no passable route reaches a
    /// shelter; `None` if the window never closes within the horizon.
    pub window_closes_h: Option<f64>,
    /// Best-case travel time (min) when leaving at t=0.
    pub travel_min_at_0: Option<f64>,
}

/// Earliest shelter-arrival time (seconds) leaving `start` at `t0_s`, or None.
fn earliest_arrival(cat: &Catchment, shelters: &[bool], ds: &DepthSeries, ep: &EvacParams, start: usize, t0_s: i64) -> Option<i64> {
    let passable = |i: usize, t_s: i64| ds.at(t_s as f64 / 3600.0, i) <= ep.thr_m;
    if !passable(start, t0_s) {
        return None;
    }
    let mut best = vec![i64::MAX; cat.n()];
    let mut heap = BinaryHeap::new();
    best[start] = t0_s;
    heap.push(Reverse((t0_s, start)));
    while let Some(Reverse((t, i))) = heap.pop() {
        if t > best[i] {
            continue;
        }
        if shelters[i] {
            return Some(t);
        }
        for j in cat.neighbours(i) {
            let d = ds.at(t as f64 / 3600.0, j);
            let step = ep.minutes_per_cell * 60.0 * (1.0 + 2.0 * (d / ep.thr_m).min(1.0));
            let ta = t + step as i64;
            if ta < best[j] && passable(j, ta) {
                best[j] = ta;
                heap.push(Reverse((ta, j)));
            }
        }
    }
    None
}

pub fn plan(cat: &Catchment, ds: &DepthSeries, ep: &EvacParams) -> Vec<CellPlan> {
    let steps = (ep.horizon_h * 60.0 / ep.step_min).round() as usize;
    let shelters: Vec<bool> = cat.cells.iter().map(|c| c.shelter).collect();
    let mut out = Vec::new();
    for (i, c) in cat.cells.iter().enumerate() {
        if c.pop <= 0.0 || c.shelter || c.outlet {
            continue;
        }
        let mut closes = None;
        let mut travel0 = None;
        for k in 0..=steps {
            let t0 = (k as f64 * ep.step_min * 60.0) as i64;
            match earliest_arrival(cat, &shelters, ds, ep, i, t0) {
                Some(ta) => {
                    if k == 0 {
                        travel0 = Some((ta - t0) as f64 / 60.0);
                    }
                }
                None => {
                    closes = Some(k as f64 * ep.step_min / 60.0);
                    break;
                }
            }
        }
        out.push(CellPlan { cell: i, pop: c.pop, window_closes_h: closes, travel_min_at_0: travel0 });
    }
    out
}
