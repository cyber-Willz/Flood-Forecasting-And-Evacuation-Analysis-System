//! GIS cell table -> catchment -> spectral hypergraph.
//!
//! Input format (CSV, header required, `#` comment lines allowed):
//! `row,col,elev_m,pop,imperv,role` where `role` is empty, `outlet` or
//! `shelter`. One row per raster cell of the study area (regular grid).

use std::fs;
use std::path::Path;

use nalgebra::{DMatrix, DVector};
use spectral_hypergraph::hypergraph::HypergraphBuilder;
use spectral_hypergraph::laplacian::dense_normalized_laplacian;
use spectral_hypergraph::SpectralHypergraph;

#[derive(Clone, Debug)]
pub struct Cell {
    pub row: usize,
    pub col: usize,
    pub elev_m: f64,
    pub pop: f64,
    /// Impervious fraction in [0, 1] (drives runoff coefficient).
    pub imperv: f64,
    pub outlet: bool,
    pub shelter: bool,
}

#[derive(Clone, Debug)]
pub struct Catchment {
    pub nrows: usize,
    pub ncols: usize,
    /// Cell edge length in metres (used only by evacuation travel times).
    pub cell_m: f64,
    pub cells: Vec<Cell>,
}

impl Catchment {
    pub fn idx(&self, r: usize, c: usize) -> usize {
        r * self.ncols + c
    }
    pub fn n(&self) -> usize {
        self.cells.len()
    }
    pub fn outlets(&self) -> Vec<usize> {
        (0..self.n()).filter(|&i| self.cells[i].outlet).collect()
    }
    pub fn shelters(&self) -> Vec<usize> {
        (0..self.n()).filter(|&i| self.cells[i].shelter).collect()
    }
    /// 4-connected neighbours (the road/flow-path adjacency).
    pub fn neighbours(&self, i: usize) -> Vec<usize> {
        let (r, c) = (self.cells[i].row as isize, self.cells[i].col as isize);
        let mut v = Vec::with_capacity(4);
        for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (rr, cc) = (r + dr, c + dc);
            if rr >= 0 && cc >= 0 && (rr as usize) < self.nrows && (cc as usize) < self.ncols {
                v.push(self.idx(rr as usize, cc as usize));
            }
        }
        v
    }

    pub fn load_csv(path: &Path, cell_m: f64) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut rows: Vec<(usize, usize, Cell)> = Vec::new();
        let mut seen_header = false;
        for (ln, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if !seen_header {
                seen_header = true;
                if !line.starts_with("row") {
                    return Err("GIS csv: header must be row,col,elev_m,pop,imperv,role".into());
                }
                continue;
            }
            let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if f.len() < 5 {
                return Err(format!("GIS csv line {}: expected >=5 fields", ln + 1));
            }
            let pu = |s: &str| s.parse::<usize>().map_err(|e| format!("line {}: {e}", ln + 1));
            let pf = |s: &str| s.parse::<f64>().map_err(|e| format!("line {}: {e}", ln + 1));
            let (row, col) = (pu(f[0])?, pu(f[1])?);
            let role = f.get(5).copied().unwrap_or("");
            rows.push((
                row,
                col,
                Cell {
                    row,
                    col,
                    elev_m: pf(f[2])?,
                    pop: pf(f[3])?,
                    imperv: pf(f[4])?.clamp(0.0, 1.0),
                    outlet: role.eq_ignore_ascii_case("outlet"),
                    shelter: role.eq_ignore_ascii_case("shelter"),
                },
            ));
        }
        let nrows = rows.iter().map(|r| r.0).max().ok_or("GIS csv is empty")? + 1;
        let ncols = rows.iter().map(|r| r.1).max().unwrap() + 1;
        if rows.len() != nrows * ncols {
            return Err(format!("GIS csv: {} rows but grid is {nrows}x{ncols}", rows.len()));
        }
        rows.sort_by_key(|r| (r.0, r.1));
        let cells: Vec<Cell> = rows.into_iter().map(|r| r.2).collect();
        let cat = Catchment { nrows, ncols, cell_m, cells };
        if cat.outlets().is_empty() {
            return Err("GIS csv: at least one cell needs role=outlet".into());
        }
        Ok(cat)
    }

    pub fn write_csv(&self, path: &Path) -> Result<(), String> {
        let mut s = String::from("row,col,elev_m,pop,imperv,role\n");
        for c in &self.cells {
            let role = if c.outlet { "outlet" } else if c.shelter { "shelter" } else { "" };
            s += &format!("{},{},{:.3},{:.1},{:.2},{}\n", c.row, c.col, c.elev_m, c.pop, c.imperv, role);
        }
        fs::write(path, s).map_err(|e| e.to_string())
    }

    /// Deterministic synthetic valley: high ground upstream (row 0), a
    /// populated, mostly-impervious floodplain draining to two outlets on
    /// the last row, shelters on high ground at the valley flanks.
    pub fn synthetic(nrows: usize, ncols: usize) -> Self {
        let mut cells = Vec::new();
        let mid = (ncols as f64 - 1.0) / 2.0;
        for r in 0..nrows {
            for c in 0..ncols {
                let down = 1.0 - r as f64 / (nrows as f64 - 1.0);
                let across = ((c as f64 - mid).abs() / mid).powf(1.5);
                let ripple = 0.08 * ((r * 7 + c * 13) as f64).sin();
                let elev = 2.6 * down + 1.4 * across + ripple;
                let in_plain = r >= nrows / 3 && (c as f64 - mid).abs() <= mid * 0.75;
                let pop = if in_plain { 180.0 * (-((c as f64 - mid) / 2.2).powi(2)).exp() } else { 8.0 };
                let imperv = if in_plain && r >= nrows / 2 { 0.8 } else { 0.3 };
                let outlet = r == nrows - 1 && (c as f64 - mid).abs() < 1.0;
                let shelter = (r == 1 && c == 1) || (r == 2 && c + 2 == ncols);
                cells.push(Cell { row: r, col: c, elev_m: elev, pop, imperv, outlet, shelter });
            }
        }
        Catchment { nrows, ncols, cell_m: 250.0, cells }
    }

    /// Hypergraph: each cell + its strictly-lower 4-neighbours is a
    /// *drainage* hyperedge (weight 1: a cell and everything it can spill
    /// into is one higher-order relation, not a set of pairwise links);
    /// every 2x2 block is a *reach* hyperedge (weight 0.5) so flat/pit
    /// cells stay connected.
    pub fn hypergraph(&self) -> Result<SpectralHypergraph, String> {
        let mut b = HypergraphBuilder::new();
        let ids: Vec<_> = (0..self.n())
            .map(|i| b.add_vertex(format!("cell_{}_{}", self.cells[i].row, self.cells[i].col)).map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        for i in 0..self.n() {
            let lower: Vec<_> = self
                .neighbours(i)
                .into_iter()
                .filter(|&j| self.cells[j].elev_m < self.cells[i].elev_m - 1e-9)
                .collect();
            if !lower.is_empty() {
                let mut m = vec![ids[i]];
                m.extend(lower.iter().map(|&j| ids[j]));
                b.add_hyperedge(&m, 1.0).map_err(|e| e.to_string())?;
            }
        }
        for r in 0..self.nrows.saturating_sub(1) {
            for c in 0..self.ncols.saturating_sub(1) {
                let m = [ids[self.idx(r, c)], ids[self.idx(r, c + 1)], ids[self.idx(r + 1, c)], ids[self.idx(r + 1, c + 1)]];
                b.add_hyperedge(&m, 0.5).map_err(|e| e.to_string())?;
            }
        }
        b.build().map_err(|e| e.to_string())
    }

    /// Convergence weight `w_i` on the rainfall source. Static topography
    /// enters as `s = Delta z_c` (z centred): cells higher than their
    /// hypergraph neighbourhood have `s > 0` and shed runoff (`w < 1`),
    /// depressions have `s < 0` and collect it (`w > 1`).
    /// `w = clamp(1 - kappa * s / max|s|, 0.3, 1.7)`.
    pub fn convergence_weights(&self, lap: &DMatrix<f64>, kappa: f64) -> Vec<f64> {
        let mean = self.cells.iter().map(|c| c.elev_m).sum::<f64>() / self.n() as f64;
        let z = DVector::from_iterator(self.n(), self.cells.iter().map(|c| c.elev_m - mean));
        let s = lap * z;
        let smax = s.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1e-12);
        s.iter().map(|&v| (1.0 - kappa * v / smax).clamp(0.3, 1.7)).collect()
    }

    /// Conservative downhill-routing operator `L` (n x n): each non-outlet cell with strictly-lower
    /// 4-neighbours sends its water to them in proportion to elevation drop (multiple-flow-direction).
    /// Column sums are 0 (mass conserving), off-diagonals <= 0, so `ds/dt = -k L s` stays Metzler
    /// (depths never go negative). Pits keep their water; outlets are sinks (their state is pinned to 0).
    pub fn downhill_operator(&self) -> DMatrix<f64> {
        let n = self.n();
        let mut l = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            if self.cells[i].outlet {
                continue;
            }
            let lower: Vec<(usize, f64)> = self
                .neighbours(i)
                .into_iter()
                .filter_map(|j| {
                    let d = self.cells[i].elev_m - self.cells[j].elev_m;
                    if d > 1e-9 { Some((j, d)) } else { None }
                })
                .collect();
            if lower.is_empty() {
                continue;
            }
            let tot: f64 = lower.iter().map(|x| x.1).sum();
            l[(i, i)] += 1.0;
            for (j, d) in lower {
                l[(j, i)] -= d / tot;
            }
        }
        l
    }

    /// `ln(flow accumulation)` scaled to [-1, 1]: contributing-cell count under the routing operator.
    /// A static hydrologic feature for the network (it is *not* derived from any flood label).
    pub fn log_flow_accumulation(&self, ldir: &DMatrix<f64>) -> Vec<f64> {
        let n = self.n();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| self.cells[b].elev_m.partial_cmp(&self.cells[a].elev_m).unwrap());
        let mut acc = vec![1.0f64; n];
        for &i in &order {
            for j in 0..n {
                let a = -ldir[(j, i)];
                if j != i && a > 0.0 {
                    acc[j] += acc[i] * a;
                }
            }
        }
        let lmax = acc.iter().cloned().fold(1.0, f64::max).ln().max(1e-9);
        acc.iter().map(|a| 2.0 * a.ln() / lmax - 1.0).collect()
    }

    pub fn laplacian(&self) -> Result<(SpectralHypergraph, DMatrix<f64>), String> {
        let hg = self.hypergraph()?;
        let lap = dense_normalized_laplacian(&hg).map_err(|e| e.to_string())?;
        Ok((hg, lap))
    }
}
