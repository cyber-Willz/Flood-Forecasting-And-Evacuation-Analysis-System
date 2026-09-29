//! SHPINN training for the runoff PDE in [`crate::truth`].
//!
//! Two trainers share one [`FloodNet`] type:
//!
//! * [`train_legacy`] — the original v0.1 algorithm, kept verbatim for A/B comparison
//!   (soft IC/BC penalties, fixed 0.5 h collocation grid, eigenmap-only input, constant LR).
//! * [`train`] — v0.2 (default):
//!   1. **Hard constraints**: `s(x,tau) = tau * g(x,tau) * interior(x)` satisfies `s(.,0)=0` and
//!      the outlet Dirichlet condition exactly, so the whole loss is the PDE residual.
//!   2. **Static physical features**: `[eigenmap(k), z_norm, imperv, w_norm, tau]`. Eigenmap
//!      coordinates alone are smooth low-frequency functions and cannot resolve cell-level
//!      structure on grids with hundreds of cells.
//!   3. **Stratified random collocation times** each epoch (fewer, but different, points per
//!      epoch -> cheaper epochs and no fixed-grid overfitting).
//!   4. **Semi-analytic time derivative**: `ds/dtau = g + tau*dg/dtau`; only `g` is differenced.
//!   5. **Cosine learning-rate decay**.

use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use shpinn::embedding::laplacian_eigenmap;
use shpinn::loss::{boundary_mask, shpinn_loss};
use shpinn::operator::dense_laplacian_tensor;
use shpinn::physics::{central_time_derivative, heat_residual};
use shpinn::pinn::{Shpinn, ShpinnConfig};
use spectral_hypergraph::SpectralHypergraph;

use nalgebra::DMatrix;

use crate::catchment::Catchment;
use crate::rainfall::Rainfall;
use crate::truth::{source, Params};

pub type Be = Autodiff<NdArray<f32>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Legacy,
    V2,
}

pub struct TrainCfg {
    pub k_embed: usize,
    pub hidden: Vec<usize>,
    pub epochs: usize,
    pub lr: f64,
    pub lr_min: f64,
    pub data_weight: f64,
    pub physics_weight: f64,
    pub train_step_h: f64,
    pub t_end: f64,
    /// v2: collocation times per epoch.
    pub n_colloc: usize,
    pub seed: u64,
}

impl Default for TrainCfg {
    fn default() -> Self {
        TrainCfg { k_embed: 6, hidden: vec![48, 48], epochs: 4000, lr: 0.004, lr_min: 0.004, data_weight: 20.0, physics_weight: 1.0, train_step_h: 0.5, t_end: 6.0, n_colloc: 4, seed: 0 }
    }
}

impl TrainCfg {
    /// Recommended v0.2 settings.
    pub fn v2(t_end: f64, epochs: usize, seed: u64) -> Self {
        TrainCfg { k_embed: 16, hidden: vec![64, 64], epochs, lr: 0.004, lr_min: 0.0001, t_end, n_colloc: 4, seed, ..Default::default() }
    }
}

pub struct FloodNet {
    pub net: Shpinn<Be>,
    feats: Tensor<Be, 2>,
    interior: Tensor<Be, 2>,
    device: <Be as Backend>::Device,
    n: usize,
    t_end: f64,
    depth_scale: f64,
    model: Model,
}

struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn with_tau(feats: &Tensor<Be, 2>, n: usize, tau: f64, dev: &<Be as Backend>::Device) -> Tensor<Be, 2> {
    let col = Tensor::<Be, 1>::from_floats(vec![tau as f32; n].as_slice(), dev).reshape([n, 1]);
    Tensor::cat(vec![feats.clone(), col], 1)
}

fn col_tensor(v: &[f64], dev: &<Be as Backend>::Device) -> Tensor<Be, 2> {
    let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::<Be, 1>::from_floats(f.as_slice(), dev).reshape([v.len(), 1])
}

fn mat_tensor(cols: &[Vec<f64>], dev: &<Be as Backend>::Device) -> Tensor<Be, 2> {
    let n = cols[0].len();
    let k = cols.len();
    let mut flat = Vec::with_capacity(n * k);
    for i in 0..n {
        for c in cols {
            flat.push(c[i] as f32);
        }
    }
    Tensor::<Be, 1>::from_floats(flat.as_slice(), dev).reshape([n, k])
}

fn dense_tensor(m: &DMatrix<f64>, dev: &<Be as Backend>::Device) -> Tensor<Be, 2> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut flat = Vec::with_capacity(r * c);
    for i in 0..r {
        for j in 0..c {
            flat.push(m[(i, j)] as f32);
        }
    }
    Tensor::<Be, 1>::from_floats(flat.as_slice(), dev).reshape([r, c])
}

/// Original v0.1 trainer (unchanged algorithm; kept for A/B). Only valid for `p.k_dir == 0`.
pub fn train_legacy(cat: &Catchment, hg: &SpectralHypergraph, w: &[f64], rain: &Rainfall, p: &Params, cfg: &TrainCfg, log_every: usize) -> Result<FloodNet, String> {
    if p.k_dir != 0.0 {
        return Err("legacy trainer predates directed routing; use --model v2 or set --k-dir 0".into());
    }
    <Be as Backend>::seed(cfg.seed);
    let dev = <Be as Backend>::Device::default();
    let n = cat.n();
    let coords = laplacian_eigenmap(hg, cfg.k_embed).map_err(|e| e.to_string())?.to_tensor::<Be>(&dev);
    let lap = dense_laplacian_tensor::<Be>(hg, &dev).map_err(|e| e.to_string())?;

    let outlets = cat.outlets();
    let all: Vec<usize> = (0..n).collect();
    let ic_mask = boundary_mask::<Be>(n, &all, &dev).map_err(|e| e.to_string())?;
    let out_mask = boundary_mask::<Be>(n, &outlets, &dev).map_err(|e| e.to_string())?;
    let interior = Tensor::<Be, 2>::ones([n, 1], &dev) - out_mask.clone();
    let zero = Tensor::<Be, 2>::zeros([n, 1], &dev);

    let nt = (cfg.t_end / cfg.train_step_h).round() as usize;
    let times: Vec<f64> = (0..=nt).map(|k| k as f64 * cfg.train_step_h).collect();
    let src: Vec<Tensor<Be, 2>> = times.iter().map(|&t| col_tensor(source(cat, w, rain, t, p).as_slice(), &dev)).collect();

    let mut net = ShpinnConfig::new(cfg.k_embed + 1, cfg.hidden.clone(), 1).init::<Be>(&dev);
    let mut opt = AdamConfig::new().init();
    let eps = 1e-3;

    for epoch in 0..cfg.epochs {
        let mut loss = Tensor::<Be, 1>::zeros([1], &dev);
        for (j, &t) in times.iter().enumerate() {
            let tau = t / cfg.t_end;
            let u = net.forward(with_tau(&coords, n, tau, &dev));
            let up = net.forward(with_tau(&coords, n, tau + eps, &dev));
            let um = net.forward(with_tau(&coords, n, tau - eps, &dev));
            let u_t = central_time_derivative::<Be>(up, um, eps) / cfg.t_end;
            let resid = (heat_residual::<Be>(&lap, u.clone(), u_t, p.d) + u.clone() * p.lam - src[j].clone()) * interior.clone();
            let mask = if j == 0 { &ic_mask } else { &out_mask };
            loss = loss + shpinn_loss::<Be>(u, &zero, mask, resid, cfg.data_weight, cfg.physics_weight);
        }
        if log_every > 0 && (epoch % log_every == 0 || epoch + 1 == cfg.epochs) {
            let v: f32 = loss.clone().into_scalar();
            println!("  epoch {epoch:5}: loss = {v:.6}");
        }
        let g = GradientsParams::from_grads(loss.backward(), &net);
        net = opt.step(cfg.lr, net, g);
    }
    Ok(FloodNet { net, feats: coords, interior, device: dev, n, t_end: cfg.t_end, depth_scale: p.depth_scale, model: Model::Legacy })
}

/// v0.2 trainer (hard constraints + physical features + random collocation + cosine LR).
pub fn train(cat: &Catchment, hg: &SpectralHypergraph, ldir: &DMatrix<f64>, w: &[f64], rain: &Rainfall, p: &Params, cfg: &TrainCfg, log_every: usize) -> Result<FloodNet, String> {
    <Be as Backend>::seed(cfg.seed);
    let dev = <Be as Backend>::Device::default();
    let n = cat.n();
    let emb = laplacian_eigenmap(hg, cfg.k_embed).map_err(|e| e.to_string())?.to_tensor::<Be>(&dev);
    let lap = dense_laplacian_tensor::<Be>(hg, &dev).map_err(|e| e.to_string())?;

    // Static per-cell features (all derivable from the GIS table + hypergraph).
    let zs: Vec<f64> = cat.cells.iter().map(|c| c.elev_m).collect();
    let (zmin, zmax) = (zs.iter().cloned().fold(f64::MAX, f64::min), zs.iter().cloned().fold(f64::MIN, f64::max));
    let z_n: Vec<f64> = zs.iter().map(|z| 2.0 * (z - zmin) / (zmax - zmin).max(1e-9) - 1.0).collect();
    let imp: Vec<f64> = cat.cells.iter().map(|c| c.imperv).collect();
    let w_n: Vec<f64> = w.iter().map(|x| (x - 1.0) / 0.7).collect();
    // v0.3: + log flow accumulation (a hydrologic feature from the DEM alone; only when routing is on,
    // so k_dir = 0 keeps the exact v0.2 input layout).
    let routed = p.k_dir != 0.0;
    let mut cols = vec![z_n, imp, w_n];
    if routed {
        cols.push(cat.log_flow_accumulation(ldir));
    }
    let stat = mat_tensor(&cols, &dev);
    let feats = Tensor::cat(vec![emb, stat], 1);
    let fdim = cfg.k_embed + cols.len();
    let ldir_t = dense_tensor(ldir, &dev);

    let outlets = cat.outlets();
    let out_mask = boundary_mask::<Be>(n, &outlets, &dev).map_err(|e| e.to_string())?;
    let interior = Tensor::<Be, 2>::ones([n, 1], &dev) - out_mask;

    let mut net = ShpinnConfig::new(fdim + 1, cfg.hidden.clone(), 1).init::<Be>(&dev);
    let mut opt = AdamConfig::new().init();
    let mut rng = Rng(cfg.seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
    let eps = 2e-3;
    let k = cfg.n_colloc.max(1);

    for epoch in 0..cfg.epochs {
        let cosf = 0.5 * (1.0 + (std::f64::consts::PI * epoch as f64 / cfg.epochs as f64).cos());
        let lr = cfg.lr_min + (cfg.lr - cfg.lr_min) * cosf;
        let mut loss = Tensor::<Be, 1>::zeros([1], &dev);
        for j in 0..k {
            let tau = ((j as f64 + rng.next_f64()) / k as f64).clamp(1e-3, 1.0);
            let src = col_tensor(source(cat, w, rain, tau * cfg.t_end, p).as_slice(), &dev);
            let g = net.forward(with_tau(&feats, n, tau, &dev));
            let gp = net.forward(with_tau(&feats, n, tau + eps, &dev));
            let gm = net.forward(with_tau(&feats, n, tau - eps, &dev));
            let g_t = central_time_derivative::<Be>(gp, gm, eps);
            let u = g.clone() * interior.clone() * tau;
            let u_t = (g + g_t * tau) * interior.clone() / cfg.t_end;
            let route = if routed { ldir_t.clone().matmul(u.clone()) * p.k_dir } else { u.clone() * 0.0 };
            let resid = (heat_residual::<Be>(&lap, u.clone(), u_t, p.d) + u * p.lam + route - src) * interior.clone();
            loss = loss + resid.powf_scalar(2.0).mean();
        }
        let loss = loss / (k as f64);
        if log_every > 0 && (epoch % log_every == 0 || epoch + 1 == cfg.epochs) {
            let v: f32 = loss.clone().into_scalar();
            println!("  epoch {epoch:5}: pde-residual loss = {v:.6}  lr = {lr:.5}");
        }
        let gr = GradientsParams::from_grads(loss.backward(), &net);
        net = opt.step(lr, net, gr);
    }
    Ok(FloodNet { net, feats, interior, device: dev, n, t_end: cfg.t_end, depth_scale: p.depth_scale, model: Model::V2 })
}

impl FloodNet {
    pub fn model(&self) -> Model {
        self.model
    }
    /// Predicted depth in metres per cell at `t_hours` (clamped at 0: physics forbids <0).
    pub fn predict(&self, t_hours: f64) -> Vec<f64> {
        let tau = t_hours / self.t_end;
        let g = self.net.forward(with_tau(&self.feats, self.n, tau, &self.device));
        let u = match self.model {
            Model::Legacy => g,
            Model::V2 => g * self.interior.clone() * tau,
        };
        let d = u.into_data().convert::<f32>();
        d.value.iter().map(|&v| (v as f64 * self.depth_scale).max(0.0)).collect()
    }
}
