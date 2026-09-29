//! Shared forecast-skill and evacuation-skill metrics (used for both trainers so A/B is apples-to-apples).

use crate::evac::CellPlan;

#[derive(Clone, Debug, Default)]
pub struct DepthSkill {
    pub rmse_m: f64,
    pub nse: f64,
    pub rel_rmse: f64,
    pub peak_rmse_m: f64,
    pub peak_bias_m: f64,
    pub csi: f64,
    pub pod: f64,
    pub far: f64,
    pub extent_acc: f64,
    pub wet_truth: usize,
    pub wet_pred: usize,
}

/// `pred[k][i]`, `truth[k][i]` over the same times `k` (held-out set) and cells `i`.
pub fn depth_skill(pred: &[Vec<f64>], truth: &[Vec<f64>], peak_pred: &[f64], peak_truth: &[f64], thr_m: f64) -> DepthSkill {
    let (mut se, mut n) = (0.0, 0usize);
    let mean_t = truth.iter().flatten().sum::<f64>() / truth.iter().map(|v| v.len()).sum::<usize>() as f64;
    let mut var = 0.0;
    let max_t = truth.iter().flatten().cloned().fold(0.0, f64::max);
    for (a, b) in pred.iter().zip(truth) {
        for (x, y) in a.iter().zip(b) {
            se += (x - y).powi(2);
            var += (y - mean_t).powi(2);
            n += 1;
        }
    }
    let rmse = (se / n as f64).sqrt();
    let np = peak_pred.len();
    let (mut pse, mut pb, mut tp, mut fp, mut fnn, mut tn) = (0.0, 0.0, 0usize, 0usize, 0usize, 0usize);
    for i in 0..np {
        pse += (peak_pred[i] - peak_truth[i]).powi(2);
        pb += peak_pred[i] - peak_truth[i];
        match (peak_pred[i] > thr_m, peak_truth[i] > thr_m) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fnn += 1,
            (false, false) => tn += 1,
        }
    }
    let d = |a: usize, b: usize| if b == 0 { f64::NAN } else { a as f64 / b as f64 };
    DepthSkill {
        rmse_m: rmse,
        nse: 1.0 - se / var.max(1e-30),
        rel_rmse: rmse / max_t.max(1e-12),
        peak_rmse_m: (pse / np as f64).sqrt(),
        peak_bias_m: pb / np as f64,
        csi: d(tp, tp + fp + fnn),
        pod: d(tp, tp + fnn),
        far: d(fp, tp + fp),
        extent_acc: (tp + tn) as f64 / np as f64,
        wet_truth: tp + fnn,
        wet_pred: tp + fp,
    }
}

#[derive(Clone, Debug, Default)]
pub struct EvacSkill {
    pub pop_total: f64,
    pub pop_cut_truth: f64,
    pub pop_cut_pred: f64,
    /// Population whose predicted window-close time is within `tol_h` of truth (both never-close counts as agree).
    pub pop_agree_frac: f64,
    /// Population-weighted mean |close-time error| (min) where both close.
    pub mae_min: f64,
    /// Population for which truth closes but the forecast says it stays open or closes >tol later (unsafe error).
    pub pop_false_safe: f64,
    /// Population for which the forecast closes but truth stays open or closes >tol earlier (over-cautious error).
    pub pop_false_alarm: f64,
    pub cells_agree: usize,
    pub cells: usize,
}

pub fn evac_skill(pred: &[CellPlan], truth: &[CellPlan], tol_h: f64) -> EvacSkill {
    let mut s = EvacSkill::default();
    let (mut agree_pop, mut abs_w, mut w_sum) = (0.0, 0.0, 0.0);
    for (a, b) in pred.iter().zip(truth) {
        s.pop_total += b.pop;
        s.cells += 1;
        if b.window_closes_h.is_some() { s.pop_cut_truth += b.pop; }
        if a.window_closes_h.is_some() { s.pop_cut_pred += a.pop; }
        let ok = match (a.window_closes_h, b.window_closes_h) {
            (None, None) => true,
            (Some(x), Some(y)) => {
                abs_w += (x - y).abs() * 60.0 * b.pop;
                w_sum += b.pop;
                if x > y + tol_h + 1e-9 { s.pop_false_safe += b.pop; }
                if x < y - tol_h - 1e-9 { s.pop_false_alarm += b.pop; }
                (x - y).abs() <= tol_h + 1e-9
            }
            (None, Some(_)) => { s.pop_false_safe += b.pop; false }
            (Some(_), None) => { s.pop_false_alarm += b.pop; false }
        };
        if ok { agree_pop += b.pop; s.cells_agree += 1; }
    }
    s.pop_agree_frac = agree_pop / s.pop_total.max(1e-12);
    s.mae_min = if w_sum > 0.0 { abs_w / w_sum } else { f64::NAN };
    s
}
