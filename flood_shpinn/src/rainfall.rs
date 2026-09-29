//! Rainfall series: CSV `hour,mm_per_h` (header required), piecewise-linear.

use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Rainfall {
    pub hours: Vec<f64>,
    pub mm_per_h: Vec<f64>,
}

impl Rainfall {
    pub fn load_csv(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (mut h, mut m) = (Vec::new(), Vec::new());
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
                return Err(format!("rain csv line {}: need hour,mm_per_h", ln + 1));
            }
            h.push(f[0].parse::<f64>().map_err(|e| format!("line {}: {e}", ln + 1))?);
            m.push(f[1].parse::<f64>().map_err(|e| format!("line {}: {e}", ln + 1))?.max(0.0));
        }
        if h.len() < 2 || h.windows(2).any(|w| w[1] <= w[0]) {
            return Err("rain csv: need >=2 strictly increasing hours".into());
        }
        Ok(Rainfall { hours: h, mm_per_h: m })
    }

    pub fn write_csv(&self, path: &Path) -> Result<(), String> {
        let mut s = String::from("hour,mm_per_h\n");
        for (h, m) in self.hours.iter().zip(&self.mm_per_h) {
            s += &format!("{h:.2},{m:.2}\n");
        }
        fs::write(path, s).map_err(|e| e.to_string())
    }

    /// Triangular design storm sampled every `step_h`.
    pub fn design_storm(peak_mm_h: f64, peak_h: f64, start_h: f64, end_h: f64, horizon_h: f64, step_h: f64) -> Self {
        let n = (horizon_h / step_h).round() as usize;
        let (mut hours, mut mm) = (Vec::new(), Vec::new());
        for k in 0..=n {
            let t = k as f64 * step_h;
            let v = if t <= start_h || t >= end_h {
                0.0
            } else if t <= peak_h {
                peak_mm_h * (t - start_h) / (peak_h - start_h)
            } else {
                peak_mm_h * (end_h - t) / (end_h - peak_h)
            };
            hours.push(t);
            mm.push(v);
        }
        Rainfall { hours, mm_per_h: mm }
    }

    pub fn at(&self, t: f64) -> f64 {
        if t <= self.hours[0] || t >= *self.hours.last().unwrap() {
            return if t == self.hours[0] { self.mm_per_h[0] } else { 0.0 };
        }
        let k = self.hours.partition_point(|&h| h <= t) - 1;
        let f = (t - self.hours[k]) / (self.hours[k + 1] - self.hours[k]);
        self.mm_per_h[k] * (1.0 - f) + self.mm_per_h[k + 1] * f
    }

    pub fn total_mm(&self) -> f64 {
        self.hours.windows(2).zip(self.mm_per_h.windows(2)).map(|(h, m)| 0.5 * (m[0] + m[1]) * (h[1] - h[0])).sum()
    }
}
