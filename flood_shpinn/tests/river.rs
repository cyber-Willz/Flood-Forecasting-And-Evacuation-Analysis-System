use flood_shpinn::catchment::Catchment;
use flood_shpinn::river::{route_mc, Hydrograph, Reach};

/// V-shaped valley draining to the last row; deterministic, no noise.
fn valley() -> Catchment {
    let mut cat = Catchment::synthetic(40, 40);
    cat.cell_m = 100.0;
    for c in cat.cells.iter_mut() {
        let across = (c.col as f64 - 19.5).abs();
        c.elev_m = 500.0 - 0.003 * 100.0 * c.row as f64 + 0.05 * across * across.min(12.0) + across * 0.4;
        c.outlet = false;
        c.shelter = false;
    }
    let i = cat.idx(39, 20);
    cat.cells[i].outlet = true;
    cat
}

#[test]
fn thalweg_follows_valley_and_bed_is_monotone() {
    let r = Reach::extract(&valley(), 0.625).unwrap();
    assert!(r.path.len() >= 38);
    let beds: Vec<f64> = r.path.iter().map(|&i| r.bed[i]).collect();
    assert!(beds.windows(2).all(|w| w[1] <= w[0] + 1e-12));
    assert!(r.slope > 1e-3 && r.slope < 1e-2, "slope {}", r.slope);
}

#[test]
fn rating_is_monotone_and_stage_roundtrips() {
    let r = Reach::extract(&valley(), 0.625).unwrap();
    let rt = r.rating(0.05, 20.0, 0.1);
    assert!(rt.q.windows(2).all(|w| w[1] >= w[0]));
    for q in [10.0, 200.0, 1500.0] {
        let h = rt.stage(q);
        assert!((rt.discharge(h) - q).abs() / q < 0.02, "q {q} -> h {h} -> {}", rt.discharge(h));
    }
}

#[test]
fn inundation_is_connected_and_grows_with_stage() {
    let r = Reach::extract(&valley(), 0.625).unwrap();
    let wet = |h: f64| r.inundation(h).0.iter().filter(|&&d| d > 0.0).count();
    assert!(wet(1.0) <= wet(3.0) && wet(3.0) <= wet(8.0));
    let (d, _) = r.inundation(3.0);
    assert!(d.iter().all(|&x| x >= 0.0));
    assert!((0..d.len()).filter(|&i| r.thal[i]).all(|i| d[i] > 0.0));
}

#[test]
fn routing_conserves_volume_attenuates_and_lags() {
    let r = Reach::extract(&valley(), 0.625).unwrap();
    let rt = r.rating(0.05, 20.0, 0.1);
    let inflow = Hydrograph::nash(5.0, 1500.0, 1.0, 1.0, 6.0, 30.0, 0.05);
    let out = route_mc(&inflow, None, &rt, 20_000.0, r.slope, 0.05, 30.0, 1.0, false);
    let (vi, vo) = (inflow.volume_m3(), out.volume_m3());
    assert!((vi - vo).abs() / vi < 0.02, "volume {vi} vs {vo}");
    let ((qi, ti), (qo, to)) = (inflow.peak(), out.peak());
    assert!(qo <= qi + 1e-6, "peak grew {qi}->{qo}");
    assert!(to >= ti, "peak arrived early");
    assert!(out.q.iter().all(|&q| q >= 0.0));
}

#[test]
fn lateral_inflow_adds_its_volume() {
    let r = Reach::extract(&valley(), 0.625).unwrap();
    let rt = r.rating(0.05, 20.0, 0.1);
    let inflow = Hydrograph::nash(5.0, 500.0, 1.0, 1.0, 6.0, 30.0, 0.05);
    let lat = Hydrograph::nash(0.0, 300.0, 2.0, 1.0, 6.0, 30.0, 0.05);
    let a = route_mc(&inflow, None, &rt, 10_000.0, r.slope, 0.05, 30.0, 1.0, false);
    let b = route_mc(&inflow, Some(&lat), &rt, 10_000.0, r.slope, 0.05, 30.0, 1.0, false);
    let dv = b.volume_m3() - a.volume_m3();
    assert!((dv - lat.volume_m3()).abs() / lat.volume_m3() < 0.03, "dv {dv} vs {}", lat.volume_m3());
}
