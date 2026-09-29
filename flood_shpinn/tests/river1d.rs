use flood_shpinn::catchment::Catchment;
use flood_shpinn::river::{Hydrograph, Reach};
use flood_shpinn::river1d::{series_peak, simulate, window_geometry, window_wse, Chain, ChannelParams, Lateral, Solver1D};

/// V-shaped valley draining to the last row (same fixture as tests/river.rs).
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

fn setup(p: &ChannelParams) -> (Reach, flood_shpinn::river1d::WindowGeom, Chain) {
    let r = Reach::extract(&valley(), 0.625).unwrap();
    let g = window_geometry(&r, p);
    let chain = Chain::build(&[&g], &[0.0], p.seg_len_m);
    (r, g, chain)
}

#[test]
fn tables_are_monotone_and_area_inverts() {
    let (_, g, _) = setup(&ChannelParams::default());
    for t in &g.tables {
        assert!(t.a.windows(2).all(|w| w[1] > w[0]));
        assert!(t.b.windows(2).all(|w| w[1] >= w[0]));
        assert!(t.k.windows(2).all(|w| w[1] >= w[0]));
        for h in [0.3, 1.7, 6.2, 14.0] {
            let back = t.stage_for_area(t.area(h));
            assert!((back - h).abs() < 1e-6, "h {h} -> {back}");
        }
    }
    assert!(g.bed.windows(2).all(|w| w[1] <= w[0] + 1e-12));
}

#[test]
fn steady_inflow_passes_through_unchanged() {
    let (_, _, chain) = setup(&ChannelParams::default());
    let inflow = Hydrograph { t_h: vec![0.0, 20.0], q: vec![150.0, 150.0] };
    let s = simulate(&chain, &inflow, &[], 150.0, 6.0, 0.25, &Solver1D::default());
    let last = s.q.last().unwrap();
    for (i, q) in last.iter().enumerate() {
        assert!((q - 150.0).abs() / 150.0 < 0.03, "node {i}: {q}");
    }
    assert!(s.mass_error().abs() < 1e-3);
}

#[test]
fn flood_wave_is_conservative_positive_lagged_and_attenuated() {
    let (_, _, chain) = setup(&ChannelParams::default());
    let inflow = Hydrograph::nash(5.0, 1500.0, 1.0, 1.0, 6.0, 30.0, 0.05);
    let s = simulate(&chain, &inflow, &[], 5.0, 30.0, 0.05, &Solver1D::default());
    assert!(s.mass_error().abs() < 2e-3, "closure {}", s.mass_error());
    assert!(s.eta.iter().flatten().all(|x| x.is_finite()));
    assert!(s.q.iter().flatten().all(|x| x.is_finite() && *x > -50.0));
    let n = chain.n();
    let up: Vec<f64> = s.q.iter().map(|r| r[1]).collect();
    let dn: Vec<f64> = s.q.iter().map(|r| r[n - 2]).collect();
    let ((qu, tu), (qd, td)) = (series_peak(&s.t_h, &up), series_peak(&s.t_h, &dn));
    assert!(qd < qu, "no attenuation: {qu} -> {qd}");
    assert!(td > tu, "peak arrived early: {tu} -> {td}");
    assert!(qd > 0.3 * qu, "over-attenuated: {qu} -> {qd}");
}

#[test]
fn lateral_inflow_adds_its_volume() {
    let (_, _, chain) = setup(&ChannelParams::default());
    let inflow = Hydrograph::nash(5.0, 500.0, 1.0, 1.0, 6.0, 30.0, 0.05);
    let lat = Hydrograph::nash(0.0, 300.0, 2.0, 1.0, 6.0, 30.0, 0.05);
    let l = Lateral { n0: 2, cnt: chain.n() - 4, hydro: lat.clone() };
    let a = simulate(&chain, &inflow, &[], 5.0, 30.0, 0.05, &Solver1D::default());
    let b = simulate(&chain, &inflow, &[l], 5.0, 30.0, 0.05, &Solver1D::default());
    assert!(b.mass_error().abs() < 3e-3, "closure {}", b.mass_error());
    let dv = b.vol_out_m3 - a.vol_out_m3 + (b.storage1_m3 - a.storage1_m3);
    assert!((dv - lat.volume_m3()).abs() / lat.volume_m3() < 0.05, "dv {dv} vs {}", lat.volume_m3());
}

#[test]
fn water_surface_follows_the_channel_downhill() {
    let p = ChannelParams::default();
    let (r, g, chain) = setup(&p);
    let inflow = Hydrograph::nash(5.0, 1500.0, 1.0, 1.0, 6.0, 30.0, 0.05);
    let s = simulate(&chain, &inflow, &[], 5.0, 30.0, 0.25, &Solver1D::default());
    // peak surface is (weakly) non-increasing downstream, so it is *not* a uniform stage
    let eta = &s.eta_max;
    assert!(eta.windows(2).filter(|w| w[1] > w[0] + 0.5).count() == 0, "adverse peak surface");
    assert!(eta[0] - eta[eta.len() - 1] > 3.0);
    let wse = window_wse(&chain, 0, &g, eta);
    assert_eq!(wse.len(), r.elev.len());
    assert!(wse.iter().all(|x| x.is_finite()));
}

#[test]
fn floodplain_roughness_changes_conveyance_but_not_channel_part() {
    let a = window_geometry(&Reach::extract(&valley(), 0.625).unwrap(), &ChannelParams { n_fp: 0.05, ..ChannelParams::default() });
    let b = window_geometry(&Reach::extract(&valley(), 0.625).unwrap(), &ChannelParams { n_fp: 0.20, ..ChannelParams::default() });
    let s = a.tables.len() / 2;
    // low stage (inside h_bf): identical; high stage: rougher floodplain carries less
    assert!((a.tables[s].conv(1.0) - b.tables[s].conv(1.0)).abs() / a.tables[s].conv(1.0) < 1e-9);
    assert!(b.tables[s].conv(15.0) < a.tables[s].conv(15.0));
    // storage geometry is independent of roughness
    assert!((a.tables[s].area(15.0) - b.tables[s].area(15.0)).abs() < 1e-9);
}
