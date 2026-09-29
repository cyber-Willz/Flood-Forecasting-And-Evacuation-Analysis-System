use flood_shpinn::calib::{auc, matched_prevalence_csi};
use flood_shpinn::catchment::Catchment;
use flood_shpinn::rainfall::Rainfall;
use flood_shpinn::truth::{simulate, Params};

#[test]
fn auc_perfect_reversed_and_ties() {
    let l = [true, true, false, false];
    assert!((auc(&[4.0, 3.0, 2.0, 1.0], &l) - 1.0).abs() < 1e-12);
    assert!(auc(&[1.0, 2.0, 3.0, 4.0], &l).abs() < 1e-12);
    assert!((auc(&[1.0, 1.0, 1.0, 1.0], &l) - 0.5).abs() < 1e-12);
    assert!((matched_prevalence_csi(&[4.0, 3.0, 2.0, 1.0], &l) - 1.0).abs() < 1e-12);
}

#[test]
fn downhill_operator_conserves_mass_and_is_metzler() {
    let cat = Catchment::synthetic(8, 10);
    let l = cat.downhill_operator();
    for j in 0..cat.n() {
        let col: f64 = (0..cat.n()).map(|i| l[(i, j)]).sum();
        assert!(col.abs() < 1e-12, "column {j} sums to {col}");
        for i in 0..cat.n() {
            if i != j {
                assert!(l[(i, j)] <= 0.0);
            }
        }
    }
}

#[test]
fn routing_off_reproduces_v02_and_routing_on_stays_nonnegative() {
    let cat = Catchment::synthetic(8, 10);
    let (_, lap) = cat.laplacian().unwrap();
    let w = cat.convergence_weights(&lap, 0.6);
    let ldir = cat.downhill_operator();
    let rain = Rainfall::design_storm(130.0, 2.5, 0.5, 5.0, 6.0, 0.25);
    let p0 = Params::default();
    let zero = nalgebra::DMatrix::<f64>::zeros(cat.n(), cat.n());
    let a = simulate(&cat, &lap, &ldir, &w, &rain, &p0, 6.0, 0.25); // k_dir = 0
    let b = simulate(&cat, &lap, &zero, &w, &rain, &p0, 6.0, 0.25);
    for (x, y) in a.iter().zip(&b) {
        for (u, v) in x.1.iter().zip(&y.1) {
            assert!((u - v).abs() < 1e-12);
        }
    }
    let p1 = Params { k_dir: 2.0, ..p0 };
    for (_, d) in simulate(&cat, &lap, &ldir, &w, &rain, &p1, 6.0, 0.25) {
        assert!(d.iter().all(|&v| v >= -1e-9));
    }
}
