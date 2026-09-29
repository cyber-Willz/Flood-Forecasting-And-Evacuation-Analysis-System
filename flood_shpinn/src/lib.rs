//! # flood_shpinn
//!
//! Flood-depth forecasting and evacuation-window analysis built on `shpinn`.
//!
//! Pipeline: GIS cell table + rainfall series -> spectral hypergraph
//! ([`catchment`]) -> reduced-order runoff PDE enforced as a SHPINN residual
//! ([`floodnet`]) -> independent RK4 ground truth ([`truth`]) -> time-dependent
//! evacuation routing over the forecast depth fields ([`evac`]).
//!
//! v0.5 adds [`river1d`]: a 1D compound-section local-inertial router (dynamic floodplain storage, slope-following
//! water surface) that replaces Muskingum-Cunge as the default fluvial router.
//!
//! v0.4 adds [`river`]: a fluvial layer (thalweg extraction, HAND synthetic rating curve, Muskingum-Cunge
//! routing of an upstream inflow hydrograph, connected inundation) that the rain-only ponding model lacked.

pub mod catchment;
pub mod evac;
#[cfg(feature = "nn")]
pub mod floodnet;
pub mod rainfall;
pub mod truth;
pub mod metrics;
pub mod calib;
pub mod river;
pub mod river1d;
