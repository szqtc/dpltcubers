//! PyO3 bindings for [`crate::ltcube_main`].
//!
//! Build with:
//!
//! ```text
//! cargo build --release --no-default-features --features python
//! ```
//!
//! This produces a single file, `target/release/libdpltcubers.so`, which is a
//! directly importable Python module (cargo always prefixes `lib` on Unix):
//!
//! ```python
//! import libdpltcubers
//! libdpltcubers.ltcube_phi("sc.fits", "ev.fits", "out.fits")
//! ```

use std::path::Path;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::ltcube_main;

/// Upper bound imposed by the HEALPix bookkeeping in `ltcube_main`.
const ORDER_MAX: u32 = 29;

/// main function of ltcube
///
/// parameters
/// ----------
/// scfile: str
///     the file name of sc data
/// evfile: str
///     the file name of events
/// outfile: str
///     the name of output ltcube
/// order: non-negative integer, default=6
///     the order of HealPix map
/// thetabins: positive integer, default=20
///     the number of theta bins
/// phibins: positive integer, default=1
///     the number of phi bins
/// zmax: double, [0., 180.), default=100.
///     the zenith angle cut in [deg]
/// nthreads: positive integer, default=1
///     the number of threads, >1 for multi-threads
///     calculations
/// linctheta: bool, default=False
///     whether the theta is linearly binned or not
#[pyfunction]
#[pyo3(signature = (scfile, evfile, outfile, order=6, thetabins=20, phibins=1,
                    zmax=100.0, nthreads=1, linctheta=false))]
fn ltcube_phi(
    py: Python<'_>,
    scfile: &str,
    evfile: &str,
    outfile: &str,
    order: u32,
    thetabins: usize,
    phibins: usize,
    zmax: f64,
    nthreads: usize,
    linctheta: bool,
) -> PyResult<()> {
    if order > ORDER_MAX {
        return Err(PyValueError::new_err(format!(
            "order must be between 0 and {}, got {}",
            ORDER_MAX, order
        )));
    }
    if thetabins == 0 {
        return Err(PyValueError::new_err(
            "thetabins must be a positive integer",
        ));
    }
    if phibins == 0 {
        return Err(PyValueError::new_err(
            "phibins must be a positive integer",
        ));
    }
    if !(0.0..180.0).contains(&zmax) {
        return Err(PyValueError::new_err(format!(
            "zmax must be in [0., 180.), got {}",
            zmax
        )));
    }
    if nthreads == 0 {
        return Err(PyValueError::new_err(
            "nthreads must be a positive integer",
        ));
    }
    for (name, path) in [("scfile", scfile), ("evfile", evfile)] {
        if !Path::new(path).is_file() {
            return Err(PyRuntimeError::new_err(format!(
                "{} not found: {}",
                name, path
            )));
        }
    }

    // Own the paths: the GIL is released below, so no Python borrow may live on.
    let scfile = scfile.to_owned();
    let evfile = evfile.to_owned();
    let outfile = outfile.to_owned();

    py.detach(move || {
        ltcube_main(
            &scfile, &evfile, &outfile, order, thetabins, phibins, 0., zmax, nthreads, linctheta,
        )
        .map_err(|e| e.to_string())
    })
    .map_err(PyRuntimeError::new_err)
}

/// Python module implemented by `libdpltcubers.so`.
#[pymodule]
fn libdpltcubers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ltcube_phi, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
