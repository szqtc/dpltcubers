use std::str::FromStr;
use dpltcubers::*;

////////////////////////
// parse arguments
////////////////////////
fn parse_run() {
    // define arg
    let args: Vec<String> = std::env::args().collect();
    let prog = args[0].clone();
    let brief = format!("Usage: {} -s SCFILE -e EVFILE -o OUTFILE [options]\n\na simple tool to calculate the livetime cube of DAMPE (ver. {})",
            prog, env!("CARGO_PKG_VERSION"));

    let mut opts = getopts::Options::new();
    opts.reqopt("s", "scfile", "the name of SC file", "SCFILE");
    opts.reqopt("e", "evfile", "the name of EVENT file", "EVFILE");
    opts.reqopt("o", "outfile", "the name of output LTCUBE file", "OUTFILE");

    opts.optopt("r", "order", "order for healpix bin, nside=2**order", "ORDER [6]");
    // opts.optopt("", "ordering", "ordering scheme of healpix pixels, default = NESTED", "ORDERING[NESTED/RING]");
    opts.optopt("t", "thetabins", "number of bins for theta", "NTHETA [20]");
    opts.optopt("p", "phibins", "number of bins for phi", "NPHI [1]");
    opts.optopt("z", "zmax", "the maximum value for zenith angle", "ZMAX [100.]");
    opts.optopt("n", "nthreads", "number of threads", "NTHR [1]");

    opts.optflag("l", "lincth", "whether the costheta bin is linear or not, default=False");

    // parse arg
    let matches = match opts.parse(&args[1..]) {
        Ok(v) => v,
        Err(_) => {
            // println!("{}", e);
            print!("{}", opts.usage(&brief));
            return;
        }
    };

    let scfile = matches.opt_str("scfile").unwrap();
    let evfile = matches.opt_str("evfile").unwrap();
    let outfile= matches.opt_str("outfile").unwrap();
    let lincth: bool = matches.opt_present("l");

    let order: u32 = match matches.opt_str("order") {
        Some(v) => u32::from_str(&v).expect("order should be an int"),
        None => 6,
    };

    let ntheta: usize = match matches.opt_str("thetabins") {
        Some(v) => usize::from_str(&v).expect("thetabins should be an int"),
        None => 20,
    };

    let nphi: usize = match matches.opt_str("phibins") {
        Some(v) => usize::from_str(&v).expect("phibins should be an int"),
        None => 1,
    };

    let zmax: f64 = match matches.opt_str("zmax") {
        Some(v) => f64::from_str(&v).expect("zmax should be a number"),
        None => 100.0,
    };

    let nthr: usize = match matches.opt_str("nthreads") {
        Some(v) => usize::from_str(&v).expect("nthreads should be an int"),
        None => 1,
    };

    // run
    ltcube_main(&scfile, &evfile, &outfile, order, ntheta, nphi, 0., zmax, nthr, lincth).unwrap();
}


////////////////////////
fn main() {
    parse_run();
}
