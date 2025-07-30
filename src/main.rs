#![allow(dead_code)]

use std::f64::consts;
use std::str::FromStr;
// use ndarray::Array3;
use fitsio::FitsFile;
use cdshealpix::nested;

const R2D: f64 = 180./consts::PI;
const D2R: f64 = consts::PI/180.;
const CTHETA_MIN: f64 = 0.5;
const MET_REF: f64 = 1577836800.0; // 2016-01-01 in seconds

////////////////////////
// sc loader
////////////////////////
#[derive(Default)]
pub struct Spacecraft {
    tstart:  Vec<f64>,
    tstop:   Vec<f64>,
    ra_x:    Vec<f64>,
    ra_y:    Vec<f64>,
    ra_z:    Vec<f64>,
    dec_x:   Vec<f64>,
    dec_y:   Vec<f64>,
    dec_z:   Vec<f64>,
    ra_zth:  Vec<f64>,
    dec_zth: Vec<f64>,
    lvtime:  Vec<f64>,
    in_saa:  Vec<bool>,
}

impl Spacecraft {

    pub fn len(&self) -> usize {
        self.tstart.len()
    }

    pub fn read_file(scfile: &str) -> Result<Self, fitsio::errors::Error> {
        let mut fptr = match FitsFile::open(scfile) {
            Ok(ptr) => ptr,
            Err(_) => panic!("scfile not found: {}", scfile),
        };
        // fptr.pretty_print()?; // show fitsinfo
        let hdu = fptr.hdu("SC_DATA")?;

        // read data
        let myobj = Self{
            tstart : hdu.read_col(&mut fptr, "START")?,
            tstop  : hdu.read_col(&mut fptr, "STOP")?,
            ra_x   : hdu.read_col(&mut fptr, "RA_SCX")?,
            ra_y   : hdu.read_col(&mut fptr, "RA_SCY")?,
            ra_z   : hdu.read_col(&mut fptr, "RA_SCZ")?,
            dec_x  : hdu.read_col(&mut fptr, "DEC_SCX")?,
            dec_y  : hdu.read_col(&mut fptr, "DEC_SCY")?,
            dec_z  : hdu.read_col(&mut fptr, "DEC_SCZ")?,
            ra_zth : hdu.read_col(&mut fptr, "RA_ZENITH")?,
            dec_zth: hdu.read_col(&mut fptr, "DEC_ZENITH")?,
            lvtime : hdu.read_col(&mut fptr, "LIVETIME")?,
            in_saa : hdu.read_col(&mut fptr, "IN_SAA")?,
        };

        Ok(myobj)
    }
}


////////////////////////
// gti loader
////////////////////////
#[derive(Default)]
struct GoodTimeInterval {
    tgstart: Vec<f64>,
    tgstop : Vec<f64>,
    fds: Vec<i32>,
}

impl GoodTimeInterval {

    pub fn len(&self) -> usize {
        self.tgstart.len()
    }

    pub fn read_file(evfile: &str) -> Result<Self, fitsio::errors::Error> {
        let mut fptr = match FitsFile::open(evfile) {
            Ok(ptr) => ptr,
            Err(_) => panic!("evfile not found: {}", evfile),
        };
        // fptr.pretty_print()?; // show fitsinfo
        let hdu = fptr.hdu("GTI")?;

        let myobj = Self{
            tgstart: hdu.read_col(&mut fptr, "START")?,
            tgstop : hdu.read_col(&mut fptr, "STOP")?,
            fds : hdu.read_col(&mut fptr, "FD")?,
        };

        Ok(myobj)
    }
}


////////////////////////
// Healpix coordinates
////////////////////////
#[derive(Default)]
struct HealpixCoordinates {
    order: u8,
    nside: u32,
    npix: u64,
    sindecs: Vec<f64>,
    cosdecs: Vec<f64>,
    ras_r: Vec<f64>,
}

impl HealpixCoordinates {

    pub fn len(&self) -> usize {
        self.npix.try_into().unwrap()
    }

    pub fn init_with_order(order: u8) -> Result<Self, Box<dyn std::error::Error>> {
        let npix: u64 = 12*(2_u64.pow((order*2).into()));
        let mut sindecs: Vec<f64> = Vec::with_capacity(npix.try_into()?);
        let mut cosdecs: Vec<f64> = Vec::with_capacity(npix.try_into()?);
        let mut ras_r: Vec<f64> = Vec::with_capacity(npix.try_into()?);

        let nestobj = cdshealpix::nested::get(order);
        for ipix in 0..npix {
            let (ra_r, dec_r) = nestobj.center(ipix);
            ras_r.push(ra_r);
            sindecs.push(dec_r.sin());
            cosdecs.push(dec_r.cos());
        }

        let myobj = Self {
            order: order,
            nside: 2_u32.pow(order.into()),
            npix: npix,
            sindecs: sindecs,
            cosdecs: cosdecs,
            ras_r: ras_r,
        };

        Ok(myobj)
    }
}


////////////////////////
// dpltcube main
////////////////////////
fn ltcube_worker() {
    
}


fn ltcube_main(scfile: &str, evfile: &str, outfile: &str,
    order: u32, thetabins: u32, phibins: u32, zmax: f64, nthreads: u32, lincth: bool)
{
    // debug
    println!("-s {} -e {} -o {}", scfile, evfile, outfile);
    println!("--lincth {} --order {} --thetabins {} --phibins {} --zmax {} --nthreads {}", lincth, order, thetabins, phibins, zmax, nthreads);
}


////////////////////////
// parse arguments
////////////////////////
#[allow(dead_code)]
fn parse_run() {
    // define arg
    let args: Vec<String> = std::env::args().collect();
    let prog = args[0].clone();
    let brief = format!("Usage: {} -s SCFILE -e EVFILE -o OUTFILE [options]\na simple tool to calculate the livetime cube of DAMPE", prog);

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

    let ntheta: u32 = match matches.opt_str("thetabins") {
        Some(v) => u32::from_str(&v).expect("thetabins should be an int"),
        None => 20,
    };

    let nphi: u32 = match matches.opt_str("phibins") {
        Some(v) => u32::from_str(&v).expect("phibins should be an int"),
        None => 1,
    };

    let zmax: f64 = match matches.opt_str("zmax") {
        Some(v) => f64::from_str(&v).expect("zmax should be a number"),
        None => 100.0,
    };

    let nthr: u32 = match matches.opt_str("nthreads") {
        Some(v) => u32::from_str(&v).expect("nthreads should be an int"),
        None => 1,
    };

    // run
    ltcube_main(&scfile, &evfile, &outfile, order, ntheta, nphi, zmax, nthr, lincth);
}

////////////////////////
// tests
////////////////////////
#[allow(dead_code)]
fn test_sc() {
    let sp = Spacecraft::read_file("testfiles/test_sc.fits").unwrap();
    println!("-------test_sc------");
    println!("ndata={}", sp.len());
    println!("first term:");
    println!("  tstart={}, tstop={}, posx=({},{}), posy=({},{}), posz=({},{}), poszth=({},{}), lvt={}, insaa={}",
        sp.tstart[0], sp.tstop[0],
        sp.ra_x[0], sp.dec_x[0],
        sp.ra_y[0], sp.dec_y[0],
        sp.ra_z[0], sp.dec_z[0],
        sp.ra_zth[0], sp.dec_zth[0],
        sp.lvtime[0], sp.in_saa[0]);
}

#[allow(dead_code)]
fn test_ev() {
    let gti = GoodTimeInterval::read_file("testfiles/test_ev.fits").unwrap();
    println!("-------test_ev------");
    println!("ndata={}", gti.len());
    println!("first term:");
    println!("  start={}, stop={}, fd={}", gti.tgstart[0], gti.tgstop[0], gti.fds[0]);
}

#[allow(dead_code)]
fn test_healpix() {
    println!("-------test_healpix------");
    let order = 6_u8;
    let lon = 12.5_f64*D2R;
    let lat = 13.2_f64*D2R;

    let nside = cdshealpix::nside(order);
    let npix = 12*nside.pow(2);
    let ipix = nested::hash(order, lon, lat);
    let (lonr, latr) = nested::center(order, ipix);

    println!("order={}, nside={}, npix={}, ipix={}", order, nside, npix, ipix);
    println!("coord({},{}) -> ipix({}) -> coord({},{})", lon*R2D, lat*R2D, ipix, lonr*R2D, latr*R2D);

    let nestobj = cdshealpix::nested::get(order);
    let ipix2 = nestobj.hash(lon, lat);
    let (lonr2, latr2) = nestobj.center(ipix);
    println!("=> ipix({}) -> coord({},{})", ipix2, lonr2*R2D, latr2*R2D);

    // test hpxcrd obj
    let start = std::time::Instant::now();
    let hpxcrd = HealpixCoordinates::init_with_order(7).unwrap();
    let dt = start.elapsed();
    println!("init time = {:?}", dt);

    for ipix in 0..10 {
        println!("ipix={}, sindec={}, cosdec={}, ra_r={}", ipix, hpxcrd.sindecs[ipix], hpxcrd.cosdecs[ipix], hpxcrd.ras_r[ipix]);
    }
    let npix = hpxcrd.len();
    for ipix in npix-10..npix {
        println!("ipix={}, sindec={}, cosdec={}, ra_r={}", ipix, hpxcrd.sindecs[ipix], hpxcrd.cosdecs[ipix], hpxcrd.ras_r[ipix]);
    }
}

#[allow(dead_code)]
fn test_all() {
    // test_sc();
    // test_ev();
    test_healpix();
}


////////////////////////
fn main() {
    test_all();
    // parse_run();

}