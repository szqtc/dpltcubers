use std::f64::consts;
// use ndarray::Array3;
use fitsio::FitsFile;
use cdshealpix::nested;

const R2D: f64 = 180./consts::PI;
const D2R: f64 = consts::PI/180.;
const CTHETA_MIN: f64 = 0.5;
const MET_REF: f64 = 1577836800.0; // 2016-01-01 in seconds

////////////////////////
// implement sc loader
#[derive(Debug, Default)]
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

// debug for sc
#[allow(dead_code)]
fn test_sc() {
    let sp = Spacecraft::read_file("test/test_sc.fits").unwrap();
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


////////////////////////
// implement gti loader
#[derive(Debug, Default)]
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

// debug for ev
#[allow(dead_code)]
fn test_ev() {
    let gti = GoodTimeInterval::read_file("test/test_ev.fits").unwrap();
    println!("-------test_ev------");
    println!("ndata={}", gti.len());
    println!("first term:");
    println!("  start={}, stop={}, fd={}", gti.tgstart[0], gti.tgstop[0], gti.fds[0]);
}


////////////////////////
#[allow(dead_code)]
fn test_healpix() {
    println!("-------test_healpix------");
    let order = 6_u8;
    let lon = 12.5_f64.to_radians();
    let lat = 13.2_f64.to_radians();

    let nside = cdshealpix::nside(order);
    let npix = 12*nside.pow(2);
    let ipix = nested::hash(order, lon, lat);
    let (lonr, latr) = nested::center(order, ipix);

    println!("order={}, nside={}, npix={}, ipix={}", order, nside, npix, ipix);
    println!("coord({},{}) -> ipix({}) -> coord({},{})", lon.to_degrees(), lat.to_degrees(), ipix, lonr.to_degrees(), latr.to_degrees());

    let nestobj = cdshealpix::nested::get(order);
    let ipix2 = nestobj.hash(lon, lat);
    let (lonr2, latr2) = nestobj.center(ipix);
    println!("=> ipix({}) -> coord({},{})", ipix2, lonr2.to_degrees(), latr2.to_degrees());
}

// main
fn main() {
    // test_sc();
    // test_ev();
    test_healpix();
}