#![allow(dead_code)]

use std::f64::consts;
use std::str::FromStr;
// use ndarray::Array3;
use fitsio::FitsFile;
use fitsio::tables::{ColumnDescription, ColumnDataType};
use chrono::prelude::*;
use ndarray::prelude::*;

const R2D: f64 = 180./consts::PI;
const D2R: f64 = consts::PI/180.;
const CTHETA_MIN: f64 = 0.5;
const MET_REF: f64 = 94608001.0; // 2016-01-01 in seconds

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

    #[inline]
    pub fn len(&self) -> usize {
        self.tstart.len()
    }

    pub fn read_file(scfile: &str) -> Result<Self, fitsio::errors::Error> {
        let mut fptr = match FitsFile::open(scfile) {
            Ok(ptr) => ptr,
            Err(_) => panic!("scfile not found: {} (path={})", scfile, std::env::current_dir().unwrap().display()),
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
struct GoodTimeIntervals {
    tgstart: Vec<f64>,
    tgstop : Vec<f64>,
    fds: Vec<i32>,
}

impl GoodTimeIntervals {

    #[inline]
    pub fn len(&self) -> usize {
        self.tgstart.len()
    }

    pub fn read_file(evfile: &str) -> Result<Self, fitsio::errors::Error> {
        let mut fptr = match FitsFile::open(evfile) {
            Ok(ptr) => ptr,
            Err(_) => panic!("evfile not found: {} (path={})", evfile, std::env::current_dir().unwrap().display()),
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
    decs_r: Vec<f64>,
}

impl HealpixCoordinates {

    #[inline]
    pub fn len(&self) -> usize {
        self.npix.try_into().unwrap()
    }

    pub fn init_with_order(order: u8) -> Result<Self, Box<dyn std::error::Error>> {
        let npix: u64 = 12*(2_u64.pow((order*2).into()));
        let mut sindecs: Vec<f64> = Vec::with_capacity(npix.try_into()?);
        let mut cosdecs: Vec<f64> = Vec::with_capacity(npix.try_into()?);
        let mut ras_r: Vec<f64> = Vec::with_capacity(npix.try_into()?);
        let mut decs_r: Vec<f64> = Vec::with_capacity(npix.try_into()?);

        let nestobj = cdshealpix::nested::get(order);
        for ipix in 0..npix {
            let (ra_r, dec_r) = nestobj.center(ipix);
            ras_r.push(ra_r);
            decs_r.push(dec_r);
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
            decs_r: decs_r,
        };

        Ok(myobj)
    }

    #[inline]
    pub fn get_ras_deg(&self) -> Vec<f64> {
        self.ras_r.iter().map(|v| v*R2D).collect()
    }

    #[inline]
    pub fn get_decs_deg(&self) -> Vec<f64> {
        self.decs_r.iter().map(|v| v*R2D).collect()
    }
}


////////////////////////
// dpltcube main
////////////////////////
fn get_time_str() -> String {
    let dt = Local::now();
    dt.format("%Y-%m-%dT%H:%M:%S").to_string()
}


fn ltcube_worker() {
    
}


fn ltcube_main(scfile: &str, evfile: &str, outfile: &str,
    order: u32, thetabins: u32, phibins: u32, zmax: f64, nthreads: u32, lincth: bool)
    -> Result<(), Box<dyn std::error::Error>>
{
    // echo the input params
    println!("[{}] scfile={} evfile={} outfile={} lincth={} order={} thetabins={} phibins={} zmax={} nthreads={}", get_time_str(), scfile, evfile, outfile, lincth, order, thetabins, phibins, zmax, nthreads);

    // read files
    println!("[{}] loading sc file ...", get_time_str());
    let spobj = Spacecraft::read_file(scfile)?;
    println!("[{}] loading ev file ...", get_time_str());
    let gtiobj = GoodTimeIntervals::read_file(evfile)?;

    // create data
    println!("[{}] preparing miscellaneous information ...", get_time_str());
    let hpxobj = HealpixCoordinates::init_with_order(order.try_into()?)?;
    let npix = hpxobj.len();
    let ras_deg = hpxobj.get_ras_deg();
    let decs_deg = hpxobj.get_decs_deg();

    let thetabins_usize: usize = thetabins.try_into()?;
    let phibins_usize: usize = phibins.try_into()?;
    let cthetabounds = if lincth {
        Array::linspace(CTHETA_MIN, 1., thetabins_usize+1)
    } else {
        Array::linspace(1., 0., thetabins_usize+1).map(|v| 1.-(1.-CTHETA_MIN)*v*v)
    }.to_vec();
    let phibounds = Array::linspace(0., 360., phibins_usize+1).to_vec();
    // println!("{:?}", cthetabounds);
    // println!("{:?}", phibounds);

    let mut lttot = 0.;
    let mut ltfd08 = Array3::<f64>::zeros([npix, phibins_usize, thetabins_usize]);
    let mut ltfd64 = Array3::<f64>::zeros([npix, phibins_usize, thetabins_usize]);
    let mut lttint = Array3::<f64>::zeros([npix, phibins_usize, thetabins_usize]);
    let mut lttint8= Array3::<f64>::zeros([npix, phibins_usize, thetabins_usize]);

    /*
    // compare with DAMPE_ltcube_20240901_phi_nside128_wint.fits[HL]
    ltfd08[[0,0,0]]=50882.426;
    ltfd08[[0,0,1]]=48357.695;
    ltfd08[[1,0,1]]=48392.543;
    ltfd08[[1,1,1]]=48473.54;
    */

    // run the workers
    println!("[{}] calculating ltcube ...", get_time_str());
    let stopwatch = std::time::Instant::now();


    let dtime = stopwatch.elapsed();
    println!("[{}] DT = {:?} x {}", get_time_str(), dtime, nthreads);

    // create a new file & remove duplicated one
    match std::fs::metadata(outfile) {
        Ok(_) => {println!("[{}] remove the existing output file: {}", get_time_str(), outfile);
                  std::fs::remove_file(outfile)},
        _ => Ok(()),
    }?;
    let mut fptr = FitsFile::create(outfile).open()?;

    // ext0
    println!("[{}] preparing save data ...", get_time_str());
    let who = whoami::username();
    let hdu0 = fptr.primary_hdu()?;

    hdu0.write_key(&mut fptr, "TELESCOP", "DAMPE")?;
    hdu0.write_key(&mut fptr, "PHFILE", evfile)?;
    hdu0.write_key(&mut fptr, "SCFILE", scfile)?;
    hdu0.write_key(&mut fptr, "CREATOR", (who.clone(), get_time_str()))?;
    hdu0.write_key(&mut fptr, "HISTORY", format!("dpltrs ver {}", env!("CARGO_PKG_VERSION")))?;
    hdu0.write_key(&mut fptr, "HISTORY", format!("DT = {:?} x {}", dtime, nthreads))?;

    // ext1
    let nvec: usize = (thetabins*phibins).try_into()?;
    let col1 = ColumnDescription::new("LL").with_type(ColumnDataType::Float).that_repeats(nvec).create()?;
    let col2 = ColumnDescription::new("HL").with_type(ColumnDataType::Float).that_repeats(nvec).create()?;
    let col3 = ColumnDescription::new("TINT").with_type(ColumnDataType::Float).that_repeats(nvec).create()?;
    let col4 = ColumnDescription::new("TINTLL").with_type(ColumnDataType::Float).that_repeats(nvec).create()?;
    let col5 = ColumnDescription::new("RA").with_type(ColumnDataType::Float).create()?;
    let col6 = ColumnDescription::new("DEC").with_type(ColumnDataType::Float).create()?;
    let des1 = [col1, col2, col3, col4, col5, col6];
    let hdu1 = fptr.create_table("EXPOSURE".to_string(), &des1)?;

    hdu1.write_col(&mut fptr, "LL", &ltfd08.flatten().to_vec())?;
    hdu1.write_col(&mut fptr, "HL", &ltfd64.flatten().to_vec())?;
    hdu1.write_col(&mut fptr, "TINT", &lttint.flatten().to_vec())?;
    hdu1.write_col(&mut fptr, "TINTLL", &lttint8.flatten().to_vec())?;
    hdu1.write_col(&mut fptr, "RA", &ras_deg)?;
    hdu1.write_col(&mut fptr, "DEC", &decs_deg)?;

    hdu1.write_key(&mut fptr, "TELESCOP", "DAMPE")?;
    hdu1.write_key(&mut fptr, "ZMAX", (zmax, "deg"))?;
    hdu1.write_key(&mut fptr, "ORDERING", "NESTED")?;
    hdu1.write_key(&mut fptr, "PIXTYPE", "HEALPIX")?;
    hdu1.write_key(&mut fptr, "NSIDE", hpxobj.nside)?;
    hdu1.write_key(&mut fptr, "COORDSYS", "CEL")?;
    hdu1.write_key(&mut fptr, "RADECSYS", ("FK5", "J2000"))?;
    hdu1.write_key(&mut fptr, "FIRSTPIX", 0)?;
    hdu1.write_key(&mut fptr, "LASTPIX", hpxobj.npix-1)?;
    hdu1.write_key(&mut fptr, "INDXSCHM", "IMPLICIT")?;
    hdu1.write_key(&mut fptr, "TIMESYS", "TT")?;
    hdu1.write_key(&mut fptr, "TIMEUNIT", "s")?;
    hdu1.write_key(&mut fptr, "TLTRFMET", (MET_REF, "TLT ref. time in MET"))?;
    hdu1.write_key(&mut fptr, "LIVETIME", (lttot, "sec"))?;
    /*
    t0 = utcref + atime.TimeDelta(gtistart[0], scale='tt', format='sec')
    hdu1.header['TSTART'] = (gtistart[0], t0.isot)
    t1 = utcref + atime.TimeDelta(gtistart[-1], scale='tt', format='sec')
    hdu1.header['TSTOP'] = (gtistop[-1], t1.isot)
     */
    hdu1.write_key(&mut fptr, "CREATOR", (who.clone(), get_time_str()))?;

    // ext2
    let col21 = ColumnDescription::new("CTHETA_MIN").with_type(ColumnDataType::Float).create()?;
    let col22 = ColumnDescription::new("CTHETA_MAX").with_type(ColumnDataType::Float).create()?;
    let des2 = [col21, col22];
    let hdu2 = fptr.create_table("CTHETABOUNDS".to_string(), &des2)?;

    hdu2.write_col(&mut fptr, "CTHETA_MIN", &cthetabounds[0..thetabins_usize])?;
    hdu2.write_col(&mut fptr, "CTHETA_MAX", &cthetabounds[1..thetabins_usize+1])?;

    hdu2.write_key(&mut fptr, "TELESCOP", "DAMPE")?;
    hdu2.write_key(&mut fptr, "CTHBINAL", if lincth {"LIN"} else {"SQ"})?;
    hdu2.write_key(&mut fptr, "CTHMIN", CTHETA_MIN)?;
    hdu2.write_key(&mut fptr, "NCTHBIN", thetabins)?;
    hdu2.write_key(&mut fptr, "CREATOR", (who.clone(), get_time_str()))?;

    // ext3
    let col31 = ColumnDescription::new("PHI_MIN").with_type(ColumnDataType::Float).create()?;
    let col32 = ColumnDescription::new("PHI_MAX").with_type(ColumnDataType::Float).create()?;
    let des3 = [col31, col32];
    let hdu3 = fptr.create_table("PHIBOUNDS".to_string(), &des3)?;

    hdu3.write_col(&mut fptr, "PHI_MIN", &phibounds[0..phibins_usize])?;
    hdu3.write_col(&mut fptr, "PHI_MAX", &phibounds[1..phibins_usize+1])?;

    hdu3.write_key(&mut fptr, "TELESCOP", "DAMPE")?;
    hdu3.write_key(&mut fptr, "SCAXIS", "-X-Y-Z")?;
    hdu3.write_key(&mut fptr, "NPHIBIN", phibins)?;
    hdu3.write_key(&mut fptr, "PHIBINAL", "LIN")?;
    hdu3.write_key(&mut fptr, "CREATOR", (who.clone(), get_time_str()))?;

    println!("[{}] done", get_time_str());
    Ok(())
}


////////////////////////
// parse arguments
////////////////////////
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
    ltcube_main(&scfile, &evfile, &outfile, order, ntheta, nphi, zmax, nthr, lincth).unwrap();
}


////////////////////////
// tests
////////////////////////
#[test]
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

    //assertions
    assert_eq!(sp.len(), 86400);
    assert_eq!(sp.tstart[0], 94608000f64);
    assert_eq!(sp.tstop[0], 94608001f64);
    assert_eq!((sp.ra_x[0], sp.dec_x[0]), (43.325313568115234f64,-73.42020416259766f64));
    assert_eq!((sp.ra_y[0], sp.dec_y[0]), (107.74408721923828f64,7.325563430786133f64));
    assert_eq!((sp.ra_z[0], sp.dec_z[0]), (15.798971176147461f64,14.79005241394043f64));
    assert_eq!((sp.ra_zth[0], sp.dec_zth[0]), (195.79896545410156f64,-14.79005241394043f64));
    assert_eq!(sp.lvtime[0], 0.8525199890136719f64);
    assert!(!sp.in_saa[0]);
}

#[test]
fn test_ev() {
    let gti = GoodTimeIntervals::read_file("testfiles/test_ev.fits").unwrap();
    println!("-------test_ev------");
    println!("ndata={}", gti.len());
    println!("first term:");
    println!("  start={}, stop={}, fd={}", gti.tgstart[0], gti.tgstop[0], gti.fds[0]);

    //assertions
    assert_eq!(gti.len(), 62);
    assert_eq!(gti.tgstart[0], 94608354f64);
    assert_eq!(gti.tgstop[0], 94610134f64);
    assert_eq!(gti.fds[0], 64i32);
}

#[test]
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
    // assertions
    assert_eq!(nside, 64);
    assert_eq!(npix, 49152);
    assert_eq!(ipix, 19725);
    assert_eq!((lonr*R2D, latr*R2D), (11.953125, 13.24801490567695));

    let nestobj = cdshealpix::nested::get(order);
    let ipix2 = nestobj.hash(lon, lat);
    let (lonr2, latr2) = nestobj.center(ipix);
    println!("=> ipix({}) -> coord({},{})", ipix2, lonr2*R2D, latr2*R2D);
    // assertions
    assert_eq!(ipix2, 19725);
    assert_eq!((lonr2*R2D, latr2*R2D), (11.953125, 13.24801490567695));

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
    // assertions
    assert_eq!((hpxcrd.sindecs[0], hpxcrd.cosdecs[0], hpxcrd.ras_r[0]), (0.005208333333333333, 0.9999864365399607, 0.7853981633974483));
    assert_eq!((hpxcrd.sindecs[196607], hpxcrd.cosdecs[196607], hpxcrd.ras_r[196607]), (-0.005208333333333333, 0.9999864365399607, 5.497787143782138));
}


////////////////////////
fn main() {
    // parse_run();

    ltcube_main("testfiles/test_sc.fits", "testfiles/test_ev.fits", "testfiles/output_ltcube.fits", 5, 20, 2, 100., 1, false).unwrap();
}