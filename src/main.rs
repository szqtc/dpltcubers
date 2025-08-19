#![allow(dead_code)]

use std::f64::consts;
use std::str::FromStr;
use std::io::{Error, ErrorKind};
use fitsio::FitsFile;
use fitsio::tables::{ColumnDescription, ColumnDataType};
use hifitime::prelude::*;
use ndarray::prelude::*;
use ndarray::OwnedRepr;

mod bisect;

const R2D: f64 = 180./consts::PI;
const D2R: f64 = consts::PI/180.;
const CTHETA_MIN: f64 = 0.5;
const TLTRF_MET_REF: f64 = 94608001.0; // 2016-01-01 in seconds from 2013-01-01


////////////////////////
// sc loader
////////////////////////
// #[derive(Default)]
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
// #[derive(Default)]
struct GoodTimeIntervals {
    tgstart: Vec<f64>,
    tgstop : Vec<f64>,
    isfd64: Vec<bool>,
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
            isfd64 : hdu.read_col(&mut fptr, "FD")?.iter().map(|v:&i32| *v==64).collect(),
        };

        Ok(myobj)
    }
}


////////////////////////
// Healpix coordinates
////////////////////////
// #[derive(Default)]
struct HealpixCoordinates {
    order: u8,
    nside: u32,
    npix: u64,
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
        let mut ras_r: Vec<f64> = Vec::with_capacity(npix.try_into()?);
        let mut decs_r: Vec<f64> = Vec::with_capacity(npix.try_into()?);

        let nestobj = cdshealpix::nested::get(order);
        for ipix in 0..npix {
            let (ra_r, dec_r) = nestobj.center(ipix);
            ras_r.push(ra_r);
            decs_r.push(dec_r);
        }

        let myobj = Self {
            order: order,
            nside: 2_u32.pow(order.into()),
            npix: npix,
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

    #[inline]
    pub fn get_sindecs(&self) -> Vec<f64> {
        self.decs_r.iter().map(|v| v.sin()).collect()
    }

    #[inline]
    pub fn get_cosdecs(&self) -> Vec<f64> {
        self.decs_r.iter().map(|v| v.cos()).collect()
    }

}


////////////////////////
// dpltcube main
////////////////////////
fn get_time_str() -> String {
    let dt = chrono::Local::now();
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}


fn ltcube_worker(
    ltfd08: &mut ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
    ltfd64: &mut ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
    lttint: &mut ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
    lttint8: &mut ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
    scobj: &Spacecraft, gtiobj: &GoodTimeIntervals, hpxobj: &HealpixCoordinates,
    isclo: usize, ischi: usize,
    ipixlo: usize, ipixhi: usize) -> f64
{
    
    0.
}


pub fn ltcube_main(scfile: &str, evfile: &str, outfile: &str,
    order: u32, thetabins: u32, phibins: u32, zmax: f64, nthreads: u32, lincth: bool)
    -> Result<(), Box<dyn std::error::Error>>
{
    // echo the input params
    println!("[{}] scfile={} evfile={} outfile={} lincth={} order={} thetabins={} phibins={} zmax={} nthreads={}", get_time_str(), scfile, evfile, outfile, lincth, order, thetabins, phibins, zmax, nthreads);

    // read files
    println!("[{}] loading sc file ...", get_time_str());
    let scobj = Spacecraft::read_file(scfile)?;
    let nsc = scobj.len();
    if nsc == 0 {
        return Err(Box::new(Error::new(ErrorKind::InvalidData, "empty scfile")));
    }

    println!("[{}] loading ev file ...", get_time_str());
    let gtiobj = GoodTimeIntervals::read_file(evfile)?;
    let ngti = gtiobj.len();
    if ngti == 0 {
        return Err(Box::new(Error::new(ErrorKind::InvalidData, "empty gti list")));
    }

    let (gtimin, gtimax): (f64, f64) = (gtiobj.tgstart[0], gtiobj.tgstop[ngti-1]);
    if gtimin<scobj.tstart[0] || gtimax>scobj.tstop[nsc-1] {
        let errstr = format!("SC file ({}--{}) must cover the entire GTI ({}--{})", scobj.tstart[0], scobj.tstop[nsc-1], gtimin, gtimax);
        return Err(Box::new(Error::new(ErrorKind::InvalidData, errstr)));
    }

    // create data
    println!("[{}] preparing miscellaneous information ...", get_time_str());
    let hpxobj = HealpixCoordinates::init_with_order(order.try_into()?)?;
    let npix = hpxobj.len();

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

    let mut ltfd08 = Array3::<f64>::zeros([npix, phibins_usize, thetabins_usize]);
    let mut ltfd64 = Array3::<f64>::zeros([npix, phibins_usize, thetabins_usize]);
    let mut lttint = Array3::<f64>::zeros([npix, phibins_usize, thetabins_usize]);
    let mut lttint8= Array3::<f64>::zeros([npix, phibins_usize, thetabins_usize]);

    // find the min/max index
    println!("[{}] calculating ltcube ...", get_time_str());
    let stopwatch = std::time::Instant::now();

    let mut idxlo = bisect::bisect_right(&scobj.tstart, &gtimin)-1;
    let mut idxhi = bisect::bisect_left(&scobj.tstart, &gtimax)+1;
    idxlo = std::cmp::max(idxlo, 0);
    idxhi = std::cmp::min(idxhi, nsc-1);
    assert!(idxlo<=idxhi);

    let comp_f = |a:&f64, b:&f64| {a.partial_cmp(b).unwrap()};
    let tmin = std::cmp::max_by(gtimin, scobj.tstart[idxlo], comp_f);
    let tmax = std::cmp::min_by(gtimax, scobj.tstop[idxhi], comp_f);
    // debug
    println!("idxlo={}, idxhi={}", idxlo, idxhi);
    println!("gtitmin={}, gtitmax={}", gtimin, gtimax);
    println!("sctmin={}, sctmax={}", scobj.tstart[idxlo], scobj.tstop[idxhi]);
    println!("tmin={}, tmax={}", tmin, tmax);

    // run worker
    let lttot = ltcube_worker(&mut ltfd08, &mut ltfd64, &mut lttint, &mut lttint8,
            &scobj, &gtiobj, &hpxobj, idxlo, idxhi, 0, npix-1);
    // TODO: multithread use axis_chunks_iter_mut() with crossbeam

    let dtime = stopwatch.elapsed();
    println!("[{}]  runtime = {:?} x {}, livetime = {} sec", get_time_str(), dtime, nthreads, lttot);

    // create a new file & remove duplicated one
    match std::fs::metadata(outfile) {
        Ok(_) => {
            println!("[{}] remove the existing output file: {}", get_time_str(), outfile);
            std::fs::remove_file(outfile)
        },
        _ => Ok(()),
    }?;
    let mut fptr = FitsFile::create(outfile).open()?;

    // ext0
    let who = whoami::username();
    let timestr = get_time_str();
    println!("[{}] preparing save data ...", timestr);

    let hdu0 = fptr.primary_hdu()?;
    hdu0.write_key(&mut fptr, "TELESCOP", "DAMPE")?;
    hdu0.write_key(&mut fptr, "PHFILE", evfile)?;
    hdu0.write_key(&mut fptr, "SCFILE", scfile)?;
    hdu0.write_key(&mut fptr, "CREATOR", (who.clone(), timestr.clone()))?;
    hdu0.write_key(&mut fptr, "HISTORY", format!("dpltrs ver. {}", env!("CARGO_PKG_VERSION")))?;
    hdu0.write_key(&mut fptr, "HISTORY", format!("DT = {} [sec] x {}", dtime.as_secs_f32(), nthreads))?;

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
    hdu1.write_col(&mut fptr, "RA", &hpxobj.get_ras_deg())?;
    hdu1.write_col(&mut fptr, "DEC", &hpxobj.get_decs_deg())?;

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
    hdu1.write_key(&mut fptr, "LIVETIME", (lttot, "sec"))?;

    let trefutc = Epoch::from_gregorian_utc(2013, 1, 1, 0, 0, 0, 0).to_time_scale(hifitime::TimeScale::TAI);
    let tltutc = trefutc + hifitime::Duration::from_seconds(TLTRF_MET_REF);
    let tminutc = trefutc + hifitime::Duration::from_seconds(tmin);
    let tmaxutc = trefutc + hifitime::Duration::from_seconds(tmax);
    hdu1.write_key(&mut fptr, "TLTRFMET", (TLTRF_MET_REF, tltutc.to_gregorian_str(hifitime::TimeScale::UTC)))?;
    hdu1.write_key(&mut fptr, "TSTART", (tmin, tminutc.to_gregorian_str(hifitime::TimeScale::UTC)))?;
    hdu1.write_key(&mut fptr, "TSTOP", (tmax, tmaxutc.to_gregorian_str(hifitime::TimeScale::UTC)))?;
    hdu1.write_key(&mut fptr, "CREATOR", (who.clone(), timestr.clone()))?;

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
    hdu2.write_key(&mut fptr, "CREATOR", (who.clone(), timestr.clone()))?;

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
    hdu3.write_key(&mut fptr, "CREATOR", (who.clone(), timestr.clone()))?;

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
#[cfg(test)]
mod tests {
use super::*;

#[test]
fn test_readfiles() {
    ///////// scfile
    println!("-------test_sc------");
    let sp = Spacecraft::read_file("testfiles/test_sc.fits").unwrap();
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

    ///////// evfile
    println!("-------test_ev------");
    let gti = GoodTimeIntervals::read_file("testfiles/test_ev.fits").unwrap();
    println!("ndata={}", gti.len());
    println!("first term:");
    println!("  start={}, stop={}, fd={}", gti.tgstart[0], gti.tgstop[0], gti.isfd64[0]);

    //assertions
    assert_eq!(gti.len(), 62);
    assert_eq!(gti.tgstart[0], 94608354f64);
    assert_eq!(gti.tgstop[0], 94610134f64);
    assert_eq!(gti.isfd64[0], true);
}

#[test]
fn test_healpix() {
    println!("-------test_healpix------");
    let order = 6_u8;
    let lon = 12.5_f64*D2R;
    let lat = 13.2_f64*D2R;

    // simple run
    let nside = cdshealpix::nside(order);
    let npix = 12*nside.pow(2);
    let ipix = cdshealpix::nested::hash(order, lon, lat);
    let (lonr, latr) = cdshealpix::nested::center(order, ipix);

    println!("order={}, nside={}, npix={}, ipix={}", order, nside, npix, ipix);
    println!("coord({},{}) -> ipix({}) -> coord({},{})", lon*R2D, lat*R2D, ipix, lonr*R2D, latr*R2D);

    assert_eq!(nside, 64);
    assert_eq!(npix, 49152);
    assert_eq!(ipix, 19725);
    assert_eq!((lonr*R2D, latr*R2D), (11.953125, 13.24801490567695));

    // nest obj
    let nestobj = cdshealpix::nested::get(order);
    let ipix2 = nestobj.hash(lon, lat);
    let (lonr2, latr2) = nestobj.center(ipix);
    println!("=> ipix({}) -> coord({},{})", ipix2, lonr2*R2D, latr2*R2D);

    assert_eq!(ipix2, 19725);
    assert_eq!((lonr2*R2D, latr2*R2D), (11.953125, 13.24801490567695));

    // test hpxcrd obj
    let start = std::time::Instant::now();
    let hpxcrd = HealpixCoordinates::init_with_order(7).unwrap();
    let dt = start.elapsed();
    println!("init time = {:?}", dt);

    let sindecs = hpxcrd.get_sindecs();
    let cosdecs = hpxcrd.get_cosdecs();
    for ipix in 0..10 {
        println!("ipix={}, sindec={}, cosdec={}, ra_r={}", ipix, sindecs[ipix], cosdecs[ipix], hpxcrd.ras_r[ipix]);
    }
    let npix = hpxcrd.len();
    for ipix in npix-10..npix {
        println!("ipix={}, sindec={}, cosdec={}, ra_r={}", ipix, sindecs[ipix], cosdecs[ipix], hpxcrd.ras_r[ipix]);
    }

    // assertions
    assert_eq!((sindecs[0], cosdecs[0], hpxcrd.ras_r[0]), (0.005208333333333333, 0.9999864365399607, 0.7853981633974483));
    assert_eq!((sindecs[196607], cosdecs[196607], hpxcrd.ras_r[196607]), (-0.005208333333333333, 0.9999864365399607, 5.497787143782138));
}

#[test]
#[should_panic] // the bug will be resolved for hifitime >4.2
fn test_leapsecond() {
    // the test fails when the version of hifitime is < 4.2
    // see https://github.com/nyx-space/hifitime/issues/417
    use hifitime::{Epoch, Unit};
    let utc0 = Epoch::from_gregorian_utc_hms(2016, 12, 31, 23, 59, 0);
    let tai0 = utc0.to_time_scale(hifitime::TimeScale::TAI);

    // gh_417
    for dt in 1..62 {
        let tai_next = tai0 + Unit::Second * dt;
        let utc_next = utc0 + Unit::Second * dt;
        println!("+{dt} s => TAI = {tai_next}\tTAI as UTC = {tai_next:?}\tUTC = {utc_next}");
    }

    // time convertion around the leap second
    let utc3 = Epoch::from_gregorian_utc_hms(2016, 12, 31, 23, 59, 0);
    println!("T3={}, TAI3={}, UTC3={}", utc3.to_gregorian_str(hifitime::TimeScale::UTC), utc3.to_tai_seconds(), utc3.to_utc_seconds());
    assert_eq!(utc3.to_gregorian_str(hifitime::TimeScale::UTC), "2016-12-31T23:59:00 UTC");
    assert_eq!(utc3.to_gregorian_str(hifitime::TimeScale::TAI), "2016-12-31T23:59:36 TAI");

    let utc4 = Epoch::from_gregorian_utc_hms(2016, 12, 31, 23, 59, 59);
    println!("T4={}, TAI4={}, UTC4={}", utc4.to_gregorian_str(hifitime::TimeScale::UTC), utc4.to_tai_seconds(), utc4.to_utc_seconds());
    assert_eq!(utc4.to_gregorian_str(hifitime::TimeScale::UTC), "2016-12-31T23:59:59 UTC");
    assert_eq!(utc4.to_gregorian_str(hifitime::TimeScale::TAI), "2017-01-01T00:00:35 TAI");

    let utc5 = Epoch::from_gregorian_utc_hms(2016, 12, 31, 23, 59, 60);
    println!("T5={}, TAI5={}, UTC5={}", utc5.to_gregorian_str(hifitime::TimeScale::UTC), utc5.to_tai_seconds(), utc5.to_utc_seconds());
    assert_eq!(utc5.to_gregorian_str(hifitime::TimeScale::UTC), "2016-12-31T23:59:60 UTC");
    assert_eq!(utc5.to_gregorian_str(hifitime::TimeScale::TAI), "2017-01-01T00:00:36 TAI");

    let utc6 = Epoch::from_gregorian_utc_hms(2017, 1, 1, 0, 0, 0);
    println!("T6={}, TAI6={}, UTC6={}", utc6.to_gregorian_str(hifitime::TimeScale::UTC), utc6.to_tai_seconds(), utc6.to_utc_seconds());
    assert_eq!(utc6.to_gregorian_str(hifitime::TimeScale::UTC), "2017-01-01T00:00:00 UTC");
    assert_eq!(utc6.to_gregorian_str(hifitime::TimeScale::TAI), "2017-01-01T00:00:37 TAI");

    // delta time
    let dt43_tai = utc4.to_tai_seconds()-utc3.to_tai_seconds();
    let dt43_utc = utc4.to_utc_seconds()-utc3.to_utc_seconds();
    println!("T4-T3 = {dt43_tai} (TAI) = {dt43_utc} (UTC)");
    assert_eq!(dt43_tai, 59.);
    assert_eq!(dt43_utc, 59.);

    let dt64_tai = utc6.to_tai_seconds()-utc4.to_tai_seconds();
    let dt64_utc = utc6.to_utc_seconds()-utc4.to_utc_seconds();
    println!("T6-T4 = {dt64_tai} (TAI) = {dt64_utc} (UTC)");
    assert_eq!(dt64_tai, 2.);
    assert_eq!(dt64_utc, 2.);

    let dt65_tai = utc6.to_tai_seconds()-utc5.to_tai_seconds();
    let dt65_utc = utc6.to_utc_seconds()-utc5.to_utc_seconds();
    println!("T6-T5 = {dt65_tai} (TAI) = {dt65_utc} (UTC)");
    assert_eq!(dt65_tai, 1.);
    assert_eq!(dt65_utc, 1.);

    let dt63_tai = utc6.to_tai_seconds()-utc3.to_tai_seconds();
    let dt63_utc = utc6.to_utc_seconds()-utc3.to_utc_seconds();
    println!("T6-T3 = {dt63_tai} (TAI) = {dt63_utc} (UTC)");
    assert_eq!(dt63_tai, 61.);
    assert_eq!(dt63_utc, 61.);

    // more cases
    let utc7a = Epoch::from_gregorian_utc_hms(2017, 12, 31, 23, 59, 59);
    let utc7b = Epoch::from_gregorian_utc_hms(2018, 1, 1, 0, 0, 0);
    assert_eq!(utc7b.to_tai_seconds()-utc7a.to_tai_seconds(), 1.);
    assert_eq!(utc7b.to_utc_seconds()-utc7a.to_utc_seconds(), 1.);

    let utc8a = Epoch::from_gregorian_utc_hms(2015, 6, 30, 23, 59, 59);
    let utc8b = Epoch::from_gregorian_utc_hms(2015, 7, 1, 0, 0, 0);
    assert_eq!(utc8b.to_tai_seconds()-utc8a.to_tai_seconds(), 2.);
    assert_eq!(utc8b.to_utc_seconds()-utc8a.to_utc_seconds(), 2.);

    let utc9a = Epoch::from_gregorian_utc_hms(1972, 1, 1, 0, 0, 0);
    let utc9b = Epoch::from_gregorian_utc_hms(2025, 1, 1, 0, 0, 0);
    assert_eq!(utc9b.to_tai_seconds()-utc9a.to_tai_seconds(), 1672617627.);
    assert_eq!(utc9b.to_utc_seconds()-utc9a.to_utc_seconds(), 1672617627.);
}

#[test]
fn test_bisect() {
    let arr1 = [0,2,4,6,8,10];
    println!("int arr: {:?}", arr1);

    let mut num1;
    let mut idx1;

    // case 1.0
    //       v --- bisect left/right
    // [0, 2, 4, 6, 8, 10]
    num1 = 3;
    idx1 = bisect::bisect_right(&arr1, &num1);
    println!("  bisect_right: num={}, idx={}, left={}, right={}", num1, idx1, arr1[idx1-1], arr1[idx1]);
    assert_eq!(idx1, 2);

    idx1 = bisect::bisect_left(&arr1, &num1);
    println!("  bisect_left: num={}, idx={}, left={}, right={}", num1, idx1, arr1[idx1-1], arr1[idx1]);
    assert_eq!(idx1, 2);

    // case 1.1
    //       v --- bisect left
    //          v --- bisect right
    // [0, 2, 4, 6, 8, 10]
    num1 = 4;
    idx1 = bisect::bisect_right(&arr1, &num1);
    println!("  bisect_right: num={}, idx={}, left={}, right={}", num1, idx1, arr1[idx1-1], arr1[idx1]);
    assert_eq!(idx1, 3);

    idx1 = bisect::bisect_left(&arr1, &num1);
    println!("  bisect_left: num={}, idx={}, left={}, right={}", num1, idx1, arr1[idx1-1], arr1[idx1]);
    assert_eq!(idx1, 2);

    // case 1.2
    num1 = 10;
    idx1 = bisect::bisect_right(&arr1, &num1);
    println!("  bisect_right: num={}, idx={}", num1, idx1);
    assert_eq!(idx1, 6);

    idx1 = bisect::bisect_left(&arr1, &num1);
    println!("  bisect_left: num={}, idx={}", num1, idx1);
    assert_eq!(idx1, 5);

    // case 1.3
    num1 = 11;
    idx1 = bisect::bisect_right(&arr1, &num1);
    println!("  bisect_right: num={}, idx={}", num1, idx1);
    assert_eq!(idx1, 6);

    idx1 = bisect::bisect_left(&arr1, &num1);
    println!("  bisect_left: num={}, idx={}", num1, idx1);
    assert_eq!(idx1, 6);

    // case 1.4
    num1 = -1;
    idx1 = bisect::bisect_right(&arr1, &num1);
    println!("  bisect_right: num={}, idx={}", num1, idx1);
    assert_eq!(idx1, 0);

    idx1 = bisect::bisect_left(&arr1, &num1);
    println!("  bisect_left: num={}, idx={}", num1, idx1);
    assert_eq!(idx1, 0);

    // float arr
    let arr2 = [0., 2., 4., 6., 8., 10.];
    println!("float arr: {:?}", arr2);

    let mut num2;
    let mut idx2;

    // case 2.0
    //       v --- bisect left/right
    // [0, 2, 4, 6, 8, 10]
    num2 = 3.1;
    idx2 = bisect::bisect_right(&arr2, &num2);
    println!("  bisect_right: num={}, idx={}, left={}, right={}", num2, idx2, arr2[idx2-1], arr2[idx2]);
    assert_eq!(idx2, 2);

    idx2 = bisect::bisect_left(&arr2, &num2);
    println!("  bisect_left: num={}, idx={}, left={}, right={}", num2, idx2, arr2[idx2-1], arr2[idx2]);
    assert_eq!(idx2, 2);

    // case 2.1
    //       v --- bisect left
    //          v --- bisect right
    // [0, 2, 4, 6, 8, 10]
    num2 = 4.0;
    idx2 = bisect::bisect_right(&arr2, &num2);
    println!("  bisect_right: num={}, idx={}, left={}, right={}", num2, idx2, arr2[idx2-1], arr2[idx2]);
    assert_eq!(idx2, 3);

    idx2 = bisect::bisect_left(&arr2, &num2);
    println!("  bisect_left: num={}, idx={}, left={}, right={}", num2, idx2, arr2[idx2-1], arr2[idx2]);
    assert_eq!(idx2, 2);

    // case 2.2
    num2 = 10.;
    idx2 = bisect::bisect_right(&arr2, &num2);
    println!("  bisect_right: num={}, idx={}", num2, idx2);
    assert_eq!(idx2, 6);

    idx2 = bisect::bisect_left(&arr2, &num2);
    println!("  bisect_left: num={}, idx={}", num2, idx2);
    assert_eq!(idx2, 5);

    // case 2.3
    num2 = 11.;
    idx2 = bisect::bisect_right(&arr2, &num2);
    println!("  bisect_right: num={}, idx={}", num2, idx2);
    assert_eq!(idx2, 6);

    idx2 = bisect::bisect_left(&arr2, &num2);
    println!("  bisect_left: num={}, idx={}", num2, idx2);
    assert_eq!(idx2, 6);

    // case 2.4
    num2 = -1.;
    idx2 = bisect::bisect_right(&arr2, &num2);
    println!("  bisect_right: num={}, idx={}", num2, idx2);
    assert_eq!(idx2, 0);

    idx2 = bisect::bisect_left(&arr2, &num2);
    println!("  bisect_left: num={}, idx={}", num2, idx2);
    assert_eq!(idx2, 0);
}


}

////////////////////////
fn main() {
    // parse_run();
    // test_leapsecond();

    ltcube_main("testfiles/test_sc.fits", "testfiles/test_ev.fits", "testfiles/output_ltcube.fits", 5, 20, 2, 100., 1, false).unwrap();
}