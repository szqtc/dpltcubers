use std::f64::consts;
use std::thread;
use std::io::{Error, ErrorKind};
use itertools::izip;
use ndarray::prelude::*;
use ndarray::ViewRepr;
use hifitime::prelude::*;
use fitsio::FitsFile;
use fitsio::tables::{ColumnDescription, ColumnDataType};
use indicatif::{ProgressBar, MultiProgress, ProgressStyle};

pub mod bisect;

pub const R2D: f64 = 180./consts::PI;
pub const D2R: f64 = consts::PI/180.;
pub const CTHETA_MIN: f64 = 0.5;
pub const TLTRF_MET_REF: f64 = 94608001.0; // 2016-01-01 in seconds from 2013-01-01


////////////////////////
// sc loader
////////////////////////
// #[derive(Default)]
pub struct Spacecraft {
    pub tstart:  Vec<f64>,
    pub tstop:   Vec<f64>,
    pub ra_x:    Vec<f64>,
    pub ra_y:    Vec<f64>,
    pub ra_z:    Vec<f64>,
    pub dec_x:   Vec<f64>,
    pub dec_y:   Vec<f64>,
    pub dec_z:   Vec<f64>,
    pub ra_zth:  Vec<f64>,
    pub dec_zth: Vec<f64>,
    pub lvtime:  Vec<f64>,
    pub in_saa:  Vec<bool>,
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
pub struct GoodTimeIntervals {
    pub tgstart: Vec<f64>,
    pub tgstop : Vec<f64>,
    pub isfd64: Vec<bool>,
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
pub struct HealpixCoordinates {
    pub order: u8,
    pub nside: u32,
    pub npix: u64,
    pub ras_r: Vec<f64>,
    pub decs_r: Vec<f64>,
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

}


////////////////////////
// dpltcube main
////////////////////////
pub fn get_time_str() -> String {
    let dt = chrono::Local::now();
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}

fn ltcube_worker(
    ltfd08: &mut ArrayBase<ViewRepr<&mut f64>, Dim<[usize; 3]>>,
    ltfd64: &mut ArrayBase<ViewRepr<&mut f64>, Dim<[usize; 3]>>,
    lttint: &mut ArrayBase<ViewRepr<&mut f64>, Dim<[usize; 3]>>,
    lttint8: &mut ArrayBase<ViewRepr<&mut f64>, Dim<[usize; 3]>>,
    scobj: &Spacecraft, gtiobj: &GoodTimeIntervals, hpxobj: &HealpixCoordinates,
    zmin: f64, zmax: f64, ntheta: usize, nphi: usize, lincth: bool,
    isclo: usize, ischi: usize, // isclo <= isc <= ischi
    ipixlo: usize, ipixhi: usize, // ipixlo <= ipix < ipixhi
    pgbar: ProgressBar
) -> f64 {
    let npix: usize = ipixhi-ipixlo;
    let nthetaf64: f64 = ntheta as f64;
    let nphif64: f64 = nphi as f64;

    let inv_1_minus_ctheta_min = 1.0 / (1.0 - CTHETA_MIN);
    // let nphi_over_2pi = nphif64/consts::PI/2.;

    let mut cosphi_lut: Vec<f64> = (0..=nphi/2).map(|v| ((v as f64)/nphif64*2.*consts::PI).cos()).collect();
    let (iphi_left, iphi_right) = ((nphi+1)/2, nphi/2);
    if cosphi_lut.len()*2 <= nphi+1 {
        cosphi_lut.push(-1.);
    }
    cosphi_lut.reverse(); // sort the array

    let coszmax = (zmax*D2R).cos();
    let coszmin = (zmin*D2R).cos();
    let sinthsk: Vec<f64> = hpxobj.decs_r[ipixlo..ipixhi].iter().map(|v| v.cos()).collect();
    let costhsk: Vec<f64> = hpxobj.decs_r[ipixlo..ipixhi].iter().map(|v| v.sin()).collect();
    let sinskphis: Vec<f64> = hpxobj.ras_r[ipixlo..ipixhi].iter().map(|v| v.sin()).collect();
    let cosskphis: Vec<f64> = hpxobj.ras_r[ipixlo..ipixhi].iter().map(|v| v.cos()).collect();

    let mut lttot: f64 = 0.;
    let mut isc = isclo;
    let mut sstop: f64 = scobj.tstop[isclo];
    let sstoplast: f64 = scobj.tstop[ischi];
    let mut sstrt: f64;

    for (gstrt, gstop, isfd64) in izip!(&gtiobj.tgstart, &gtiobj.tgstop, &gtiobj.isfd64) {
        while sstop<=*gstrt && isc<ischi {
            isc += 1;
            sstop = scobj.tstop[isc];
        }

        if *gstrt >= sstoplast {
            break;
        }

        while isc<=ischi {
            // drop the time intervals
            if scobj.in_saa[isc] || scobj.lvtime[isc]<=0. {
                //println!("{}, {}", scobj.in_saa[isc], scobj.lvtime[isc]);
                isc += 1;
                continue;
            }
            pgbar.set_position((isc-isclo).try_into().unwrap());

            // break the loop
            sstrt = scobj.tstart[isc];
            sstop = scobj.tstop[isc];
            if sstrt>= *gstop {
                break;
            }

            // prepare angles
            let sdtmid = (sstrt+sstop)/2. - TLTRF_MET_REF;
            let srax_r = scobj.ra_x[isc]*D2R;
            let sray_r = scobj.ra_y[isc]*D2R;
            let sraz_r = scobj.ra_z[isc]*D2R;
            let srazth_r = scobj.ra_zth[isc]*D2R;
            let (sinrax, cosrax) = srax_r.sin_cos();
            let (sinray, cosray) = sray_r.sin_cos();
            let (sinraz, cosraz) = sraz_r.sin_cos();
            let (sinrazth, cosrazth) = srazth_r.sin_cos();

            let sdcx_r = scobj.dec_x[isc]*D2R;
            let sdcy_r = scobj.dec_y[isc]*D2R;
            let sdcz_r = scobj.dec_z[isc]*D2R;
            let sdczth_r = scobj.dec_zth[isc]*D2R;
            let (costhx, sinthx) = sdcx_r.sin_cos();
            let (costhy, sinthy) = sdcy_r.sin_cos();
            let (costhz, sinthz) = sdcz_r.sin_cos();
            let (costhzth, sinthzth) = sdczth_r.sin_cos();

            // calc livetime
            let frtl: f64 = if sstrt< *gstrt {
                *gstrt
            } else {
                sstrt
            };
            let frth: f64 = if sstop> *gstop {
                *gstop
            } else {
                sstop
            };
            let slt = scobj.lvtime[isc]*(frth-frtl)/(sstop-sstrt);
            let slttmid = slt*sdtmid;
            lttot += slt;

            // loop the healpix pixels
            for ipix0 in 0..npix {
                let costh: f64 = costhsk[ipix0];
                let sinth: f64 = sinthsk[ipix0];
                let cosskphi: f64 = cosskphis[ipix0];
                let sinskphi: f64 = sinskphis[ipix0];

                // constrain the zenith angle
                let cos_skphi_minus_razth = cosskphi*cosrazth + sinskphi*sinrazth;
                let cossepzth = (costhzth*costh + sinthzth*sinth*cos_skphi_minus_razth).min(1.);
                if cossepzth<coszmax || cossepzth>coszmin {
                    continue;
                }

                // calc the inclination angle
                let cos_skphi_minus_raz = cosskphi*cosraz + sinskphi*sinraz;
                let cossepz = (-(costhz*costh + sinthz*sinth*cos_skphi_minus_raz)).min(1.);
                if cossepz<=CTHETA_MIN { // too large inclination angle
                    continue;
                }

                // calc index of theta bins
                let idxval: f64 = if lincth {
                    (1.-cossepz)*inv_1_minus_ctheta_min * nthetaf64
                } else {
                    ((1.-cossepz)*inv_1_minus_ctheta_min).sqrt() * nthetaf64
                };
                let itheta: usize = ntheta-1 - (idxval as usize);

                // calc index of phi bins
                let iphi: usize;
                // let iphi2: usize;
                if nphi>1 && cossepz<1. {
                    let sinsepz = (1.-cossepz*cossepz).sqrt();
                    let cos_skphi_minus_rax = cosskphi*cosrax + sinskphi*sinrax;
                    let cossepx = -(costhx*costh + sinthx*sinth*cos_skphi_minus_rax);
                    let cosphi = cossepx/sinsepz;

                    if cosphi<=-1. {
                        iphi = nphi/2;
                    } else if cosphi >= 1. {
                        iphi = 0;
                    } else {
                        let iphiraw = bisect::bisect_left(&cosphi_lut, &cosphi); // not at boundary

                        // let phi = cosphi.acos(); // may further improve by 25% if replaced with look up table
                        let cos_skphi_minus_ray = (cosskphi*cosray + sinskphi*sinray).min(1.).max(-1.);
                        if costhy*costh+sinthy*sinth*cos_skphi_minus_ray <= 0. {
                            // 0 -- 180 deg
                            // iphi2 = (phi*nphi_over_2pi) as usize;
                            iphi = iphi_left-iphiraw;
                        } else {
                            // 180 -- 360 deg
                            // iphi2 = (nphif64 - phi*nphi_over_2pi) as usize;
                            iphi = iphi_right+iphiraw-1;
                        }
                        // if iphi!=iphi2 {
                        //     println!("iphi={}, iphi2={}", iphi, iphi2);
                        // }
                    }
                } else {
                    iphi = 0;
                }

                // set the array
                //println!("{} {} {}", ipix0, itheta, iphi);
                let array_idx = [ipix0, iphi, itheta];
                lttint[array_idx] += slttmid;
                if *isfd64 {
                    ltfd64[array_idx] += slt;
                } else {
                    ltfd08[array_idx] += slt;
                    lttint8[array_idx] += slttmid;
                }
            }
            // assert!(false);

            if *gstop >= sstop {
                isc+=1;
            } else {
                break;
            }
        }
    }
    pgbar.finish_and_clear();

    lttot
}


pub fn ltcube_main(scfile: &str, evfile: &str, outfile: &str,
    order: u32, nthetabins: usize, nphibins: usize, zmin: f64, zmax: f64, nthreads: usize, lincth: bool)
    -> Result<(), Box<dyn std::error::Error>>
{
    // echo the input params
    let ncpus = num_cpus::get();
    let nthr = nthreads.max(1).min(ncpus);
    println!("[{}] scfile={} evfile={} outfile={} lincth={} order={} thetabins={} phibins={} zmin={} zmax={} nthreads={}",
            get_time_str(), scfile, evfile, outfile, lincth, order, nthetabins, nphibins, zmin, zmax, nthr);
    assert!(nthetabins>=1 && nphibins>=1);

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
        let errstr = format!("SC file ({}--{}) must cover the entire GTI ({}--{})",
                scobj.tstart[0], scobj.tstop[nsc-1], gtimin, gtimax);
        return Err(Box::new(Error::new(ErrorKind::InvalidData, errstr)));
    }

    // create data
    println!("[{}] preparing miscellaneous information ...", get_time_str());
    let hpxobj = HealpixCoordinates::init_with_order(order.try_into()?)?;
    let npix = hpxobj.len();

    let cthetabounds = if lincth {
        Array::linspace(CTHETA_MIN, 1., nthetabins+1)
    } else {
        Array::linspace(1., 0., nthetabins+1).map(|v| 1.-(1.-CTHETA_MIN)*v*v)
    }.to_vec();
    let phibounds = Array::linspace(0., 360., nphibins+1).to_vec();
    // println!("{:?}", cthetabounds);
    // println!("{:?}", phibounds);

    let mut ltfd08 = Array3::<f64>::zeros([npix, nphibins, nthetabins]);
    let mut ltfd64 = Array3::<f64>::zeros([npix, nphibins, nthetabins]);
    let mut lttint = Array3::<f64>::zeros([npix, nphibins, nthetabins]);
    let mut lttint8= Array3::<f64>::zeros([npix, nphibins, nthetabins]);

    // find the min/max index
    println!("[{}] calculating ltcube ...", get_time_str());
    let stopwatch = std::time::Instant::now();

    let idxlo = (bisect::bisect_right(&scobj.tstart, &gtimin)-1).max(0);
    let idxhi = (bisect::bisect_left(&scobj.tstart, &gtimax)+1).min(nsc-1);
    assert!(idxlo<=idxhi);

    let tmin = scobj.tstart[idxlo].max(gtimin);
    let tmax = scobj.tstop[idxhi].min(gtimax);
    // debug
    //println!("idxlo={}, idxhi={}", idxlo, idxhi);
    //println!("gtitmin={}, gtitmax={}", gtimin, gtimax);
    //println!("sctmin={}, sctmax={}", scobj.tstart[idxlo], scobj.tstop[idxhi]);
    //println!("tmin={}, tmax={}", tmin, tmax);

    // run worker
    let lttot: f64;
    let chunk_size = npix/nthr;
    let mut ltfd08iter = ltfd08.axis_chunks_iter_mut(Axis(0), chunk_size);
    let mut ltfd64iter = ltfd64.axis_chunks_iter_mut(Axis(0), chunk_size);
    let mut lttintiter = lttint.axis_chunks_iter_mut(Axis(0), chunk_size);
    let mut lttint8iter= lttint8.axis_chunks_iter_mut(Axis(0), chunk_size);

    if nthr == 1 {
        let mut ltfd08p = ltfd08iter.next().unwrap();
        let mut ltfd64p = ltfd64iter.next().unwrap();
        let mut lttintp = lttintiter.next().unwrap();
        let mut lttint8p= lttint8iter.next().unwrap();

        let pb = ProgressBar::new((idxhi-idxlo+1).try_into().unwrap());
        pb.set_style(ProgressStyle::with_template("[{elapsed_precise}] {bar:50}  {percent:>3}%  ETA: {eta}").unwrap());

        lttot = ltcube_worker(&mut ltfd08p, &mut ltfd64p,
            &mut lttintp, &mut lttint8p,
            &scobj, &gtiobj, &hpxobj, zmin, zmax, nthetabins, nphibins, lincth,
            idxlo, idxhi, 0, npix, pb);
    } else {
        let mut lttots = Array1::<f64>::zeros(nthr);
        let mut lttotsiter = lttots.axis_chunks_iter_mut(Axis(0), 1);

        thread::scope(|scope| {
            let mpb = MultiProgress::new();

            for ith in 0..nthr {
                let mut ltfd08p = ltfd08iter.next().unwrap();
                let mut ltfd64p = ltfd64iter.next().unwrap();
                let mut lttintp = lttintiter.next().unwrap();
                let mut lttint8p= lttint8iter.next().unwrap();
                let mut lttotsp = lttotsiter.next().unwrap();

                let ipixlop = ith*chunk_size;
                let ipixhip = std::cmp::min(npix, ipixlop+chunk_size);

                let scobjref = &scobj;
                let gtiobjref = &gtiobj;
                let hpxobjref = &hpxobj;

                let pb = ProgressBar::new((idxhi-idxlo+1).try_into().unwrap());
                let npb = mpb.add(pb);
                npb.set_style(ProgressStyle::with_template("[{elapsed_precise}] {bar:50}  {percent:>3}%  ETA={eta}").unwrap());

                scope.spawn(move || {
                    lttotsp[0] = ltcube_worker(&mut ltfd08p, &mut ltfd64p,
                        &mut lttintp, &mut lttint8p,
                        scobjref, gtiobjref, hpxobjref, zmin, zmax, nthetabins, nphibins, lincth,
                        idxlo, idxhi, ipixlop, ipixhip, npb);
                });
            }
        });
        lttot = lttots[0];
    }

    let dtime = stopwatch.elapsed();
    println!("[{}] >>> runtime = {:?} x {}, livetime = {} sec <<<", get_time_str(), dtime, nthr, lttot);

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
    hdu0.write_key(&mut fptr, "HISTORY", format!("DT = {} [sec] x {}", dtime.as_secs_f32(), nthr))?;

    // ext1
    let nvec: usize = nthetabins*nphibins;
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
    hdu1.write_key(&mut fptr, "ZMIN", (zmin, "deg, minimum zenith angle"))?;
    hdu1.write_key(&mut fptr, "ZMAX", (zmax, "deg, maximum zenith angle"))?;
    hdu1.write_key(&mut fptr, "COORDSYS", "CEL")?;
    hdu1.write_key(&mut fptr, "RADECSYS", ("FK5", "J2000"))?;
    hdu1.write_key(&mut fptr, "PIXTYPE", "HEALPIX")?;
    hdu1.write_key(&mut fptr, "INDXSCHM", "IMPLICIT")?;
    hdu1.write_key(&mut fptr, "ORDERING", "NESTED")?;
    hdu1.write_key(&mut fptr, "ORDER", hpxobj.order)?;
    hdu1.write_key(&mut fptr, "NSIDE", hpxobj.nside)?;
    hdu1.write_key(&mut fptr, "FIRSTPIX", 0)?;
    hdu1.write_key(&mut fptr, "LASTPIX", hpxobj.npix-1)?;
    hdu1.write_key(&mut fptr, "TIMESYS", "TT")?;
    hdu1.write_key(&mut fptr, "TIMEUNIT", "s")?;
    hdu1.write_key(&mut fptr, "LIVETIME", (lttot, "sec"))?;

    let treftai = Epoch::from_gregorian_utc_hms(2013, 1, 1, 0, 0, 0).to_time_scale(hifitime::TimeScale::TAI);
    let tlttai = treftai + hifitime::Duration::from_seconds(TLTRF_MET_REF);
    let tmintai = treftai + hifitime::Duration::from_seconds(tmin);
    let tmaxtai = treftai + hifitime::Duration::from_seconds(tmax);
    hdu1.write_key(&mut fptr, "TLTRFMET", (TLTRF_MET_REF, tlttai.to_gregorian_str(hifitime::TimeScale::UTC)))?;
    hdu1.write_key(&mut fptr, "TSTART", (tmin, tmintai.to_gregorian_str(hifitime::TimeScale::UTC)))?;
    hdu1.write_key(&mut fptr, "TSTOP", (tmax, tmaxtai.to_gregorian_str(hifitime::TimeScale::UTC)))?;
    hdu1.write_key(&mut fptr, "CREATOR", (who.clone(), timestr.clone()))?;

    // ext2
    let col21 = ColumnDescription::new("CTHETA_MIN").with_type(ColumnDataType::Float).create()?;
    let col22 = ColumnDescription::new("CTHETA_MAX").with_type(ColumnDataType::Float).create()?;
    let des2 = [col21, col22];
    let hdu2 = fptr.create_table("CTHETABOUNDS".to_string(), &des2)?;

    hdu2.write_col(&mut fptr, "CTHETA_MIN", &cthetabounds[0..nthetabins])?;
    hdu2.write_col(&mut fptr, "CTHETA_MAX", &cthetabounds[1..nthetabins+1])?;

    hdu2.write_key(&mut fptr, "TELESCOP", "DAMPE")?;
    hdu2.write_key(&mut fptr, "CTHBINAL", if lincth {("LIN", "linear")} else {("SQ", "squared")})?;
    hdu2.write_key(&mut fptr, "CTHMIN", CTHETA_MIN)?;
    hdu2.write_key(&mut fptr, "NCTHBIN", nthetabins as u32)?;
    hdu2.write_key(&mut fptr, "CREATOR", (who.clone(), timestr.clone()))?;

    // ext3
    let col31 = ColumnDescription::new("PHI_MIN").with_type(ColumnDataType::Float).create()?;
    let col32 = ColumnDescription::new("PHI_MAX").with_type(ColumnDataType::Float).create()?;
    let des3 = [col31, col32];
    let hdu3 = fptr.create_table("PHIBOUNDS".to_string(), &des3)?;

    hdu3.write_col(&mut fptr, "PHI_MIN", &phibounds[0..nphibins])?;
    hdu3.write_col(&mut fptr, "PHI_MAX", &phibounds[1..nphibins+1])?;

    hdu3.write_key(&mut fptr, "TELESCOP", "DAMPE")?;
    hdu3.write_key(&mut fptr, "SCAXIS", "-X-Y-Z")?;
    hdu3.write_key(&mut fptr, "NPHIBIN", nphibins as u32)?;
    hdu3.write_key(&mut fptr, "PHIBINAL", ("LIN", "linear"))?;
    hdu3.write_key(&mut fptr, "CREATOR", (who.clone(), timestr.clone()))?;

    println!("[{}] done", get_time_str());
    Ok(())
}


// ////////////////////////
// // PyO3 Python Interface
// ////////////////////////
// use pyo3::prelude::*;
// use pyo3::exceptions::PyValueError;
// use pyo3::types::PyModule;

// /// Calculate livetime cube for DAMPE data
// #[pyfunction]
// #[pyo3(signature = (scfile, evfile, outfile, order, nthetabins, nphibins, zmin, zmax, nthreads, lincth))]
// fn ltcube(
//     scfile: String,
//     evfile: String, 
//     outfile: String,
//     order: u32,
//     nthetabins: usize,
//     nphibins: usize,
//     zmin: f64,
//     zmax: f64,
//     nthreads: usize,
//     lincth: bool,
// ) -> PyResult<()> {
//     // 参数验证
//     if order > 10 {
//         return Err(PyValueError::new_err("HEALPix order must be <= 10"));
//     }
//     if nthetabins == 0 || nphibins == 0 {
//         return Err(PyValueError::new_err("Number of bins must be > 0"));
//     }
//     if zmin < 0.0 || zmax <= zmin || zmax > 180.0 {
//         return Err(PyValueError::new_err("Zenith angles must satisfy 0 <= zmin < zmax <= 180"));
//     }
    
//     ltcube_main(&scfile, &evfile, &outfile, order, nthetabins, nphibins, zmin, zmax, nthreads, lincth)
//         .map_err(|e| PyValueError::new_err(format!("Livetime cube calculation failed: {}", e)))
// }

// /// DAMPE livetime cube calculation toolkit
// #[pymodule]
// fn dpltcubers(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(ltcube, m)?)?;
    
//     // 添加版本信息
//     m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
//     // 添加常量
//     m.add("R2D", R2D)?;
//     m.add("D2R", D2R)?;
//     m.add("CTHETA_MIN", CTHETA_MIN)?;
    
//     Ok(())
// }