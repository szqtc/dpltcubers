////////////////////////
// tests
////////////////////////
#[cfg(test)]
mod tests {
use dpltcubers::*;

#[test]
fn test_readfiles() {
    ///////// scfile
    println!("Spacecraft:");
    let sp = Spacecraft::read_file("tests/testfiles/test_sc.fits").unwrap();
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
    println!();

    ///////// evfile
    println!("GoodTimeIntervals:");
    let gti = GoodTimeIntervals::read_file("tests/testfiles/test_ev.fits").unwrap();
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

    let sindecs: Vec<_> = hpxcrd.decs_r.iter().map(|v| v.sin()).collect();
    let cosdecs: Vec<_> = hpxcrd.decs_r.iter().map(|v| v.cos()).collect();
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
    // see https://github.com/nyx-space/hifitime/issues/417
    use hifitime::{Epoch, Unit};
    let utc0 = Epoch::from_gregorian_utc_hms(2016, 12, 31, 23, 59, 0);
    let tai0 = utc0.to_time_scale(hifitime::TimeScale::TAI);

    // gh_417
    for dt in 1..62 {
        let tai_next = tai0 + Unit::Second * dt;
        let utc_next = utc0 + Unit::Second * dt;
        println!("+{dt} s => TAI = {tai_next}\tTAI as UTC = {tai_next:?}\tUTC = {utc_next}");
        // test if the leap second is set at UTC 00:00:00
        if dt<61 {
            assert_eq!(format!("{tai_next:?}"), format!("{utc_next}")); // it will fail when hifitime is < 4.2.0
        }
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
    assert_eq!(utc5.to_gregorian_str(hifitime::TimeScale::UTC), "2016-12-31T23:59:59 UTC"); // hifitime will leap back to 59th sec
    assert_eq!(utc5.to_gregorian_str(hifitime::TimeScale::TAI), "2017-01-01T00:00:35 TAI");

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
    assert_eq!(dt64_utc, 1.); // no leap seconds in UTC for hifitime

    let dt54_tai = utc5.to_tai_seconds()-utc4.to_tai_seconds();
    let dt54_utc = utc5.to_utc_seconds()-utc4.to_utc_seconds();
    println!("T6-T5 = {dt54_tai} (TAI) = {dt54_utc} (UTC)");
    assert_eq!(dt54_tai, 0.);
    assert_eq!(dt54_utc, 0.);

    let dt63_tai = utc6.to_tai_seconds()-utc3.to_tai_seconds();
    let dt63_utc = utc6.to_utc_seconds()-utc3.to_utc_seconds();
    println!("T6-T3 = {dt63_tai} (TAI) = {dt63_utc} (UTC)");
    assert_eq!(dt63_tai, 61.);
    assert_eq!(dt63_utc, 60.); // no leap seconds in UTC for hifitime

    // more cases
    let utc7a = Epoch::from_gregorian_utc_hms(2017, 12, 31, 23, 59, 59);
    let utc7b = Epoch::from_gregorian_utc_hms(2018, 1, 1, 0, 0, 0);
    assert_eq!(utc7b.to_tai_seconds()-utc7a.to_tai_seconds(), 1.);
    assert_eq!(utc7b.to_utc_seconds()-utc7a.to_utc_seconds(), 1.);

    let utc8a = Epoch::from_gregorian_utc_hms(2015, 6, 30, 23, 59, 59);
    let utc8b = Epoch::from_gregorian_utc_hms(2015, 7, 1, 0, 0, 0);
    assert_eq!(utc8b.to_tai_seconds()-utc8a.to_tai_seconds(), 2.);
    assert_eq!(utc8b.to_utc_seconds()-utc8a.to_utc_seconds(), 1.); // no leap seconds in UTC for hifitime

    let utc9a = Epoch::from_gregorian_utc_hms(1972, 1, 1, 0, 0, 0);
    let utc9b = Epoch::from_gregorian_utc_hms(2025, 1, 1, 0, 0, 0);
    assert_eq!(utc9b.to_tai_seconds()-utc9a.to_tai_seconds(), 1672617627.);
    assert_eq!(utc9b.to_utc_seconds()-utc9a.to_utc_seconds(), 1672617600.); // no leap seconds in UTC for hifitime
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
