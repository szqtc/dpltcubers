// main.rs
use fitsio::FitsFile;
use ndarray::Array3;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::f64::consts::PI;

const R2D: f64 = 180.0 / PI;
const D2R: f64 = PI / 180.0;
const CTHETA_MIN: f64 = 0.5;
const MET_REF: f64 = 1577836800.0; // 2016-01-01 in seconds

struct SpacecraftData {
    starts: Vec<f64>,
    stops: Vec<f64>,
    ra_x: Vec<f64>,
    dec_x: Vec<f64>,
    // 其他字段...
}

struct GTIData {
    starts: Vec<f64>,
    stops: Vec<f64>,
    fds: Vec<i32>,
}

fn read_sc_file(path: &str) -> SpacecraftData {
    let mut f = FitsFile::open(path).unwrap();
    let hdu = f.hdu(1).unwrap();
    // 读取SC_DATA数据...
    unimplemented!()
}

fn read_gti_file(path: &str) -> GTIData {
    let mut f = FitsFile::open(path).unwrap();
    let hdu = f.hdu("GTI").unwrap();
    // 读取GTI数据...
    unimplemented!()
}

fn healpix_pixel_angles(nside: u32, order: &str) -> (Vec<f64>, Vec<f64>) {
    // 实现HEALPix坐标计算...
    unimplemented!()
}

fn calculate_ltcube(
    sc_data: &SpacecraftData,
    gti_data: &GTIData,
    ntheta: usize,
    nphi: usize,
    zmax: f64,
    nthreads: usize,
) -> (Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>, f64) {
    let npix = healpix_npix(6); // HEALPix nside=64 (order=6)
    let (skthetas_r, skphis_r) = healpix_pixel_angles(6, "NESTED");
    let cos_zmax = (zmax * D2R).cos();

    // 初始化共享数据结构
    let lt_fd08 = Arc::new(Mutex::new(Array3::zeros((npix, nphi, ntheta))));
    let lt_fd64 = Arc::new(Mutex::new(Array3::zeros((npix, nphi, ntheta))));
    let lttint = Arc::new(Mutex::new(Array3::zeros((npix, nphi, ntheta))));
    let lttint08 = Arc::new(Mutex::new(Array3::zeros((npix, nphi, ntheta))));
    let total_lt = Arc::new(Mutex::new(0.0));

    // 对SC数据按时间分块处理
    sc_data.starts.par_chunks(nthreads).enumerate().for_each(
        |(chunk_idx, chunk)| {
            let mut local_lt08 = Array3::zeros((npix, nphi, ntheta));
            let mut local_lt64 = Array3::zeros((npix, nphi, ntheta));
            let mut local_tint = Array3::zeros((npix, nphi, ntheta));
            let mut local_tint08 = Array3::zeros((npix, nphi, ntheta));
            let mut local_total_lt = 0.0;

            for j in chunk {
                if sc_data.in_saa[*j] || sc_data.livetime[*j] <= 0.0 {
                    continue;
                }

                // 计算时间重叠部分
                let (sstrt, sstop) = (sc_data.starts[*j], sc_data.stops[*j]);
                let slt = calculate_overlap(sstrt, sstop, &gti_data);

                // 坐标转换
                let sraz_r = sc_data.ra_z[*j] * D2R;
                let sdcz_r = sc_data.dec_z[*j] * D2R;
                let sinthz = sdcz_r.sin();
                let costhz = sdcz_r.cos();

                // 遍历每个HEALPix像素
                for ipix in 0..npix {
                    let theta_r = skthetas_r[ipix];
                    let phi_r = skphis_r[ipix];
                    let costh = theta_r.cos();
                    let sinth = theta_r.sin();

                    // 计算cossepz
                    let cossepz = -(costhz * costh + sinthz * sinth * (phi_r - sraz_r).cos());
                    if cossepz <= CTHETA_MIN {
                        continue;
                    }

                    // 计算itheta和iphi
                    let itheta = if linctheta {
                        ntheta - 1 - ((1.0 - cossepz) / (1.0 - CTHETA_MIN) * ntheta as f64) as usize
                    } else {
                        ntheta - 1 - ((1.0 - cossepz).sqrt() / (1.0 - CTHETA_MIN).sqrt() * ntheta as f64) as usize
                    };

                    let iphi = if nphi > 1 && cossepz < 1.0 {
                        // 计算phi分箱逻辑
                        unimplemented!() // 需补充完整phi计算
                    } else {
                        0
                    };

                    // 累加活时间
                    if gti_data.fds[chunk_idx] == 8 {
                        local_lt08[[ipix, iphi, itheta]] += slt;
                        local_tint08[[ipix, iphi, itheta]] += slt * (sstrt + sstop)/2.0;
                    } else if gti_data.fds[chunk_idx] == 64 {
                        local_lt64[[ipix, iphi, itheta]] += slt;
                    }
                    local_tint[[ipix, iphi, itheta]] += slt * (sstrt + sstop)/2.0;
                }

                local_total_lt += slt;
            }

            // 合并到共享数据结构
            let mut gt_fd08 = lt_fd08.lock().unwrap();
            let mut gt_fd64 = lt_fd64.lock().unwrap();
            let mut gt_tint = lttint.lock().unwrap();
            let mut gt_tint08 = lttint08.lock().unwrap();
            let mut t_lt = total_lt.lock().unwrap();

            *gt_fd08 += &local_lt08;
            *gt_fd64 += &local_lt64;
            *gt_tint += &local_tint;
            *gt_tint08 += &local_tint08;
            *t_lt += local_total_lt;
        },
    );

    // 解包Arc<Mutex>返回结果
    let lt08 = Arc::try_unwrap(lt_fd08).unwrap().into_inner().unwrap();
    let lt64 = Arc::try_unwrap(lt_fd64).unwrap().into_inner().unwrap();
    let tint = Arc::try_unwrap(lttint).unwrap().into_inner().unwrap();
    let tint08 = Arc::try_unwrap(lttint08).unwrap().into_inner().unwrap();
    let total = *total_lt.lock().unwrap();

    (lt08, lt64, tint, tint08, total)
}

// 辅助函数：计算时间重叠
fn calculate_overlap(sstrt: f64, sstop: f64, gti_data: &GTIData) -> f64 {
    gti_data.starts.iter().zip(gti_data.stops.iter())
        .filter(|(gstrt, gstop)| sstop > *gstrt && sstrt < *gstop)
        .map(|(gstrt, gstop)| {
            let overlap_start = sstrt.max(*gstrt);
            let overlap_end = sstop.min(*gstop);
            (overlap_end - overlap_start).max(0.0)
        })
        .sum()
}

fn write_output(
    outfile: &str,
    lt_fd08: &Array3<f64>,
    lt_fd64: &Array3<f64>,
    total_lt: f64,
) {
    // 使用fitsio库写入FITS文件...
}

fn main() {
    let sc_data = read_sc_file("input.sc.fits");
    let gti_data = read_gti_file("input.evt.fits");
    
    let (lt08, lt64, total_lt) = calculate_ltcube(
        &sc_data,
        &gti_data,
        20, 1, 100.0, 4
    );
    
    write_output("output.fits", &lt08, &lt64, total_lt);
}