#!/usr/bin/env python
'''
a simple tool to calculate the livetime cube of DAMPE

NOTE:
+ may be problematic for fine ltcube with coarse spacecraft
  records, the recommendation is given:
    - 30s sc: nside<=32  or order<=5
    - 10s sc: nside<=64  or order<=6
    -  5s sc: nside<=128 or order<=7
    -  1s sc: nside<=512 or order<=9
+ KNOWN BUG:
  1. [Windows/MacOS]
     the 'spawn' method used in the multiprocessing will
     cause global variables to be unavailable in subprocesses.

Authors:
    DUAN Kai-Kai (duankk[AT]pmo.ac.cn),
    SHEN Zhao-Qiang (zqshen[AT]pmo.ac.cn)
References:
    1. Duan K-K et al. RAA 19(9),132 [arXiv:1904.13098]
    2. Shen Z-Q et al. PoS(ICRC2023)670

\033[3mTHIS CODE SHARES THE SAME LICENSE AS DMPST\033[2m
=============================================================
Date         Author        Description
-------------------------------------------------------------
****-**-**   DUAN K-K      original ltcube_phi from DmpST
2019-05-31   SHEN Z-Q      1. optimize with numba.jit
                           2. enable multi-processing
2019-12-03   SHEN Z-Q      enable linear costheta binning
2019-12-26   SHEN Z-Q      implement the ltcube_sum
2020-04-22   SHEN Z-Q      solve the bad message length error
2020-06-19   SHEN Z-Q      change the definition of phi
2022-09-21   SHEN Z-Q      share ltcubes to solve pickle err
2023-02-23   SHEN Z-Q      add \int t dt map into the fits
2023-11-27   SHEN Z-Q      add \int t dt map for fd=08
=============================================================\033[0m
'''
from __future__ import print_function
import time
import bisect
import warnings
import multiprocessing as mp
import numpy as np
import healpy as hp
from multiprocessing import Pool, Value, Array
from numba import jit
from astropy import time as atime
from astropy.io import fits

try:
    mp.set_start_method('fork')
except AttributeError:
    pass

__version__ = '0.992a'

r2d = 180./np.pi
d2r = np.pi/180.

ctheta_min = 0.5
utcref = atime.Time('2013-01-01', scale='utc')
metref = (atime.Time('2016-01-01', scale='utc')-utcref).sec


def _tprint(*a, **k):
    "print a string along with a time tag"
    tstr = time.strftime('[%Y-%m-%dT%H:%M:%S]', time.localtime())
    print(tstr, *a, **k)


@jit(nopython=True)#, parallel=True)
def ltcubephi_calculator(
    ntheta, nphi, zmax_d, linctheta,
    skthetas_r, skphis_r,
    gstrts, gstops, gfds,
    sstrts, sstops,
    sraxs_d, sdcxs_d, srays_d, sdcys_d, srazs_d, sdczs_d,
    srazths_d, sdczths_d,
    sinsaas, slts
    ):
    """
    the core code of the ltcube calculator, aggressive optimization
    is included

    parameters
    ----------
    ntheta, nphi: integer
        the number of theta and phi bins
    zmax_d: double
        the maximum zenith angle to include the livetime in [deg]
    linctheta: bool
        whether the cos(theta) is linearly binned or not
    skthetas_r, skphis_r: array of double, (npix, )
        the sky coordinates of each pixel in [rad], where
        ra_r = phi_r, decl_r = pi/2.-theta
    gstrts, gstops: array of double, (ngti, )
        the starting and stoping time of each gti record
    gfds: array of integer, (ngti, )
        the fd of each gti record
    sstrts, sstops: array of double, (nsc, )
        the starting and stoping time of each sc record
    sraxs_d, sdcxs_d, srays_d, sdcys_d,
    srazs_d, sdczs_d: array of double, (nsc, )
        the coordinates of X, Y, Z axis for each sc record in [deg]
    srazths_d, sdczths_d: array of double, (nsc, )
        the coordinates of zenith in [deg]
    sinsaas: array of bool, (nsc, )
        a flag indicating whether sc is in saa
    slts: array of double, (nsc, )
        the live time of each sc record

    returns
    -------
    ltfd08, ltfd64: array of double, (npix, ntheta, nphi)
        the live time for fd = 8 and 64
    lttint, lttint08: array of double, (npix, ntheta, nphi)
        \int t dt for fd=8+64 and 8
    lttot: float
        the total livetime in sec
    """
    npix = skphis_r.shape[0]
    ltfd08 = np.zeros((npix, nphi, ntheta))
    ltfd64 = np.zeros((npix, nphi, ntheta))
    lttint = np.zeros((npix, nphi, ntheta)) # 1st order correction of efficiency variance
    lttint08 = np.zeros((npix, nphi, ntheta)) # 1st order correction of efficiency variance

    coszmax = np.cos(zmax_d*d2r)
    sinthsk, costhsk = np.sin(skthetas_r), np.cos(skthetas_r)

    j = 0
    lttot = 0.
    nsc = sstrts.shape[0]
    sstop = sstops[0]
    sstoplast = sstops[-1]
    for gstrt, gstop, gfd in zip(gstrts, gstops, gfds):
        # calculate ltcube
        while (sstop<=gstrt and j+1<nsc):
            j += 1
            sstop = sstops[j]

        if gstrt>=sstoplast:
            break

        #print('gti:', gstrt, gstop)
        while (j<nsc):
            if sinsaas[j] or slts[j]<=0.:
                j += 1
                continue

            sstrt, sstop = sstrts[j], sstops[j]
            if sstrt>=gstop:
                break

            #print('sc:', sstrt, sstop)
            sdtmid = (sstrt+sstop)/2.-metref
            srax_r, sdcx_r = sraxs_d[j]*d2r, sdcxs_d[j]*d2r
            sray_r, sdcy_r = srays_d[j]*d2r, sdcys_d[j]*d2r
            sraz_r, sdcz_r = srazs_d[j]*d2r, sdczs_d[j]*d2r
            srazth_r, sdczth_r = srazths_d[j]*d2r, sdczths_d[j]*d2r

            sinthx, costhx = np.cos(sdcx_r), np.sin(sdcx_r)
            sinthy, costhy = np.cos(sdcy_r), np.sin(sdcy_r)
            sinthz, costhz = np.cos(sdcz_r), np.sin(sdcz_r)
            sinthzth, costhzth = np.cos(sdczth_r), np.sin(sdczth_r)

            if (sstrt<gstrt) or (sstop>gstop):
                frt0 = max(sstrt, gstrt)
                frt1 = min(sstop, gstop)
                slt = (frt1-frt0)/(sstop-sstrt) * slts[j]
            else:
                slt = slts[j]
            lttot += slt
            sltmid = slt*sdtmid
            #print('lt:', slt)

            for ipix in range(npix):
                phi_r = skphis_r[ipix]
                costh, sinth = costhsk[ipix], sinthsk[ipix]

                cossepzth = costhzth*costh+sinthzth*sinth*np.cos(phi_r-srazth_r)
                if cossepzth<coszmax:
                    continue

                cossepz = min(-(costhz*costh+sinthz*sinth*np.cos(phi_r-sraz_r)), 1.) # inj. ang. wrt. +z
                #cossepz = min(-(np.sin(sdcz_r+skthetas_r[ipix])-2.*sinthz*sinth*(np.sin((phi_r-sraz_r)/2.))**2), 1.)
                if cossepz<=ctheta_min:
                    continue
                if linctheta:
                    itheta = ntheta-1-int((1.-cossepz)/(1.-ctheta_min)*ntheta)
                else:
                    itheta = ntheta-1-int(np.sqrt((1.-cossepz)/(1.-ctheta_min))*ntheta)

                if (nphi>1) and (cossepz<1.):
                    sinsepz = np.sqrt(1.-cossepz**2)
                    cossepx = -(costhx*costh+sinthx*sinth*np.cos(phi_r-srax_r)) # inj. ang. wrt. +x
                    cosphi = cossepx/sinsepz
                    if cosphi <= -1.:
                        iphi = int(nphi/2.)
                    elif cosphi >= 1.:
                        iphi = 0
                    else:
                        phi = np.arccos(cosphi)
                        if costhy*costh+sinthy*sinth*np.cos(phi_r-sray_r)<=0: # ctheta_y >= 0
                            iphi = int(phi/np.pi*nphi/2)
                        else:
                            iphi = int(nphi-phi/np.pi*nphi/2)
                else:
                    iphi = 0

                #print(ipix, gfd, itheta, iphi)
                lttint[ipix, iphi, itheta] += sltmid
                if gfd == 8:
                    ltfd08[ipix, iphi, itheta] += slt
                    lttint08[ipix, iphi, itheta] += sltmid
                elif gfd == 64:
                    ltfd64[ipix, iphi, itheta] += slt
                else:
                    assert 0

            if gstop >= sstop:
                j += 1
            else:
                break

    return ltfd08, ltfd64, lttint, lttint08, lttot


def ltcubephi_calculator_worker(
    idx0, idx1,
    ntheta, nphi, zmax_d, linctheta,
    skthetas_r, skphis_r,
    #mplttot, mpltdata_FD08, mpltdata_FD64, mpltdata_tint, mpltdata_tint08
    ):
    # calculate
    ltfd08, ltfd64, lttint, lttint08, lttot = ltcubephi_calculator(
            ntheta, nphi, zmax_d, linctheta,
            skthetas_r, skphis_r,
            gtistart, gtistop, gtiFD,
            START[idx0:idx1], STOP[idx0:idx1],
            RA_X[idx0:idx1], DEC_X[idx0:idx1], RA_Y[idx0:idx1], DEC_Y[idx0:idx1], RA_Z[idx0:idx1], DEC_Z[idx0:idx1],
            RA_ZENITH[idx0:idx1], DEC_ZENITH[idx0:idx1], IN_SAA[idx0:idx1], LIVETIME[idx0:idx1])

    # combine the data
    with mplttot.get_lock():
        mplttot.value += lttot
    with mpltdata_FD08.get_lock():
        mpltdata_FD08[:] += ltfd08.ravel()
    with mpltdata_FD64.get_lock():
        mpltdata_FD64[:] += ltfd64.ravel()
    with mpltdata_tint.get_lock():
        mpltdata_tint[:] += lttint.ravel()
    with mpltdata_tint08.get_lock():
        mpltdata_tint08[:] += lttint08.ravel()


def ltcube_phi(
    scfile, evfile, outfile,
    order=6, ordering='NESTED',
    thetabins=20, phibins=1, zmax=100,
    nthreads=1, linctheta=False
    ):
    """
    main function of ltcube

    parameters
    ----------
    scfile: str
        the file name of sc data
    evfile: str
        the file name of events
    outfile: str
        the name of output ltcube
    order: non-negative integer, default=6
        the order of HealPix map
    ordering: str, 'NESTED'|'RING', default='NESTED'
        the ordering scheme of pixel
    thetabins: positive integer, default=20
        the number of theta bins
    phibins: positive integer, default=1
        the number of phi bins
    zmax: double, [0., 180.), default=100.
        the zenith angle cut in [deg]
    nthreads: positive integer, default=1
        the number of threads, >1 for multi-threads
        calculations
    linctheta: bool, default=False
        whether the theta is linearly binned or not
    """
    # read the sc file
    _tprint('loading sc file ...')
    global START, STOP, RA_X, DEC_X, RA_Y, DEC_Y, RA_Z, DEC_Z, RA_ZENITH, DEC_ZENITH, IN_SAA, LIVETIME
    hdulist = fits.open(scfile)
    scdata = hdulist['SC_DATA'].data
    START = scdata.field('START').astype('float')
    STOP = scdata.field('STOP').astype('float')
    RA_X = scdata.field('RA_SCX').astype('float')
    DEC_X = scdata.field('DEC_SCX').astype('float')
    RA_Y = scdata.field('RA_SCY').astype('float')
    DEC_Y = scdata.field('DEC_SCY').astype('float')
    RA_Z = scdata.field('RA_SCZ').astype('float')
    DEC_Z = scdata.field('DEC_SCZ').astype('float')
    RA_ZENITH = scdata.field('RA_ZENITH').astype('float')
    DEC_ZENITH = scdata.field('DEC_ZENITH').astype('float')
    IN_SAA = scdata.field('IN_SAA').astype('bool')
    LIVETIME = scdata.field('LIVETIME').astype('float')
    del scdata
    hdulist.close()

    # read the gti file
    _tprint('loading ev file ...')
    global gtistart, gtistop, gtiFD
    hdulist = fits.open(evfile)
    gtidata = hdulist['GTI'].data
    gtistart = gtidata.field('START').astype('float')
    gtistop = gtidata.field('STOP').astype('float')
    gtiFD = gtidata.field('FD').astype('int')
    del gtidata
    hdulist.close()

    if (not START.size) or (not STOP.size) or (gtistart[0]<START[0]) or (gtistop[-1]>STOP[-1]):
        raise RuntimeError(
        'SC file ({}--{}) must cover the entire GTI ({}--{})'.format(START[0], STOP[-1], gtistart[0], gtistop[-1])
    )

    # calculate the ltmap coordinates
    _tprint('preparing miscellaneous information ...')
    nside = 2**order
    npix = 12*nside**2
    sktheta, skphi = hp.pix2ang(nside, np.arange(npix), nest=(ordering=='NESTED'))

    skra_d = skphi*r2d
    skdc_d = 90.-sktheta*r2d

    # prepare misc data
    if linctheta:
        cthetabounds = np.linspace(ctheta_min, 1., thetabins+1)
    else:
        cthetabounds = 1.-(1.-ctheta_min)*np.linspace(1., 0., thetabins+1)**2
    phibounds = np.linspace(0., 360., phibins+1)

    # accumulate the lt
    t0 = time.time()
    # sc data is selected between [idx0,idx1)
    idx0 = bisect.bisect_right(START, gtistart[0])-1
    idx1 = bisect.bisect_left(STOP, gtistop[-1])+1
    assert (nthreads>0) and (idx1>idx0)
    if (idx1-idx0)<nthreads:
        warnings.warn('nthreads ({}) is larger than the # of valid records ({}) in the scfile, set nthreads=1'.format(nthreads, idx1-idx0))
        nthreads = 1
    #assert (idx1-idx0)>nthreads
    if nthreads > 1:
        _tprint('creating the shared data ...')
        global mplttot, mpltdata_FD08, mpltdata_FD64, mpltdata_tint, mpltdata_tint08
        mplttot = Value('d', 0.)
        mpltdata_FD08 = Array('d', npix*phibins*thetabins)
        mpltdata_FD64 = Array('d', npix*phibins*thetabins)
        mpltdata_tint = Array('d', npix*phibins*thetabins)
        mpltdata_tint08 = Array('d', npix*phibins*thetabins)

        _tprint('calculating ltcube with {} threads (Ctrl+\\ to break) ...'.format(nthreads))
        mypool = Pool(nthreads)
        results = []

        nterm_per_thread = int((idx1-idx0+0.)/nthreads)+1
        idx0_, idx1_ = idx0, idx0+nterm_per_thread
        for _ in range(nthreads):
            if idx0_ >= idx1: break # deal with the corner case
            res = mypool.apply_async(ltcubephi_calculator_worker, [
                idx0_, idx1_, thetabins, phibins, zmax, linctheta, sktheta, skphi,
                #mplttot, mpltdata_FD08, mpltdata_FD64, mpltdata_tint, mpltdata_tint08
            ])
            results.append(res)
            idx0_ += nterm_per_thread
            idx1_ += nterm_per_thread

        mypool.close()
        mypool.join()
        for r in results:
            r.get() # get errors

        ltdata_lttt = mplttot.value
        ltdata_FD08 = np.frombuffer(mpltdata_FD08.get_obj()).reshape((npix, phibins, thetabins))
        ltdata_FD64 = np.frombuffer(mpltdata_FD64.get_obj()).reshape((npix, phibins, thetabins))
        ltdata_tint = np.frombuffer(mpltdata_tint.get_obj()).reshape((npix, phibins, thetabins))
        ltdata_tint08 = np.frombuffer(mpltdata_tint08.get_obj()).reshape((npix, phibins, thetabins))
    else:
        nthreads = 1 # in case of invalid input
        _tprint('calculating ltcube ...')
        ltdata_FD08, ltdata_FD64, ltdata_tint, ltdata_tint08, ltdata_lttt = ltcubephi_calculator(
                thetabins, phibins, zmax, linctheta,
                sktheta, skphi,
                gtistart, gtistop, gtiFD,
                START[idx0:idx1], STOP[idx0:idx1],
                RA_X[idx0:idx1], DEC_X[idx0:idx1],
                RA_Y[idx0:idx1], DEC_Y[idx0:idx1],
                RA_Z[idx0:idx1], DEC_Z[idx0:idx1],
                RA_ZENITH[idx0:idx1], DEC_ZENITH[idx0:idx1],
                IN_SAA[idx0:idx1], LIVETIME[idx0:idx1]
        )

    dt = time.time()-t0
    _tprint('DT = {:.3f}x{}[s]'.format(dt, nthreads))
    _tprint('total livetime = {:.2f}[s]'.format(ltdata_lttt))
    #assert 0

    # prepare each hdu
    _tprint('preparing save data ...')
    hdu0 = fits.PrimaryHDU([])
    hdu0.header['TELESCOP']= 'DAMPE'
    hdu0.header['PHFILE'] = evfile
    hdu0.header['SCFILE'] = scfile
    hdu0.header['HISTORY'] = 'DPLT ver.{}'.format(__version__)
    hdu0.header['HISTORY'] = 'DT: {:3f}x{}[s]'.format(dt, nthreads) # used for test benchmark

    col1 = fits.Column(name='LL', format='{:d}E'.format(thetabins*phibins), array=ltdata_FD08, unit='s')
    col2 = fits.Column(name='HL', format='{:d}E'.format(thetabins*phibins), array=ltdata_FD64, unit='s')
    col3 = fits.Column(name='TINT', format='{:d}E'.format(thetabins*phibins), array=ltdata_tint, unit='s2')
    col4 = fits.Column(name='TINTLL', format='{:d}E'.format(thetabins*phibins), array=ltdata_tint08, unit='s2')
    col5 = fits.Column(name='RA', format='1E', array=skra_d, unit='deg')
    col6 = fits.Column(name='DEC', format='1E', array=skdc_d, unit='deg')
    cols = fits.ColDefs([col1, col2, col3, col4, col5, col6])
    hdu1 = fits.BinTableHDU.from_columns(cols, name='EXPOSURE')
    hdu1.header['TELESCOP']= 'DAMPE'
    hdu1.header['ZMAX'] = (zmax, 'deg')
    hdu1.header['PIXTYPE'] = 'HEALPIX' # healpix header
    hdu1.header['ORDERING'] = ordering
    hdu1.header['NSIDE'] = nside
    hdu1.header['COORDSYS'] = 'CEL'
    hdu1.header['RADECSYS'] = ('FK5', 'J2000')
    hdu1.header['FIRSTPIX'] = 0
    hdu1.header['LASTPIX'] = npix-1
    hdu1.header['INDXSCHM'] = 'IMPLICIT'
    hdu1.header['TIMESYS'] = 'TT'
    hdu1.header['TIMEUNIT'] = 's'
    hdu1.header['LIVETIME'] = (ltdata_lttt, 'sec')
    hdu1.header['TLTRFMET'] = (metref, 'TLT ref. time in MET')
    t0 = utcref + atime.TimeDelta(gtistart[0], scale='tt', format='sec')
    hdu1.header['TSTART'] = (gtistart[0], t0.isot)
    t1 = utcref + atime.TimeDelta(gtistop[-1], scale='tt', format='sec')
    hdu1.header['TSTOP'] = (gtistop[-1], t1.isot)

    col1 = fits.Column(name='CTHETA_MIN', format='1E', array=cthetabounds[:-1])
    col2 = fits.Column(name='CTHETA_MAX', format='1E', array=cthetabounds[1:])
    cols = fits.ColDefs([col1, col2])
    hdu2 = fits.BinTableHDU.from_columns(cols, name='CTHETABOUNDS')
    hdu2.header['TELESCOP']= 'DAMPE'
    hdu2.header['CTHBINAL'] = 'LIN' if linctheta else 'SQ'
    hdu2.header['CTHMIN'] = ctheta_min
    hdu2.header['NCTHBIN'] = thetabins

    col1 = fits.Column(name='PHI_MIN', format='1E', array=phibounds[:-1], unit='deg')
    col2 = fits.Column(name='PHI_MAX', format='1E', array=phibounds[1:], unit='deg')
    cols = fits.ColDefs([col1, col2])
    hdu3 = fits.BinTableHDU.from_columns(cols, name='PHIBOUNDS')
    hdu3.header['TELESCOP']= 'DAMPE'
    hdu3.header['SCAXIS'] = '-X-Y-Z'
    hdu3.header['PHIBINAL'] = 'LIN'
    hdu3.header['NPHIBIN'] = phibins

    hdulist = fits.HDUList([hdu0, hdu1, hdu2, hdu3])
    _tprint('saving ...')
    hdulist.writeto(outfile, overwrite=True, checksum=True)
    _tprint('done')


def ltcube_sum(infile_list, outfile):
    """
    add up the ltcube file. You must be fully aware of the
    caveats listed below before using the function

    NOTE:
    1. no attemps are made to check the time continuity or
       overlap of the infile_list
    2. no attemps are made to check the consistency of the
       costheta or phi bins
    3. only support the infile generated with this scipts
    4. no memory optimization is done here

    parameters
    ----------
    infile_list: list
        the list of input ltcube files, len(infile_list)>1
    outfile: str
        the name of output file
    """
    assert isinstance(infile_list, list) and len(infile_list)>1

    # deal with the first ltcube file
    _tprint('dealing with {} ...'.format(infile_list[0]))
    hdl = fits.open(infile_list[0])

    hdu0 = hdl[0]
    hdu0.header['HISTORY'] = '^'*10 + infile_list[0] + '^'*10

    hdu1 = hdl[1]
    lt_fd08 = hdu1.data.field('LL')
    lt_fd64 = hdu1.data.field('HL')
    try:
        lt_tint = hdu1.data.field('TINT')
    except:
        lt_tint = None
    try:
        lt_tint08 = hdu1.data.field('TINTLL')
    except:
        lt_tint08 = None

    ra = hdu1.data.field('RA')
    dec = hdu1.data.field('DEC')
    coordsys = hdu1.header['COORDSYS']
    ordering = hdu1.header['ORDERING']
    tstart = hdu1.header['TSTART']
    tstop = hdu1.header['TSTOP']
    lt_lttt = hdu1.header['LIVETIME']

    # deal with other infiles
    for infile2 in infile_list[1:]:
        _tprint('dealing with {} ...'.format(infile2))
        hdl2 = fits.open(infile2)

        hdu0N = hdl2[0]
        for line in hdu0N.header['HISTORY']:
            hdu0.header['HISTORY'] = line
        hdu0.header['HISTORY'] = '^'*10 + infile2 + '^'*10

        hdu1N = hdl2[1]
        assert hdu1N.header['COORDSYS'] == coordsys
        assert hdu1N.header['ORDERING'] == ordering
        tstartN = hdu1N.header['TSTART']
        tstopN = hdu1N.header['TSTOP']
        if tstopN>tstop:
            tstop = tstopN
        if tstartN<tstart:
            tstart = tstartN
        lt_fd08 += hdu1N.data.field('LL')
        lt_fd64 += hdu1N.data.field('HL')
        if lt_tint is not None:
            lt_tint += hdu1N.data.field('TINT')
        if lt_tint08 is not None:
            lt_tint08 += hdu1N.data.field('TINTLL')
        lt_lttt += hdu1N.header['LIVETIME']

        del hdu1N.data
        hdl2.close()

    _tprint('total livetime = {:.2f}[s]'.format(lt_lttt))

    _tprint('preparing save data ...')
    ntheta_phi = lt_fd08.shape[1]
    col1 = fits.Column(name='LL', format='{:d}E'.format(ntheta_phi), array=lt_fd08, unit='s')
    col2 = fits.Column(name='HL', format='{:d}E'.format(ntheta_phi), array=lt_fd64, unit='s')
    col3 = fits.Column(name='RA', format='1E', array=ra, unit='deg')
    col4 = fits.Column(name='DEC', format='1E', array=dec, unit='deg')
    cols0 = [col1, col2, col3, col4]
    if lt_tint08 is not None:
        col5 = fits.Column(name='TINTLL', format='{:d}E'.format(ntheta_phi), array=lt_tint08, unit='s2')
        cols0.insert(2, col5)
    if lt_tint is not None:
        col6 = fits.Column(name='TINT', format='{:d}E'.format(ntheta_phi), array=lt_tint, unit='s2')
        cols0.insert(2, col6)
    cols = fits.ColDefs(cols0)

    hdu1a = fits.BinTableHDU.from_columns(cols, name='EXPOSURE')
    hdu1a.header['TELESCOP']= 'DAMPE'
    try:
        hdu1a.header['ZMAX'] = hdu1.header['ZMAX']
    except:
        pass
    hdu1a.header['PIXTYPE'] = 'HEALPIX' # healpix header
    hdu1a.header['ORDERING'] = ordering
    hdu1a.header['NSIDE'] = hdu1.header['NSIDE']
    hdu1a.header['COORDSYS'] = hdu1.header['COORDSYS']
    hdu1a.header['RADECSYS'] = hdu1.header['RADECSYS']
    hdu1a.header['FIRSTPIX'] = hdu1.header['FIRSTPIX']
    hdu1a.header['LASTPIX'] = hdu1.header['LASTPIX']
    hdu1a.header['INDXSCHM'] = 'IMPLICIT'
    hdu1a.header['TIMESYS'] = 'TT'
    hdu1a.header['TIMEUNIT'] = 's'
    hdu1a.header['LIVETIME'] = (lt_lttt, 'sec')
    t0 = utcref + atime.TimeDelta(tstart, scale='tt', format='sec')
    hdu1a.header['TSTART'] = (tstart, t0.isot)
    t1 = utcref + atime.TimeDelta(tstop, scale='tt', format='sec')
    hdu1a.header['TSTOP'] = (tstop, t1.isot)
    if lt_tint is not None:
        try:
            hdu1.header['TLTRFMET'] = hdu1.header['TLTRFMET']
        except:
            pass

    hdulist = fits.HDUList([hdu0, hdu1a, hdl[2], hdl[3]])
    _tprint('saving ...')
    hdulist.writeto(outfile, overwrite=True, checksum=True)
    _tprint('done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prefix_chars='-+', description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('scfile', type=str, help='The name of SC file')
    parser.add_argument('evfile', type=str, help='The name of EVENT file')
    parser.add_argument('outfile', type=str, help='The name of LTCUBE file output')

    parser.add_argument('--order', '-ord', type=int, default=6, help='order for healpix bin, nside=2**order, default = 6')
    parser.add_argument('--ordering', '-ordering', type=str, default='NESTED', choices=['NESTED', 'RING'], help='ordering scheme of healpix pixels, default = "NESTED"')
    parser.add_argument('--thetabins', '-ntheta', type=int, default=20, help='number of bins for theta, default = 20')
    #parser.add_argument('--phibins', '-nphi', type=int, default=72, help='number of bins for phi, default = 72')
    parser.add_argument('--phibins', '-nphi', type=int, default=1, help='number of bins for phi, default = 1')
    parser.add_argument('--zmax', '-zmax', type=float, default=100, help='the maximum value for zenith angle, default = 100')
    parser.add_argument('--nthreads', '-nthr', type=int, default=1, help='number of threads, default = 1')
    parser.add_argument('+linctheta', action='store_true', default=False, help='whether the costheta bin is linear or not, default=False')
    args = parser.parse_args()

    ltcube_phi(args.scfile, args.evfile, args.outfile, args.order, args.ordering, args.thetabins, args.phibins, args.zmax, args.nthreads, args.linctheta)
