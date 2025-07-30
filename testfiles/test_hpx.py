## test healpix
import numpy as np
import healpy as hp

order = 7
nside = 2**order
npix = 12*nside**2

d2r = np.pi/180.

pixes = [0,1,2,3,4,5,6,7,8,9, npix-10, npix-9, npix-8, npix-7, npix-6, npix-5, npix-4, npix-3, npix-2, npix-1]

for i in pixes:
    ra_deg, dec_deg = hp.pix2ang(nside=nside, ipix=i, nest=True, lonlat=True)
    print(i, np.sin(dec_deg*d2r), np.cos(dec_deg*d2r), ra_deg*d2r)
