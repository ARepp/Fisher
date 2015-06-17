"""
  p(delta), moments and cumulants for the Denmu simulations

  (c) 2014 Istvan Szapudi
  started: Fri Aug  1 14:03:14 CEST 2014 
"""

import numpy as N
import matplotlib.pylab as P
#import f77io as F
import pdelta as CIC
from glob import glob
from scipy.io.idl import readsav

datadir = '/data/denmu/'
mapLCDM = 'demnu_lcdm_z0_r3.90_map_IDL.sav'         
mapNu1 = 'demnu_nu53_nu_z0_r3.90_map_IDL.sav'
mapNu2 = 'demnu_nu53_cdm_z0_r3.90_map_IDL.sav'

def getDemNu(fname):
    """
    read IDL data cube assumed to be stored in a flat structure 
    of 

    data['tabn',n**3]
    """
    d = readsav(fname)['tabn']
    print d[0:11]
    n3 = d.shape
    n = ((int) (2**(N.log2(n3)/3)))
    print n, n3
    return d.reshape((n,n,n))

def testPN():
    cd = getDemNu(mapLCDM)
    npix = N.size(cd.flat)
    cd = cd/((N.double) (N.sum(cd))/npix) #turn into a density field 
    print cd.shape
    CIC.cumulate(cd)
    for i in range(5,6):
        d = CIC.dFromCumulate(cd,i)
        p = CIC.pHist(d,res = 0.001)
        outName = 'pdelta'+str(i)+'-0.001.txt'
        N.savetxt(outName,p)
        moms,lmoms = CIC.directMoments(d)
        outName = 'moms'+str(i)+'.txt'
        N.savetxt(outName,moms)
        outName = 'lmoms'+str(i)+'.txt'
        N.savetxt(outName,lmoms)


def testPk():
    d = getDemNu(mpLCDM) 

if __name__=="__main__":
    test()
