""">>> pt.py <<<

   Perturbation Theory module. It also contains CAMB drivers

Current revision:
    ID:         $Id: pt.py 62 2008-08-21 21:57:49Z neyrinck $
    Date:       $Date: 2008-08-21 11:57:49 -1000 (Thu, 21 Aug 2008) $
    Revision:   $Revision: 62 $

(C) 2007 The CosmoPy Team (see Copyright for details)

COMMENTS:

    Camb class is working now...PowerSpectrum is not yet tested,
    but rewriten.
    in legacy code  Bispectru I simply change scipy to numarray,
    probably does not work...

    moreover, the ThreePointFunction never worked, I used a c-program
    instead,
"""

import os
import unittest
import numpy as N
import pylab as M
#import scipy.signal as SS
import scipysplines as SS
#import scipy.interpolate as SI
import param
import utils
import cex

sf = utils.SpecialFunctions()

def sigma2fromPk(c,r):
    """
    calculate sigma^2 from pk

    this function can be called with vectors or scalars, but always returns a vector
    """
    r = M.asarray(r)
    return 9.0/r**2*N.trapz(c.k*c.pk*sf.j1(M.outer(r,c.k))**2,
                         M.log(c.k))/2.0/M.pi**2

def xi2fromCambPk(c,r):
    """
    calculate 2pt corr. function from Camb instance (with its
    associated k,pk)

    this function can be called with vectors
    """
    r = M.asarray(r)
    return N.trapz(c.k**3*c.pk*sf.j0(M.outer(r,c.k)),
                         M.log(c.k))/2.0/M.pi**2

def xi2fromPk(k,pk,r):
    """
    calculate 2pt corr. function from k, p(k).
    It doesn't seem to work particularly well, though.

    this function can be called with vectors
    """
    r = M.asarray(r)

    print len(k),len(pk),len(r)

    return N.trapz(k**3*pk*sf.j0(M.outer(r,k)),
                         M.log(k))/2.0/M.pi**2

def wPk(c,r,w):
    """
    convolve Pk with an arbitrary window function w(kr)
    
    int k^2 P(k)w(kr) dk/2/pi^2

    e.g., the previous two functions can be realized as
    wPk(c,r,j0)
    wPk(c,r,lambda x: (3*pt.sf.j1(x)/x)**2))
    """
    r = M.asarray(r)
    return N.trapz(c.k**3*c.pk*w(M.outer(r,c.k)),
                         M.log(c.k))/2.0/M.pi**2

def normalizePk(c,sig8):
    """
    renormalize the power spectrum to the given sigma_8

    Note: this doesn't take into account the redshift; it just blindly sets c.pk to have
    sigma_8 = sig8.
    """
    sig2now = sigma2fromPk(c,8.)
    #print 'sig2now=',sig2now
    c.pk *= sig8**2/sig2now
    c.logpk = M.log(c.pk)

    c.cp.scalar_amp[0] = c.cp.scalar_amp[0] * sig8**2/sig2now[0] #inelegant tuple change

    # for scipy splines
    c.pkSplineCoeff = SS.cspline1d(c.logpk)
    return sig2now


class Camb(utils.SpecialFunctions):
    """
    class to drive camb and return vectors from it

    see camb example for details
    """
    def __init__(self,iniName = 'cosmopy_camb.ini',cambPath = '../CAMB',
                 cambParam=None,**kws):
        """
        kws is a set of arguments to change

        e.g. CambParams(output_root='test1') etc
        """
        if cambParam==None:
            self.cp = param.CambParams(**kws)
        else:
            self.cp = cambParam
            
        self.iniName = iniName
        self.cambPath = cambPath


    def printIniFile(self):
        """
        print CAMB ini file
        """
        from types import ListType
        
        f = file(self.iniName,'w')
    
        for kw in self.cp.keys():
            if type(self.cp[kw]) == ListType:
                for i,val in enumerate(self.cp[kw]):
                    f.write(kw+'('+str(i+1)+') = '+str(val)+'\n')
            else:
                f.write(kw+' = '+str(self.cp[kw])+'\n')

        f.close()

    def run(self):
        """
        create an inifile from the parameters and run camb on it
        and store the results in k,pk
        """
        self.printIniFile()
        os.system(self.cambPath+'/camb '+self.iniName)
        self.k,self.pk = utils.readColumns(self.cp.output_root
                                           +'_matterpower.dat')
        
        self.logk, self.logpk = M.log(self.k), M.log(self.pk)
        self.pkSplineCoeff = SS.cspline1d(self.logpk)

    def runCosmicEmu(self, sig8):
        """
        use the CosmicEmu (Lawrence, Heitmann, Habib et al.) nonlinear power spectrum emulator.

        Note that littleh will be (re)set, according to CMB constraint
        """
        # first run in camb

        #c_lin = copy.copy(self)
        #c_lin.

        a = 1./(self.cp.transfer_redshift[0] + 1)
        if (a < 0.5)|(a > 1.):
            print 'Warning! outside range of z accuracy (1 - 0).'
        
        scalefax=N.arange(0.5,1.001,0.1)
        coltoreturn = N.where(N.abs(a - scalefax) == min(N.abs(a - scalefax)))[0][0]
        print 'returning results at a=',scalefax[coltoreturn]
        f = file('emu.in','w')
        f.write('emu.out\n')
        f.write(str(self.cp.omch2+self.cp.ombh2)+'\n')
        f.write(str(self.cp.ombh2)+'\n')
        f.write(str(self.cp.scalar_spectral_index[0])+'\n')
        f.write(str(sig8)+'\n')
        f.write(str(self.cp.w)+'\n')
        f.write('2\n')
        f.close()
        
        os.system('/Users/neyrinck/CosmicEmu_v1.0/emu.exe < emu.in > emu.err')

        # read little h
        f = open('emu.out','r')
        for i in range(6):
            dumb=f.readline()
        littleh = float(f.readline().split()[-1])
        self.cp.hubble = 100.*littleh
        print 'littleh changed to ',littleh

        f.close()
        
        emu = N.loadtxt('emu.out')
        kemu = emu[:,0]/littleh # should now be in h/Mpc
        pemu = emu[:,coltoreturn+1]*littleh**3 # should now be in (Mpc/h)^3
        if self.cp.transfer_k_per_logint == 0:
            self.cp.transfer_k_per_logint = 512.
        
        #need to get into log-scale k
        self.k = kemu[0]*10.**N.arange(0.,N.log10(kemu[-1]/kemu[0]),1./self.cp.transfer_k_per_logint)

        interpemu = SI.interp1d(N.log(kemu),N.log(pemu))#,kind='cubic')
        #self.pk = interpemu(self.k)
        self.pk = N.exp(interpemu(N.log(self.k)))
        #self.pk = utils.splineIntLinExt(pemu, kemu, self.k)

        self.logk = 1.*N.log(self.k)
        self.logpk = 1.*N.log(self.pk)

        #self.kextend(-5,3,calcsplinecoeff=True)

        self.pkSplineCoeff = SS.cspline1d(self.logpk)

        return

    def runEisensteinHu(self, sig8):
        """
        use (much faster, but somewhat less accurate) Eisenstein & Hu
        code.

        Not tested recently.
        """
        #Output EHu file
        f = file('ehu.in','w')

        #f.write((str(self.cp.omega_baryon + self.cp.omega_cdm))+', '+str(self.cp.omega_lambda)+', '+\
        #        str(self.cp.omega_neutrino)+', '+str(self.cp.omega_baryon)+'\n')

        h = self.cp.hubble/100.
        om0 = (self.cp.ombh2 + self.cp.omch2)/h**2
        f.write(str(om0)+', '+str(1.-om0)+', '+ str(self.cp.omega_neutrino)+', '+str(self.cp.ombh2/h**2)+'\n')
        f.write(str(h)+', '+str(self.cp.temp_cmb)+', '+str(self.cp.massless_neutrinos)+'\n')
        f.write(str(self.cp.transfer_redshift[0])+'\n')
        f.write(str(self.cp.transfer_kmax)+', '+str(self.cp.transfer_k_per_logint)+'\n')
        f.write('1\n')
        tilt = self.cp.scalar_spectral_index[0]
        f.write(str(tilt)+'\n')
        f.write('0\n')

        f.close()

        # run EHu code
        os.system('../ehu/power < ehu.in > ehu.crap')

        # read into c.k, c.pk
        eh = N.loadtxt('trans.dat')
        self.k = eh[:,0]*1.
        #print self.k
        self.logk = M.log(self.k)
        self.trans = eh[:,1]*1.
        if tilt == 1.:
            delH =  1.94e-5*(self.cp.omega_cdm + self.cp.omega_baryon)**(-0.785)
            delta = delH**2*(3000.0*self.k/(self.cp.hubble/100.))**4*self.trans**2
        else:
            delH =  1.94e-5*self.cp.omega_cdm**(-0.785 - 0.05*M.log(tilt))\
                   * M.exp(-0.95*(tilt - 1.) - 0.169*(tilt - 1)**2)
            delta = delH**2*(3000.0*self.k/(self.cp.hubble/100.))**(3 + tilt)*self.trans**2

        # Just an approximate normalization; really need sig8.
            
        self.pk = (2.*M.pi**2 * delta/self.k**3)*(self.cp.hubble/100.)**3
        if self.cp.transfer_redshift[0] > 0.:
            ps = PowerSpectrum(self.cp)
            sig8use = sig8*ps.d1(self.cp.transfer_redshift[0])/ps.d1(0.)
        else:
            sig8use = sig8
        normalizePk(self,sig8use) # sets c.logpk, too

        return

    def kextend(self,mini,maxi,logkoverk3extrap = 0,calcsplinecoeff = False):
        """
        Extend range of pk over which k is known, from 10^mini to 10^maxi.
        Keeps c.k the same in middle, but extrapolates on both sides.

        The logk/k**3 fit (used if last argument = 1) only works reliably if the highest k is around 100
        """

        log10interval = M.log10(self.k[-1]/self.k[0])/(len(self.k) -1.)
        self.numtomini = M.floor((M.log10(self.k[0]) - mini)/log10interval)
        if self.numtomini < 0:
            self.numtomini = 0
        realmini = M.log10(self.k[0]) - log10interval*self.numtomini

        kx = 10.**M.arange(realmini,maxi+log10interval,log10interval)
        px = self.pkInterp(kx,calcsplinecoeff=calcsplinecoeff)
        self.k = kx
        self.logk = M.log(kx)
        self.pk = px

        kmax = self.k[-1]

        #fit logk/k**3 at high end; this is presumably more accurate, but may crash
        if logkoverk3extrap == 1:
            p0 = (self.pk[-1]*self.k[-1]**3 - self.pk[-2]*self.k[-2]**3)/\
                 (M.log(self.k[-1]) - M.log(self.k[-2]))

            const = (self.pk[-1]*self.k[-1]**3*M.log(self.k[-2]) - \
                     self.pk[-2]*self.k[-2]**3*M.log(self.k[-1]))/\
                     (self.pk[-1]*self.k[-1]**3 - self.pk[-2]*self.k[-2]**3)

            if ((p0 > 0.) * (maxi > kmax)):  # if it worked okayly
                w = M.where(self.k > kmax)[0][0]
                self.pk[w:] = p0*(self.logk[w:] - const)/self.k[w:]**3

        self.logpk = M.log(px)
        self.pkSplineCoeff = SS.cspline1d(self.logpk)

    def ktrunc(self,trunc):
        """
        Set pk=0 for k<trunc
        """
        for i in range(len(self.k)):
            if self.k[i] < trunc:
                self.pk[i] = 0.
                self.logpk[i] = -700. #kinda arbitrary

    def sigma2(self,r):
        """
        calculate sigma^2
        """
        return 9.0/r**2*N.trapz(self.k*self.pk*(self.j1(self.k*r))**2,
                              M.log(self.k))/2.0/M.pi**2

    def logpkInterp(self, logknew, lin=False):
        """
        log P(k) interpolated in log space for arbitrary k
        Linearly interpolate if lin=True
        """
        if lin == True:
            return utils.interpolateLin(self.logpk,self.logk,logknew)
        else:
            return utils.splineResampleUniform(self.logpk,self.logk,logknew)
    def pkInterp(self, knew, calcsplinecoeff = False):
        """
        P(knew) interpolated in log space for arbitrary k.
        Call with calcsplinecoeff = True if the power spectrum (self.pk) might have changed recently.
        """
        if (calcsplinecoeff):
            self.pkSplineCoeff = SS.cspline1d(self.logpk)
        
        newy = 0.*knew

        wn0 = N.where(knew > 0.)
        if len(wn0) > 0:
            newy[wn0[0]] = M.exp(utils.splineIntLinExt(self.logpk, self.logk, M.log(knew[wn0]),splinecoeff = self.pkSplineCoeff))
        # do 0
        return newy
            
#
#    def deltavir(self):
#        """
#        calculate delta_vir
#        Taken from Hamana's code
#        """
#        x = (1./(self.cp.transfer_redshift + 1.))*\
#            (1./(self.cp.omega_baryon + self.cp.omega_cdm) - 1.)
#        deltavir=18.0*M.pi**2*(1.0+0.4093*x**(2.71572/3.))
#        return deltavir

def realEqual(x,y,eps=10e-10):
    """
    equality to be used in test of real numbers
    """
    return abs(x-y) < eps

class CambTests(unittest.TestCase):
    """
    unit tests for the Camb class
    """
    def testCamb0(self):
        c = Camb(do_nonlinear=0);
        c.printIniFile();
        c.run()
#        print c.k[79],c.pk[79]
        self.failIf( c.k[79]!=0.00011671999999999999 or c.pk[79] != 486.12)
        self.failIf( not realEqual(M.sqrt(c.sigma2(8)),0.8842783, 10e-5 ))
        self.failIf( not realEqual(sigma2fromPk(c,8)[0], 0.78195126729980657))
        self.failIf( not realEqual(xi2fromPk(c,5)[0], 1.1622225420470442))
        self.failIf( not realEqual(wPk(c,8,lambda x: (3*sf.j1(x)/x)**2)[0],
                                   0.78195126729980657))
        self.failIf(not realEqual(wPk(c,5,sf.j0)[0],1.1622225420470442))
        normalizePk(c,0.9)
        self.failIf(not realEqual(sigma2fromPk(c,8)[0],0.9**2))


class PowerSpectrum(utils.SpecialFunctions,param.CosmoParams):
    """
    Numerical Calculation of the Power Spectrum

    Includes transfer functions, growth factor
    P(k), 2-pt function,etc.
    """

    def __init__(self,cambParam=None,**kws):
        """
        Initialize cosmological parameters
        """
        if cambParam==None:
            self.cp = param.CambParams(**kws)
        else:
            self.cp = cambParam

        (self.h,self.om0,self.omB,self.omL, self.omR,self.n) = (
         self.cp.hubble/100.,self.cp.omega_cdm,self.cp.omega_baryon,
         self.cp.omega_lambda, 4.17e-5/(self.cp.hubble/100.)**2,
         self.cp.scalar_spectral_index[0])
        self.omC = self.om0-self.omB
        self.delH = 1.94e-5*self.om0**(-0.785 - 0.05*M.log(self.om0))*M.exp(-0.95*(self.n - 1) - 0.169*(self.n - 1)**2)
        self.t = self.bbks
##   h^-1 normalization for sigma8=0.9
        self.bondEfsNorm = 216622.0
##      bare normalizatin by Pan (apparently correct for h_1)   
        self.bondEfsNorm = 122976.0

    def bbks(self,k):
        """
        BBKS transfer function
        """
        q = k/self.om0/self.h**2*M.exp(self.omB + M.sqrt(2*self.h)*self.omB/self.om0)
        return M.log(1.0 + 2.34*q)/(2.34 *q) *(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)

    def bondEfs(self,k):
        """
        alternative transfer function fit by Bond end Efstathiou for
        gif simulations which apparently use it

        this is normalized differently from the bbks transfer function
        and can only be used at z=0
        """
        Alpha=6.4
        b=3.0
        c=1.7
        nu=1.13
        gamma=self.om0*self.h
        q=k/gamma
        return self.bondEfsNorm/(1.0+(Alpha*q +(b*q)**1.5 + (c*q)**2)**nu)**(2.0/nu)


    def om0z(self,z):
        """
        Omega as a function of redshift
        """
        return self.om0*(1.0 + z)**3/(self.omL+self.omR*(1.0 + z)**2+self.om0*(1.0 + z)**3)


    def omLz(self,z):
        """
        Lambda as a function of redshift
        """
        return self.omL/(self.omL + self.omR*(1.0 + z)**2 + self.om0*(1.0 + z)**3)

    def d1(self,z):
        """
        Growth function
        """
        return 5*self.om0z(z)/(1.0 + z)/2.0/(self.om0z(z)**(4./7.0) - self.omLz(z) + (1.0 + self.om0z(z)/2.0)*(1.0 + self.omLz(z)/70.0))

    def deltac(self,z):
        """
        calculate delta_collapse (the number that's ~1.686 for E-dS)
        Taken from Nakamura and Suto, 1997
        """
        deltac0 = 3.*(12.*M.pi)**(2./3.)/20.*(1.+0.0123*M.log10(self.om0z(z)))
        deltac = deltac0*self.d1(0)/self.d1(z)
                                           
        return deltac

    def delta(self,k,z=0):
        """
        k^3P(k)/2*pi^2: the dimensionless power spectrum
        """
        if self.t == self.bondEfs:
            if z != 0:
                raise cex.ZNotZeroInBondEfsTk
            return k**4*self.bondEfs(k)
        else:
            return self.delH**2*(3000.0*k/self.h)**(3 + self.n)*self.t(k)**2 * self.d1(z)**2/self.d1(0)**2

    def p(self,k,z=0):
        """
        bare power spectrum
        """
        return 2*M.pi**2 * self.delta(k, z)/k**3;

    def ph(self,k,z=0):
        """
        power spectrum in hMpc-1 units
        """
        return self.p(k*self.h,z)*self.h**3

class TriBiSpectrumFromCamb(utils.SpecialFunctions):
    """
    functions to calculate bispectrum in the weakly nonlinear
    regime, using a camb instance

    """
    def __init__(self,c):
        #Einstein-de Sitter value of mu
        #self.mu = 3./7.

        #Use Omega_matter for mu.
        self.mu = 3.0/7.0*((c.cp.ombh2+c.cp.omch2)/(c.cp.hubble/100.)**2)**(-2./63.)

        # use just Omega_cdm for mu.  Doesn't make much difference
        #self.mu = 3.0/7.0*c.cp.omega_cdm**(-2./63.)

        print 'mu ratio:',self.mu/(3./7.)
        self.p0 = c.pk[0]/c.k[0]
        self.f1 = 1.
        self.g1 = 1.
        self.kFloor = 1e-20
 
    def sumDotProd(self,start1,end1,start2,end2,k,cosTheta):
        """
        Calculates (k_start1 + ... + k_end1) . (k_start2 + ... + k_end2)
        k: [|k0|,|k1|,|k2| ...]
        thetarr: [[cos(theta_k0,k0),cos(theta_k0,k1),...]
                  [cos(theta_k1,k0), ...]]
        ...]
        """
        start1 -= 1
        start2 -= 1
        # Account for Python non-inclusive last element of array
        nentries = (end1-start1)*(end2-start2)

        #print start1,end1,start2,end2
        #print M.outer(k[start1:end1],k[start2:end2])

        ans = M.sum(M.reshape(M.outer(k[start1:end1],k[start2:end2]) * \
                              cosTheta[start1:end1,start2:end2], nentries))
        return ans

    def f2(self,k1,k2,cosTheta):
        """
        f2 perturbation theory kernel, symmetrized

        Actually, this is DOUBLE the standard f2 kernel.
        """
        ans = (1.0 + self.mu) +(k1/k2 + k2/k1)*cosTheta + (1.0 - self.mu)*cosTheta**2
        return ans

    def g2(self,k1,k2,cosTheta):
        
        return (self.mu + 0.5*(k1/k2 + k2/k1)*cosTheta + (1. - self.mu)*cosTheta**2)

    def f3(self,k,cosTheta):
        """
        symmetrized f3 -- see Cooray & Sheth, eq. 36
        k = 0 not ok

        Not recently tested
        """
        n = 3
        m = 1

        m1term = self.g1 * (2.*n + 1.)*(self.f2(k[1],k[2],cosTheta[1,2])/2.) * \
                 (self.sumDotProd(1,n, 1,m, k,cosTheta)/self.sumDotProd(1,m, 1,m, k,cosTheta)) +   \
                 (self.sumDotProd(1,n, 1,n, k,cosTheta)*self.sumDotProd(1,m, m+1,n, k,cosTheta)) / \
                 (self.sumDotProd(1,m, 1,m, k,cosTheta)*self.sumDotProd(m+1,n, m+1,n,k,cosTheta)) * \
                 self.g2(k[1],k[2],cosTheta[1,2])
        m = 2
        m2term = self.g2(k[1],k[1],cosTheta[0,1]) * (2.*n + 1.)* self.f1 * \
                 (self.sumDotProd(1,n, 1,m, k,cosTheta)/self.sumDotProd(1,m, 1,m, k,cosTheta)) +   \
                 (self.sumDotProd(1,n, 1,n, k,cosTheta)*self.sumDotProd(1,m, m+1,n, k,cosTheta)) / \
                 (self.sumDotProd(1,m, 1,m, k,cosTheta)*self.sumDotProd(m+1,n, m+1,n,k,cosTheta)) * \
                 self.g1
        return (m1term+m2term)/((n-1.)*(2.*n+3))

    def b10(self,k1,k2,c):
        """
        monopole of the unpermuted bispectrum
        """
        return 2.0/3.0*(2.0+self.mu)*c.pkInterp(k1)*c.pkInterp(k2)
        
    def b11(self,k1,k2,c):
        """
        dipole of the unpermuted bispectrum
        """
        return (k1/k2+k2/k1)*c.pkInterp(k1)*c.pkInterp(k2)
    
    def b12(self,k1,k2,c):
        """
        quadrupole of the unpermuted bispectrum
        """
        return 2.0/3.0*(1-self.mu)*c.pkInterp(k1)*c.pkInterp(k2)

    def b1(self,k1,k2,cosTheta,c):
        """
        unpermuted bispectrum
        """
#        if k1 == 0.:
#            return self.p0*k2*c.pkInterp(k2)*cosTheta
#        if k2 == 0.:
#            return self.p0*k1*c.pkInterp(k1)*cosTheta
        
        return self.f2(k1,k2,cosTheta)*c.pkInterp(k1)*c.pkInterp(k2)

    def k3Length(self,k1,k2,cosTheta):
        """
          the length third k vector, when k1+k2+k3=0
        """
        return M.sqrt(k1**2 + k2**2 + 2*k1*k2*cosTheta)

    def cos1(self,k1,k2,cos12):
        """
          k1.k3/|k1|/|k3}=k1.(-k1-k2)/|k1|/|k3|
          the cos angle between 1 and 3
        """
        return (-k1 - k2*cos12)/self.k3Length(k1, k2, cos12)

    def cos12_1(self,k1,k2,k12,cos12):
        """
        Cos(angle between (k1+k2) and k2)
        """
        return (k1 + k2*cos12)/k12

    def cos1m2_2(self,k1,k2,k12,cos12):
        """
        Cos(angle between (k1-k2) and k1)
        """
        return (k1*cos12 - k2)/k12

    def cos12_3(self,k1,k2,k12,cos13,cos23):
        """
        Cos(angle between (k1+k2) and k3)
        k12 = |k1+k2|
        """
        return (k1*cos13 + k2*cos23)/k12

    def b(self,k1,k2,cosTheta,c):
        """
        finally, the full PT bispectrum
        """
        return self.b1(k1, k2, cosTheta,c) + \
               self.b1(k1, self.k3Length(k1, k2, cosTheta), \
                       self.cos1(k1, k2, cosTheta),c) +\
               self.b1(k2, self.k3Length(k2, k1, cosTheta), \
                       self.cos1(k2, k1, cosTheta),c)
        
    def qB(self,k1,k2,cosTheta):
        return self.b(k1, k2, cosTheta)/(c.pkInterp(k1)*c.pkInterp(k2) +\
               c.pkInterp(k2)*c.pkInterp(self.k3Length(k1, k2, cosTheta)) + \
               c.pkInterp(self.k3Length(k1, k2, cosTheta))*c.pkInterp(k1))

    # Trispectrum
    
    def tf21(self, i1,i2,i3,k,cosTheta,pk,c):
        """
        Calculates one of the Trispectrum F2 terms (of which there are 12 permutations)

        Not recently tested
        """

        k12length = self.k3Length(k[i1],k[i2],cosTheta[i1,i2])
        costh12_1 = self.cos12_1(k[i1],k[i2],k12length,cosTheta[i1,i2])
        costh12_3 = self.cos12_3(k[i1],k[i2],k12length,cosTheta[i1,i3],cosTheta[i2,i3])

        f2prod = self.f2(k12length, k[i1], -costh12_1) * self.f2(k12length, k[i3], costh12_3)/4.
        ans = f2prod * pk[i1] * c.pkInterp(k12length) * pk[i3]
        #print i1,i2,i3,k[i1],k[i2],k[i3],'\t',cosTheta[i1,i2],costh12_1,costh12_3,f2prod,k12length,'\t',ans
        return ans


    def tf31(self, i ,k,cosTheta,pk):
        """
        Calculates one of the Trispectrum F3 terms (of which there are 4 permutations)

        Not recently tested
        """
        cosThetaPermuted = cosTheta*0.

        one23 =  M.arange(3)
        for i0 in one23:
            for i1 in one23:
                cosThetaPermuted[i0,i1] = cosTheta[i[i0],i[i1]]
                    
        return self.f3(M.array([k[i[0]],k[i[1]],k[i[2]]]), cosThetaPermuted) * pk[i[0]] * pk[i[1]] * pk[i[2]]

    def t(self,k,cosTheta,pk,c):
        """
        Raw trispectrum

        Not recently tested
        """
        pk = c.pkInterp(k)
        f2term = (self.tf21(0,1,2, k,cosTheta,pk,c)+self.tf21(1,2,0, k,cosTheta,pk,c)+self.tf21(2,0,1, k,cosTheta,pk,c)+ \
                  self.tf21(1,2,3, k,cosTheta,pk,c)+self.tf21(2,3,1, k,cosTheta,pk,c)+self.tf21(3,1,2, k,cosTheta,pk,c)+ \
                  self.tf21(2,3,0, k,cosTheta,pk,c)+self.tf21(3,0,2, k,cosTheta,pk,c)+self.tf21(0,2,3, k,cosTheta,pk,c)+ \
                  self.tf21(3,0,1, k,cosTheta,pk,c)+self.tf21(0,1,3, k,cosTheta,pk,c)+self.tf21(1,3,0, k,cosTheta,pk,c)) * 4.

        f3term = (self.tf31(M.array([0,1,2]),k,cosTheta,pk) + self.tf31(M.array([1,2,3]),k,cosTheta,pk) + \
                  self.tf31(M.array([2,3,1]),k,cosTheta,pk) + self.tf31(M.array([3,1,2]),k,cosTheta,pk)) * 6.

        #print cosTheta,f2term, f3term, ft2term+f3term
        return f2term + f3term

    def alpha(self,k1,k2,c):
        """
        returns the alpha wavevector-coupling function.
        a,b are the magnitudes of the two vectors it accepts;
        c is the cosine of the angle between them.
        """
        if k1 == 0.:
            return 0.
        else:
            return 1. + k2*c/k1

    def beta(self,k1,k2,c):
        """
        returns the beta wavevector-coupling function.
        a,b are the magnitudes of the two vectors it accepts;
        c is the cosine of the angle between them.
        """
        if (k1*k2 == 0.):
            return 0.
        else:
            return c*(k1/k2 + k2/k1)/2. + c*c

    def f3a_ab(self,a,b,c):
        """
        returns F3(a,-a,b)
        a,b = vector lengths
        c = cos(angle between them)
        """
        ans = ((b*b/a) * (b*c - a)/(a*a + b*b - 2.*a*b*c)) * self.g2(a,b,-c) +\
              7. * (b*c/a) * self.f2(a,b,-c)/2.

        return ans/18.

    def f3ab_a(self,a,b,c):
        """
        returns F3(a,b,-a)
        a, b = vector lengths
        c = cos(angle between them)
        """
        ans = 7*b*c/a * self.f2(b,a,-c)/2. + \
              (b*b/a)*(b*c-a)/(a*a+b*b-2.*a*b*c)*self.g2(b,a,-c) +\
              self.g2(a,b,c)*(7.*(b*b + a*b*c) - (b/a)**2 * (a*a + a*b*c))/\
              (a*a+b*b+2.*a*b*c)

        return ans/18.

    def f3ba_a(self,a,b,c):
        """
        returns F3(b,a,-a)
        a,b = vector lengths
        c = cos(angle between them)
        """
        ans = self.g2(a,b,c)*(7.*(b*b + a*b*c) - (b/a)**2 * (a*a + a*b*c))/\
              (a*a+b*b+2.*a*b*c)

        return ans/18.

    def f3aa_aSymmetrized(self,a,one = 0.9999999):
        """
        returns F3(a,a,-a)
        one = cos(angle between them)
        """
        return 2.*(self.f3ab_a(a,a,one) + \
                   self.f3a_ab(a,a,one) + \
                   self.f3ab_a(a,a,-one))/6.

    def f3a_abSymmetrized_old(self,a,b,c):
        """
        returns symmetrized F3(a,-a,b)
        a,b = vector lengths
        c = cos(angle between them)

        This does a lot of things manually; slower than the next one,
        but allows for mu != 3/7
        """
        return (self.f3a_ab(a,b,c) + self.f3a_ab(a,b,-c) +\
                self.f3ab_a(a,b,c) + self.f3ab_a(a,b,-c) +\
                self.f3ba_a(a,b,c) + self.f3ba_a(a,b,-c))/6.

    def f3a_abSymmetrized(self,a,b,c):
        """
        returns symmetrized F3(a,-a,b),
        derived from eq. 83 from Valageas (2004)
        a,b = vector lengths
        c = cos(angle between them)

        Doesn't allow for mu != 3/7
        """
        return b*b*(-21*b**4*c*c + a**4*(10-59*c*c+28*c**4) + \
                    2*a*a*b*b*(5-22*c*c+38*c**4))/\
                    (126*a*a*(a**4+b**4+2*a*a*b*b*(1-2*c*c)))

    def f3_ab_b(self,a,b,c):
        """
        returns 18*F3(-a,b,-b)
        a = length of first vector
        b = length of second vector
        c = cos(angle between them)

        Seems like it might be wrong (but it's not used for anything much)
        """
        ans = (7.*(a*a - a*b*c) + (a*a/b)*(a*c - b))*self.g2(a,b,-c)/(a*a + b*b - 2.*a*b*c)
        #print 'b_b:',ans,
        return ans

    def tk1mk1k2mk2_array(self,k1,k2,cos12,c,term):
        """
        Return tk1mk1k2mk2, looping over an array of cos12's
        """
        ans = cos12*0.
        for i in range(len(cos12)):
            ans[i] = self.tk1mk1k2mk2(k1,k2,cos12[i],c,term)

        return ans

    def tk1mk1k2mk2(self,k1,k2,cos12,c,term):
        """
        T(k1,-k1,k2,-k2), where cos(theta_12) = cos12.
        term=3: just return terms involving f_2
        term=2: "    "         "    "       f_3

        Note: only for use while angle-averaging!!!
        It uses simplifications only valid if we're integrating through both c, -c.

        See Neyrinck & Szapudi 2007 (arXiv:0710.3586), appendix A
        """

        k = M.array([k1, k2])
        pk = c.pkInterp(k)

        k1mk2 = self.k3Length(k1,k2,-cos12)
        k1pk2 = self.k3Length(k1,k2,cos12)
        if k1mk2 == 0.:
            pk1mk2 = 0.
        else:
            pk1mk2 = c.pkInterp(k1mk2)

        if term != 3:
            cos1m2_2 = self.cos1m2_2(k1,k2,k1mk2,cos12)
            cos2m1_1= self.cos1m2_2(k2,k1,k1mk2,cos12)
            f2k1mk2_2 = self.f2(k1mk2,k2,cos1m2_2)/2.
            f2k2mk1_1 = self.f2(k1mk2,k1,cos2m1_1)/2.
            f2term = 8.*pk1mk2*(f2k1mk2_2*pk[1] + f2k2mk1_1*pk[0])**2

        if term != 2:
            f3term = 12. * (self.f3a_abSymmetrized(k1,k2,cos12)*pk[0]**2*pk[1] +\
                            self.f3a_abSymmetrized(k2,k1,cos12)*pk[0]*pk[1]**2)

        if term == 2:
            return f2term
        if term == 3:
            return f3term

        return f2term+f3term

class ThreePointFunction(TriBiSpectrumFromCamb):
    """
    class to predict the three-point correlation function in
    the weakly nonlinear regime
    """
    def __init__(self,params):
        """
        initialize cosmology
        """
        BiSpectrum.__init__(self,params)

    def pkj0(self,r,z=0):
        """
          integral of the power spectrum over j0
          (essentially the 2point function)
        """
        return M.sqrt(M.pi/2.0)*self.besselInt.besselInt(lambda k: self.delta(k/r, z)/k**1.5,0.5,self.besselN,self.besselh)
##        return M.integrate.quad(lambda k: self.delta(k, z)/k*self.j0(k*r),0.0,self.besselMax/r)[0]

    def pkj1pk(self,r,z=0):
        """
          integral of the power spectrum*k over j1
        """
        return M.sqrt(M.pi/2.0)/r*self.besselInt.besselInt(lambda k: self.delta(k/r, z)/M.sqrt(k),1.5,self.besselN,self.besselh)
#        return M.integrate.quad(lambda k: self.delta(k, z)*self.j1(k*r),0.0,self.besselMax/r)[0]

    def pkj1mk(self,r,z=0):
        """
          integral of the power spectrum/k over j1
        """
        return M.sqrt(M.pi/2.0)*r**self.besselInt.besselInt(lambda k: self.delta(k/r, z)/k**2.5,1.5,self.besselN,self.besselh)
##        return M.integrate.quad(lambda k: self.delta(k, z)/k**2*self.j1(k*r),0.0,self.besselMax/r)[0]

    def pkj2(self,r,z=0):
        """
          integral of the power spectrum over j2
        """
        return M.sqrt(M.pi/2.0)*self.besselInt.besselInt(lambda k: self.delta(k/r, z)/k**1.5,2.5,self.besselN,self.besselh)
##        return M.integrate.quad(lambda k: self.delta(k, z)/k*self.j2(k*r),0.0,50.0/r)[0]

    def zeta1(self,r1,r2,x,z=0):
        """
        Three-point correlation function without the permutations
        """
## real dynamics
        return (2.0/3.0*(2.0+self.mu)*self.pkj0(r1,z)*self.pkj0(r2,z)-(self.pkj1mk(r1,z)*self.pkj1pk(r2,z)+self.pkj1mk(r2,z)*self.pkj1pk(r1,z))*x+2.0/3.0*(1-self.mu)*self.pkj2(r1,z)*self.pkj2(r2,z)*(-0.5+1.5*x**2))
## real dynamics with neg. dipole sign
##        return (2.0/3.0*(2.0+self.mu)*self.pkj0(r1,z)*self.pkj0(r2,z)+(self.pkj1mk(r1,z)*self.pkj1pk(r2,z)+self.pkj1mk(r2,z)*self.pkj1pk(r1))*x+2.0/3.0*(1-self.mu)*self.pkj2(r1,z)*self.pkj2(r2,z)*(-0.5+1.5*x**2))
## Zeldovich dynamics
##        return self.pkj0(r1,z)*self.pkj0(r2,z)-(self.pkj1mk(r1,z)*self.pkj1pk(r2,z)+self.pkj1mk(r2,z)*self.pkj1pk(r1))*x/2.0
## Zeldovich dynamics with negative dipole
##        return self.pkj0(r1,z)*self.pkj0(r2,z)+(self.pkj1mk(r1,z)*self.pkj1pk(r2,z)+self.pkj1mk(r2,z)*self.pkj1pk(r1))*x/2.0

    def zeta(self,r1,r2,cosTheta):
        """
        finally, the full three-point
        note that for this r2-r1=r3 (triangle), which means
        cosTheta -> -cosTheta compared to the bispectrum
        """
        r3 = self.k3Length(r1,r2,-cosTheta)
        return self.zeta1(r1, r2, cosTheta) + self.zeta1(r1, r3, -self.cos1(r1, r2, -cosTheta)) + self.zeta1(r2, r3, -self.cos1(r2, r1, -cosTheta))
        
    def qZeta(self,r1,r2,cosTheta):
        r3 = self.k3Length(r1,r2,-cosTheta)
        xi1 = self.pkj0(r1)
        xi2 = self.pkj0(r2)
        xi3 = self.pkj0(r3)
        return self.zeta(r1, r2, cosTheta)/(xi1*xi2+xi2*xi3+xi3*xi1)


def main():
    unittest.main()
if __name__=='__main__' :
    main()
##    c.plotCl()
