import numpy as N
import matplotlib.pyplot as plt
import f77io as F
from glob import glob
import power as Pow
import pdeltaNU as CIC
import pt
from scipy.integrate import simps

datadir = '/home/melody/Desktop/Simulations/denmu/'
mapNu = datadir+'demnu_lcdm_z0_a3.90625_map_cubic_IDL.sav'
mapNu1 = datadir+'demnu_nu17_z0_a3.90625_map_cubic_IDL.sav'
mapNu2 = datadir+'demnu_nu30_z0_a3.90625_map_cubic_IDL.sav'

dataJ = '/home/melody/Desktop/Simulations/PK_DEMNU/'

"""
Read files
"""

cdNu1 = CIC.getDemNu(mapNu1)
cdNu1 = N.double(cdNu1)
den1 = N.sum(cdNu1)/2000.**3
npix = N.size(cdNu1.flat)
cdNu1 = cdNu1/((N.double) (N.sum(cdNu1))/npix) #turn into a density field


cdNu2 = CIC.getDemNu(mapNu2)
cdNu2 = N.double(cdNu2)
den2 = N.sum(cdNu2)/2000.**3
npix = N.size(cdNu2.flat)
cdNu2 = cdNu2/((N.sum(cdNu2))/npix) #turn into a density field

print N.var(cdNu1)
print N.var(cdNu2)
print N.var(N.log(cdNu1))
print N.var(N.log(cdNu2))

"""
Calculate Power spectrum
"""
print 'Power spectrum min mass'
Pow.pk(cdNu1, boxsize=2000., filename='PkNu17z0.txt')
out = N.loadtxt(open('./Output/PkNu17z0.txt'))
kmin = out[:,0]
Pkmin = out[:,1]
print 'Log power spectrum min mass'
Pow.pk(N.log(cdNu1), boxsize=2000., filename='PklogNu17z0.txt') 
out = N.loadtxt(open('./Output/PklogNu17z0.txt'))
klogmin = out[:,0]
Pklogmin = out[:,1]

print 'Power spectrum max mass'
Pow.pk(cdNu2, boxsize=2000., filename='PkNu30z0.txt')
out = N.loadtxt(open('./Output/PkNu30z0.txt'))
kmax = out[:,0]
Pkmax = out[:,1]
print 'Log power spectrum max mass'
Pow.pk(N.log(cdNu2), boxsize=2000., filename='PklogNu30z0.txt')
out = N.loadtxt(open('./Output/PklogNu30z0.txt'))
klogmax = out[:,0]
Pklogmax = out[:,1]


knqy = N.pi/((2000./512.))
print 'Nyquist frequency %f' %  (knqy)

indx = N.where(kmin>0.40)
kmin = N.delete(kmin, indx)
Pkmin = N.delete(Pkmin, indx)
kmax = N.delete(kmax, indx)
Pkmax = N.delete(Pkmax, indx)
klogmin = N.delete(klogmin, indx)
Pklogmin = N.delete(Pklogmin, indx)
klogmax = N.delete(klogmax, indx)
Pklogmax = N.delete(Pklogmax, indx)


"""
Cosmology
"""
h=0.67
omega_b_0=0.05
omega_M_0=0.32
sigma_8=0.83
omega_lambda_0=1.-omega_M_0

"""
Comparison with CAMB
"""

print 'AMP'
amp= 2.1265e-9
print amp

omega_nu_0 = 0.30/(93.14*h**2)
omega_c_0 = omega_M_0 - omega_b_0 - omega_nu_0

c = pt.Camb(hubble = 100.*h, ombh2 = omega_b_0*h**2,omnuh2=omega_nu_0*h**2, massless_neutrinos=0.046, massive_neutrinos=3, transfer_redshift = 0, do_nonlinear=1, omega_lambda=omega_lambda_0, omch2 = omega_c_0*h**2, scalar_amp = [amp])
c.run()
power = N.loadtxt(open('tmpCosmoPyCamb_matterpower.dat')) #Pk has units fof (Mpc/h)^3
kcambmax = N.array(power[:,0])
Pkcambmax = N.array(power[:,1])

omega_nu_0 = 0.17/(93.14*h**2)
omega_c_0 = omega_M_0 - omega_b_0 - omega_nu_0

c = pt.Camb(hubble = 100.*h, ombh2 = omega_b_0*h**2,omnuh2=omega_nu_0*h**2, massless_neutrinos=0.046, massive_neutrinos=3, transfer_redshift = 0, do_nonlinear=1, omega_lambda=omega_lambda_0, omch2 = omega_c_0*h**2, scalar_amp = [amp])
c.run()
power = N.loadtxt(open('tmpCosmoPyCamb_matterpower.dat')) #Pk has units fof (Mpc/h)^3
kcambmin = N.array(power[:,0])
Pkcambmin = N.array(power[:,1])


maxk = N.max(kmin)
mink = N.min(kmin)

plt.loglog(kmin, Pkmin, 'b', label="demnu mv=0.17eV z=0")
plt.loglog(kmax, Pkmax, 'g', label="demnu mv=0.30eV z=0")
plt.loglog(kcambmin, Pkcambmin, 'b--', label="camb mv=0.17eV z=0")
plt.loglog(kcambmax, Pkcambmax, 'g--', label="camb mv=0.30eV z=0")
plt.loglog(klogmin, Pklogmin, 'r', label="log demnu mv=0.17eV z=0")
plt.loglog(klogmax, Pklogmax, 'r--', label="log demnu mv=0.30eV z=0")
plt.axis([mink,maxk,1.e3, 4.e4])
plt.legend(loc=3)
plt.show()


deriv = ((N.log(Pkmax)-N.log(Pkmin)))/((N.log(0.30)-N.log(0.17)))
derivlog = ((N.log(Pklogmax)-N.log(Pklogmin)))/((N.log(0.30)-N.log(0.17)))
derivcamb = ((N.log(Pkcambmax)-N.log(Pkcambmin))/(N.log(0.00322096)-N.log(0.00182521)))
d

lnk = N.linspace(N.log(mink), N.log(maxk), 10000)
derivlnPk = N.interp(lnk, N.log(kmin), deriv)
derivlnPklog = N.interp(lnk, N.log(klogmin), derivlog)
derivlnPkcamb= N.interp(lnk, N.log(kcambmin), derivcamb)


plt.semilogx(kmin, deriv, 'b', label="simu")
plt.semilogx(kcambmin, derivcamb, 'r', label="CAMB")
plt.semilogx(klogmin, derivlog, 'g')
plt.axis([0.0001,10.,-0.25, 0.25])
plt.show()

miniter = 10
bin=100
Niter = ((int)((len(lnk)-miniter)/(1.*bin)))

F_G = N.zeros(Niter+1)
F_A = N.zeros(Niter+1)
SN = N.zeros(Niter+1)
klim = N.zeros(Niter+1)
F_G2 = N.zeros(Niter+1)
F_A2 = N.zeros(Niter+1)
F_Glog = N.zeros(Niter+1)

u=0
V=2000.**3.
for i in range(miniter, len(lnk), bin):
    F_G[u] = (V/(4*N.pi**2))*simps((N.exp(lnk[0:i]))**3*(derivlnPkcamb[0:i])**2, lnk[0:i])
    F_A[u] = (V/(4*N.pi**2))*simps((N.exp(lnk[0:i]))**3*(derivlnPkcamb[0:i]), lnk[0:i])

    F_G2[u] = (V/(4*N.pi**2))*simps((N.exp(lnk[0:i]))**3*(derivlnPk[0:i])**2, lnk[0:i])
    F_A2[u] = (V/(4*N.pi**2))*simps((N.exp(lnk[0:i]))**3*(derivlnPk[0:i]), lnk[0:i])
    F_Glog[u] = (V/(4*N.pi**2))*simps((N.exp(lnk[0:i]))**3*(derivlnPklog[0:i])**2, lnk[0:i])

  
    SN[u] = (V/(4*N.pi**2))*simps((N.exp(lnk[0:i]))**3, lnk[0:i])
    klim[u] = N.exp(lnk[i-1])
    u=u+1


sigma2min = (3.5*1e-6) #test
F = F_G - sigma2min*(F_A**2/(1.+sigma2min*SN))
F2 = F_G2 - sigma2min*(F_A2**2/(1.+sigma2min*SN))

print 'Gain ', F_Glog/F2

plt.loglog(klim, F, 'b')
plt.loglog(klim, F_G, 'r')
plt.loglog(klim, F2, 'b--')
plt.loglog(klim, F_G2, 'r--')
plt.loglog(klim, F_Glog, 'g--')
plt.show()
