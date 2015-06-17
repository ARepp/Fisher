# Modules from bspline.py, in SciPy ver. 0.4.8

# Imports from scipy file
#
#import scipy.special
#from numpy import sarray, logical_and, asarray, pi, zeros_like, \
#     piecewise, array, arctan2, tan, zeros, arange, floor
#from numpy.core.umath import sqrt, exp, greater, less, equal, cos, add, sin, \
#     less_equal, greater_equal
#from spline import *      # C-modules
#from scipy.misc import comb

#gamma = scipy.special.gamma

from numpy import asarray, zeros_like, arctan2, zeros, arange, floor
from numpy.core.umath import sqrt, less, cos, add, sin

def cubic(x):
    """Special case of bspline.  Equivalent to bspline(x,3).
    """
    ax = abs(asarray(x))
    res = zeros_like(ax)
    cond1 = less(ax, 1)
    if cond1.any():
        ax1 = ax[cond1]
        res[cond1] = 2.0/3 - 1.0/2*ax1**2 * (2-ax1)
    cond2 = ~cond1 & less(ax, 2)
    if cond2.any():
        ax2 = ax[cond2]
        res[cond2] = 1.0/6*(2-ax2)**3
    return res

def _coeff_smooth(lam):
    xi = 1 - 96*lam + 24*lam * sqrt(3 + 144*lam)
    omeg = arctan2(sqrt(144*lam-1),sqrt(xi))
    rho = (24*lam - 1 - sqrt(xi)) / (24*lam)
    rho = rho * sqrt((48*lam + 24*lam * sqrt(3+144*lam))/xi)
    return rho,omeg

def _cubic_smooth_coeff(signal,lamb):
    rho, omega = _coeff_smooth(lamb)
    cs = 1-2*rho*cos(omega) + rho*rho
    K = len(signal)
    yp = zeros((K,),signal.dtype.char)
    k = arange(K)
    yp[0] = _hc(0,cs,rho,omega)*signal[0] + \
            add.reduce(_hc(k+1,cs,rho,omega)*signal)

    yp[1] = _hc(0,cs,rho,omega)*signal[0] + \
            _hc(1,cs,rho,omega)*signal[1] + \
            add.reduce(_hc(k+2,cs,rho,omega)*signal)

    for n in range(2,K):
        yp[n] = cs * signal[n] + 2*rho*cos(omega)*yp[n-1] - rho*rho*yp[n-2]

    y = zeros((K,),signal.dtype.char)

    y[K-1] = add.reduce((_hs(k,cs,rho,omega) + _hs(k+1,cs,rho,omega))*signal[::-1])
    y[K-2] = add.reduce((_hs(k-1,cs,rho,omega) + _hs(k+2,cs,rho,omega))*signal[::-1])

    for n in range(K-3,-1,-1):
        y[n] = cs*yp[n] + 2*rho*cos(omega)*y[n+1] - rho*rho*y[n+2]

    return y

def _cubic_coeff(signal):
    zi = -2 + sqrt(3)
    K = len(signal)
    yplus = zeros((K,),signal.dtype.char)
    powers = zi**arange(K)
    yplus[0] = signal[0] + zi*add.reduce(powers*signal)
    for k in range(1,K):
        yplus[k] = signal[k] + zi*yplus[k-1]
    output = zeros((K,),signal.dtype)
    output[K-1] = zi / (zi-1)*yplus[K-1]
    for k in range(K-2,-1,-1):
        output[k] = zi*(output[k+1]-yplus[k])
    return output*6.0

def cspline1d(signal,lamb=0.0):
    """Compute cubic spline coefficients for rank-1 array.

    Description:

      Find the cubic spline coefficients for a 1-D signal assuming
      mirror-symmetric boundary conditions.   To obtain the signal back from
      the spline representation mirror-symmetric-convolve these coefficients
      with a length 3 FIR window [1.0, 4.0, 1.0]/ 6.0 .

    Inputs:

      signal -- a rank-1 array representing samples of a signal.
      lamb -- smoothing coefficient (default = 0.0)

    Output:

      c -- cubic spline coefficients.
    """
    if lamb != 0.0:
        return _cubic_smooth_coeff(signal,lamb)
    else:
        return _cubic_coeff(signal)


def cspline1d_eval(cj, newx, dx=1.0, x0=0):
    """Evaluate a spline at the new set of points.
    dx is the old sample-spacing while x0 was the old origin.

    In other-words the old-sample points (knot-points) for which the cj
    represent spline coefficients were at equally-spaced points of

    oldx = x0 + j*dx  j=0...N-1

    N=len(cj)

    edges are handled using mirror-symmetric boundary conditions.
    """
    newx = (asarray(newx)-x0)/float(dx)
    res = zeros_like(newx)
    if (res.size == 0):
        return res
    N = len(cj)
    cond1 = newx < 0
    cond2 = newx > (N-1)
    cond3 = ~(cond1 | cond2)
    # handle general mirror-symmetry
    res[cond1] = cspline1d_eval(cj, -newx[cond1])
    res[cond2] = cspline1d_eval(cj, 2*(N-1)-newx[cond2])
    newx = newx[cond3]
    if newx.size == 0:
        return res
    result = zeros_like(newx)
    jlower = floor(newx-2).astype(int)+1
    for i in range(4):
        thisj = jlower + i
        indj = thisj.clip(0,N-1) # handle edge cases
        result += cj[indj] * cubic(newx - thisj)
    res[cond3] = result
    return res
