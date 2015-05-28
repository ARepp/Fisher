# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 10:39:02 2014

fisher.py
Calculates the Fisher information matrix for Gaussian and lognormal cases.

@author: Andrew
"""

import pt
import param

import numpy as np
from scipy.integrate import simps

global h
h = param.CosmoParams()['hubble']/100.
increment = 0.05
sig8_fid = 0.8159   # Planck 2015 (Table 4)


#--------------------------------------------------------------------------
# Obtains a power spectrum given specification of parameters (the others
# being assumed to have fiducial values). Note that if the bias differs
# from unity, this routine returns the galaxy power spectrum, not the
# dark matter power spectrum.
#--------------------------------------------------------------------------
def get_powerspec(z, linear=False, bias=1.0, normalize=False, sig8=sig8_fid,
                  **kwargs): 
    
    # If normalize, we normalize the spectrum to the value of sigma_8 given
    # at the top of this program.
    if normalize:
        # Call CAMB for the power spectrum, uses HALOFIT 1st at z=0
        c = pt.Camb(transfer_redshift = 0, do_nonlinear=0, **kwargs)
        c.run()

        # Obtain correct normalization
        R = 8  # Mpc
        sigz2 = pt.sigma2fromPk(c, R)
        ampl = c.cp.scalar_amp[0]
        f = float(sig8**2 / sigz2)

        # Run CAMB again with the correct amplitude and at redshift z
        kwargs['scalar_amp'] = [ampl*f]
    
    c = pt.Camb(transfer_redshift = [z], do_nonlinear=int(linear==False), **kwargs) 
    c.run()

    k = c.k; Pk = c.pk * bias**2
    
    return k, Pk
    
    

#-----------------------------------------------------------------------
# Lets me increment either floats or lists with the same command
#-----------------------------------------------------------------------
def gen_incr(target, increment):
    if type(target) is list:
        return map(lambda x:x+increment, target)
    else:
        return target + increment

#-----------------------------------------------------------------------
# Multiplies the increment by the fiducial target value, whether that value
# is a float or [the first element of] a list. If the fiducial value is
# zero, it simply returns the increment.
#-----------------------------------------------------------------------
def getincr(target, increment):
    if type(target) is list:
        if target[0] != 0:
            return float(target[0]) * increment
        else:
            return increment
            
    else:
        if target != 0:
            return target * increment
        else:
            return increment


#----------------------------------------------------------------------------
# Takes the arguments and truncates them all to the minimum length.
#----------------------------------------------------------------------------
def fix_length(*args):
    
    reflen = len(min(args, key=len))
    return_list = []
    for arg in args:
        return_list.append(arg[:reflen])
            
    return return_list

#----------------------------------------------------------------------------
# Correction for trispectrum. V (volume) must be in Gpc^3.
#
# Units:
#    V  : (Gpc/h)^3
#    Pk : (Mpc/h)^3
#----------------------------------------------------------------------------
#def trispect_alpha_sq(V, model='Neyrinck'):
def sig_sq_IS(V, Pk):

    # First convert V to (Mpc/h)^3
    V *= 1e9
    
    return 8 * Pk/V
            

#----------------------------------------------------------------------------
# Correction for supersurvey modes.
#
# Units:
#    V  : (Gpc/h)^3
#    ks : h/Mpc
#    Pk : (Mpc/h)^3 —note that this must be the linear power spectrum
#----------------------------------------------------------------------------
def SSM_sig_sq(V, ks, Pk):

    # V is in (Gpc/h)^3, but CAMB has k and Pk in units involving h and Mpc.
    # So first convert V to (Mpc/h)^2
    V *= 1e9   
    
    R = (3*V/(4*np.pi))**(1./3.)
    x = ks * R
    
    Wk = (3.*V)/(x*x*x)*(np.sin(x) - x * np.cos(x))

    lnk = np.linspace(np.log(np.min(ks)), np.log(np.max(ks)), 10000)
    P = np.interp(lnk, np.log(ks), Pk)
    W = np.interp(lnk, np.log(ks), Wk)

    sigma2V = (1./V**2)*(1./(2.*np.pi**2))*simps((np.exp(lnk))**3 * P * W**2, lnk)
    
    # For the global case (e.g., weak lensing) . . .
    # return sigma2V * (68./21.)**2
    # For the local case (e.g., galaxy surveys) . . .
    return sigma2V * (26./21.)**2
    
    

#----------------------------------------------------------------------------
# Returns the correction parameter sigma2_min to the covariance matrix,
# resulting from the trispectrum (intra-survey) and super-survey modes.
# Note that the resulting sig_sq_min is k-dependent and thus will be an
# array of the same size as ks.
#
# Units:
#    V         : (Gpc/h)^3
#    ks        : h/Mpc
#    Pk, linPk : (Mpc/h)^3
#----------------------------------------------------------------------------
def get_correc(V, ks, Pk, linPk):
  
    sig_sq_min = sig_sq_IS(V, Pk) + SSM_sig_sq(V, ks, linPk)
    return sig_sq_min


#----------------------------------------------------------------------------
# Calculates the partial derivative of the 3D power spectrum with respect to a
# given parameter; returns both the k's and the Pk. If log=True, the function
# differentiates ln P(k) rather than P(k). If logparam=True, the function
# differentiates with respect to the logarithm of the parameter.
#
# The addend term gets added to the logarithmic derivative in order to
# allow handling of non-standard parameters.
#
# Units:
#    sig  : (Mpc/h)^3
#----------------------------------------------------------------------------
def del_param(z, parameter, sig=0, log=False, logparam=False, linear=False,
              bias=1.0, normalize=False, addend=0):
    
    fiducial_params = param.CosmoParams()

    if parameter == 'sigma_8':
        norm_incr = getincr(sig8_fid, increment)
        upsig = gen_incr(sig8_fid, norm_incr)
        dnsig = gen_incr(sig8_fid, -norm_incr)
        k1, Pup = get_powerspec(z, linear=linear, bias=bias, sig8=upsig,
                                normalize=True, **fiducial_params)
        k2, Pdn = get_powerspec(z, linear=linear, bias=bias, sig8=dnsig,
                                normalize=True, **fiducial_params)
    elif not parameter in fiducial_params.keys():
        print 'Error--parameter %s not in defaultCosmoParamDict' % parameter
        raise ValueError
    else:
        if parameter == 'scalar_amp':
            normalize = False
        upparams = fiducial_params.copy(); dnparams = fiducial_params.copy()
        norm_incr = getincr(fiducial_params[parameter], increment)
        upparams[parameter] = gen_incr(fiducial_params[parameter], norm_incr)
        dnparams[parameter] = gen_incr(fiducial_params[parameter], -norm_incr)

        k1, Pup = get_powerspec(z, linear=linear, bias=bias,
                                normalize=normalize, **upparams)
        k2, Pdn = get_powerspec(z, linear=linear, bias=bias,
                                normalize=normalize, **dnparams)

    # CAMB sometimes returns power spectra (and thus ks) of differing
    # length; however, the shorter ks seem to be simply truncated versions
    # of the longer. The next line truncates the values to allow for
    # broadcasting.
    k, k2, Pup, Pdn = fix_length(k1, k2, Pup, Pdn)
        
    lnPup = np.log(Pup + sig)
    lnPdn = np.log(Pdn + sig)
    
    del_lnPk = (lnPup - lnPdn)/(2.*norm_incr) + addend

    # del y / del ln alpha = alpha * del y / del alpha
    if logparam:
        if parameter == 'sigma_8':
            del_lnPk = del_lnPk * sig8_fid
        else:
            del_lnPk = del_lnPk * fiducial_params[parameter]

    if log:       
        return k, del_lnPk
    else:
        Pk = (Pup + Pdn)/2. + sig
        return k, del_lnPk*Pk
            
    return k, del_param
        


       
#----------------------------------------------------------------------------
# Calculates a series of Fisher matrix entries for a series of k_maxes.
# The parameters specifying the matrix position are alpha and beta. It returns
# the vectors of k_max as well as the Fisher matrix entry for each of those
# k_max.
#
# The 'log' flags (alphlogparam, betalogparam) indicate whether or not the
# parameter indexing the Fisher matrix are the log of a CAMB parameter.
#       
# The 'mult' values (alphmult, betamult) get multiplied into the derivative
# to help account for nonstandard parameters (e.g., if a parameter were
# ln sigma_8^2).
#
# See function header of del_param for explanation of the 'add' values
# (alph_add, beta_add).
#
# For efficiency considerations, we also allow the user to pass in correc_sig_sq_min
# and Pk
#
# Units:
#    V in  : (Gpc/h)^3
#    n_bar : (h/Mpc)^3 i.e., galazies per h^-3 Mpc^3
#    k_min : h/Mpc     
#
#----------------------------------------------------------------------------
def get_Fish_kmax(V, z, alpha, beta, n_bar=np.inf, gauss=False, k_min=0,
                  linear=False, bias=1.0, normalize=False, special=None,
                  alphlogparam=False, betalogparam=False, alphmult=1.0,
                  betamult=1.0, alph_add=0.0, beta_add=0.0,
                  correc_sig_sq_min=None, Pk=None):

    # We assume shot noise
    sig_sq = 1./n_bar
    
    fiducial_params = param.CosmoParams()

    # For the parameter, calculate d(ln Pk)/d(parm)
    k, full_alpha_deriv = del_param(z, alpha, sig=sig_sq, log=True,
                                  logparam=alphlogparam, linear=linear,
                                  bias=bias, normalize=normalize,
                                  addend=alph_add)
    k, full_beta_deriv = del_param(z, beta, sig=sig_sq, log=True,
                                  logparam=betalogparam, linear=linear,
                                  bias=bias, normalize=normalize,
                                  addend=beta_add)
    full_alpha_deriv *= alphmult
    full_beta_deriv *= betamult
    
    # If for some reason the correct values have not gotten passed in, then
    # calculate them here.
    if Pk is None:    
        k, Pk = get_powerspec(z, linear=linear, bias=bias, normalize=normalize,
                          **fiducial_params)
    if (correc_sig_sq_min is None) and (gauss==False):
        k, linPk = get_powerspec(z, linear=True, bias=1., normalize=normalize,
                          **fiducial_params)
        # Note sig_sq_min assumes UNBIASED power spectra; thus must undo the
        # bias before passing it Pk
        correc_sig_sq_min = get_correc(V, k, Pk/(bias**2), linPk)
    
    min_index = np.searchsorted(k, k_min)    
    
    
    full_lnA_deriv = Pk/(Pk + sig_sq)
    
    # Sometimes CAMB returns lists of different length; this routine truncates
    # them to the shortest one to allow for broadcasting
    k, full_alpha_deriv, full_beta_deriv, full_lnA_deriv = \
        fix_length(k, full_alpha_deriv, full_beta_deriv, full_lnA_deriv)    
    
    
    # Now prepare and compute the matrix entry, using successive values of
    # k as k_max
    gauss_fishes = np.ones_like(k)
    nongauss_fishes = np.ones_like(k)
    # V is in (Gpc/h)^3, whereas CAMB puts k in h/Mpc, so volume must be in (Mpc/h)^3
    hMpc_V = V * 1e9
    pre = hMpc_V/(4 * np.pi**2)

    for i, k_max in enumerate(k):
        max_index = np.searchsorted(k, k_max)
        
        if (max_index < 2) or (max_index < min_index):
            gauss_fishes[i] = np.nan
            nongauss_fishes[i] = np.nan
            continue
        
        ks = k[min_index:max_index]
        alph_deriv = full_alpha_deriv[min_index:max_index]
        beta_deriv = full_beta_deriv[min_index:max_index]
        lnA_deriv = full_lnA_deriv[min_index:max_index]
        
        gauss_fishes[i] = pre *          \
               np.trapz(ks**3 * alph_deriv * beta_deriv, np.log(ks))
            
        if not gauss:
            # Now compute F^G_{\param \ln A}, F^G_{\ln A \param}, F^G_{\ln A \ln A} ;
            # we integrate in logspace.
            fish_alph_lnA=pre*np.trapz(ks**3 * alph_deriv * lnA_deriv, np.log(ks))
            fish_lnA_beta=pre*np.trapz(ks**3 * lnA_deriv * beta_deriv, np.log(ks))    
            fish_lnA_lnA =pre*np.trapz(ks**3 * lnA_deriv * lnA_deriv, np.log(ks))
            
            if special == 'alph_lnA':
                nongauss_fishes[i] = fish_alph_lnA
            elif special == 'lnA_beta':
                nongauss_fishes[i] = fish_lnA_beta
            elif special == 'lnA_lnA':
                nongauss_fishes[i] = fish_lnA_lnA
            elif special == 'minuend':
                nongauss_fishes[i] = (correc_sig_sq_min[i]) * \
                (fish_alph_lnA * fish_lnA_beta) / \
                (1. + correc_sig_sq_min[i] * fish_lnA_lnA)
            else:
                nongauss_fishes[i] = gauss_fishes[i] - \
                  (correc_sig_sq_min[i]) * (fish_alph_lnA * fish_lnA_beta) \
                  / (1. + correc_sig_sq_min[i] * fish_lnA_lnA)

    if gauss:
        return k, gauss_fishes
    else:
        return k, nongauss_fishes


#---------------------------------------------------------------------------
# Calculates the Fisher matrix for a series of increasing k_max. Returns
# the vectors of k_max and an array of matrices, the nth matrix (corresponding
# to k_max = nth k) being matr_array[:,:,n].
# 
# Required inputs: survey volume, redshift, and parameter list. Can optionally
# specify a Gaussian distribution, an average galaxy
# number density (n_bar), a non-zero k_min, a linear power spectrum, and
# galaxy bias that differs from unity.
#
# The parameter list must contain only CAMB parameters and/or 'sigma_8'. Pass
# in values for log_flags, multipliers, and addends to handle non-standard
# paramter formulations.
#
# Units:
#    V in  : (Gpc/h)^3
#    n_bar : (h/Mpc)^3 i.e., galazies per h^-3 Mpc^3
#    k_min : h/Mpc     
#---------------------------------------------------------------------------
def get_fisher_series(V, z, param_list, gauss=False, n_bar=np.inf,
                      k_min=0, linear=False, bias=1.0,
                      log_flags=None, multiplier=None, addend=None):
    
    if log_flags==None:
        log_flags=np.zeros(len(param_list)).astype(bool)
    if multiplier==None:
        multiplier=np.ones_like(param_list).astype(int)
    if addend==None:
        addend=np.zeros(len(param_list))

    normalize = ('sigma_8' in param_list)                  
    len_k = -1

    # Calculate the correction for non-Gaussian covariance, and pass it in
    # to avoid having to recalculate it for every parameter pair
    fiducial_params = param.CosmoParams()
    k, Pk = get_powerspec(z, linear=linear, bias=bias, normalize=normalize,
                          **fiducial_params)
    if gauss:
        correc_sig_sq_min = None
    else:
        k, linPk = get_powerspec(z, linear=True, bias=1., normalize=normalize,
                          **fiducial_params)
        # Note that get_correc requires V in (Gpc/h)^3. Note that sig_sq_min
        # assumes and UNBIASED power spectrum, thus must undo the Pk bias
        # before passing it in.
        correc_sig_sq_min = get_correc(V, k, Pk/(bias**2), linPk)

    for i, alpha in enumerate(param_list):
        for j, beta in enumerate(param_list):
            if j >= i:
                k, log_Fishes = get_Fish_kmax(V, z, alpha, beta, n_bar=n_bar,
                              gauss=gauss, k_min=k_min, linear=linear,
                              bias=bias, normalize=normalize,
                              alphlogparam=log_flags[i], alphmult=multiplier[i],
                              alph_add=addend[i],
                              betalogparam=log_flags[j], betamult=multiplier[j],
                              beta_add=addend[j],
                              correc_sig_sq_min=correc_sig_sq_min, Pk=Pk)

                # The first time through the loop, set up array. It will have one row
                # for each Fisher matrix position. The vector of ks that we return
                # will be this one that we get the first time
                if len_k < 0:
                    ret_k = k.copy()
                    len_k = len(k)
                    matr_array = np.zeros((len(param_list), len(param_list), len_k))
                # Make sure CAMB hasn't somehow returned a different vector of ks
                elif len_k < len(k):
                    print 'Truncating values for %s, %s from %i to %i entries.' % \
                        (alpha, beta, len(k), len_k)
                    log_Fishes = log_Fishes[:len_k]
                elif len_k > len(k):
                    print 'Padding values for %s, %s from %i to %i entries.' %\
                        (alpha, beta, len(k), len_k)
                    log_Fishes = np.pad(log_Fishes, [(0,len_k - len(k))], mode='maximum')
                matr_array[i, j, :] = log_Fishes

            else:
                matr_array[i, j, :] = matr_array[j, i, :]
                
    return ret_k, matr_array

