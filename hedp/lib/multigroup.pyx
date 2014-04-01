#!/usr/bin/python
# -*- coding: utf-8 -*-
# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
## Copyright CNRS 2012
## Roman Yurchak (LULI)
## This software is governed by the CeCILL-B license under French law and
## abiding by the rules of distribution of free software.

import numpy as np
cimport numpy as np
from os import path
import ctypes
import sys

from scipy.integrate import quad, quadpack #quadrature, simps, quad, fixed_quad
from scipy.integrate._quadpack import _qagse

from libc.math cimport exp
#from cython.parallel import parallel, prange
#from joblib import Parallel, delayed

cdef int PLANCK_MEAN=1, ROSSELAND_MEAN=2, PLANCK_EMISS_MEAN=3


cpdef double planck_weight_func_scalar(double u, double offset) nogil:
    """
    Planck weighting function

    Parameters:
    -----------
     - nu [ndarray]: photon energy [eV]
     - temp [float]: plasma temperature [eV]
    The result is multiplied by exp(+offset) for normalization

    Returns
    -----------

     u**3 exp(offset)
     ----------------    with u = nu/temp
        exp(u) - 1
    """

    return u**3*exp(-u+offset)/(1 - exp(-u))



cpdef double rosseland_weight_func_scalar(double u, double offset) nogil:
    """
    Rosseland weighting function

    Parameters:
    -----------
     - nu [ndarray]: photon energy [eV]
     - temp [ndarray]: plasma temperature [eV]
    The result is multiplied by exp(+offset) for normalization

    Returns
    -----------

      u**4 exp(offset-u)
     ------------------    with u = nu/temp
        (1 - exp(-u))²
    """
    return u**4*exp(-u+offset)/(1-exp(-u))**2

def planck_weight_int(a, b):
    """
    Integrate Planck weighting function in the approximation u>>1.
    The result is multiplied by exp(+a) for normalization
    """
    res = 0
    for idx, bd in enumerate([a,b]):
        sign = np.sign((idx - 0.5))
        res += -sign*(bd**3 + 3*bd**2 + 6*bd + 6)*np.exp(a-bd)
    return res

def rosseland_weight_int(a, b):
    """
    Integrate Rosseland weighting function in the approximation u>>1.
    The result is multiplied by exp(+a) for normalization
    """
    res = 0
    for idx, bd in enumerate([a,b]):
        sign = np.sign((idx - 0.5))
        res += -sign*(bd**4 + 4*bd**3 + 12*bd**2 +24*bd + 24)*np.exp(a-bd)
    return res


cpdef mg_weight_base(double [:] U, mode, epsrel_max=1e-9):
    """
    Produce weights used in Planck and Rosseland computations

    Parameters
    ----------
     - U : ndarray, (Ng+1)
               group boundaries [eV] / plasma temperature [eV]
     - mode   : int
               PLANCK_MEAN or ROSSELAND_MEAN
    """
    cdef int Ng, Nbreak, idx, kind_int

    if mode == PLANCK_MEAN:
        weight_int = planck_weight_int
    elif mode == ROSSELAND_MEAN:
        weight_int = rosseland_weight_int
    else:
        raise ValueError

    Ng = len(U)-1
    cdef double [:] B=np.zeros(Ng)
    cdef double [:] err = np.zeros(Ng)
    Nbreak =  np.argmin(np.abs(U.base-10000))
    pars = [0, 0.0, 1e-9, 10]
    if mode == PLANCK_MEAN:
        for idx in range(Nbreak):
            #_qagse(func,a,b,args,full_output,epsabs,epsrel,limit)
            B[idx], err[idx], _ = _qagse(planck_weight_func_scalar,
                                    U[idx], U[idx+1],
                                    (U[idx],), *pars)
    elif mode == ROSSELAND_MEAN:
        for idx in range(Nbreak):
            B[idx], err[idx], _ = _qagse(rosseland_weight_func_scalar,
                                    U[idx], U[idx+1],
                                    (U[idx],), *pars)

    B.base[Nbreak:Ng] = weight_int(U.base[Nbreak:Ng], U.base[Nbreak+1:Ng+1])
    epsrel = (err.base/B.base).max()
    if epsrel > epsrel_max:
        raise ValueError
    return B.base, epsrel



cpdef avg_mg_spectra(double [:] U, double [:] op,\
                        double [:] weight, long [:] group_idx, int mode):
    """
    Average one spectra

    Parameters
    ----------
    U     :  (Np+1,)
              photon energy normalized array
    op     :  (Np)  ndarray 
              opacity data
    weight :  (Np,)
              weights for averaging
    group_idx :(Ng+1,)
             an array of indices indicating where to put group boundaries in nu
    mode   : int
               PLANCK_MEAN, ROSSELAND_MEAN, EPS_MEAN
    """
    cdef int Ng
    Ng = len(group_idx) - 1
    cdef int i, ig
    cdef double opg_g, norm_g, norm_i
    cdef double [:] opg = np.empty(Ng)
    cdef double begin_group
    with nogil:
        for ig in range(Ng):
            opg_g = 0.0
            norm_g = 0.0
            begin_group = U[group_idx[ig]]
            if mode == PLANCK_MEAN:
                for i in range(group_idx[ig], group_idx[ig+1]):
                    norm_i = weight[i]*exp(begin_group -U[i])
                    opg_g +=  op[i]*norm_i
                    norm_g += norm_i
                opg[ig] = opg_g/norm_g
            elif mode == ROSSELAND_MEAN:
                for i in range(group_idx[ig], group_idx[ig+1]):
                    norm_i = weight[i]*exp(begin_group -U[i])
                    opg_g +=  norm_i/op[i]
                    norm_g += norm_i
                opg[ig] = norm_g/opg_g
            #elif mode == PLANCK_EMISS_MEAN:
            #    for i in range(group_idx[ig], group_idx[ig+1]):
            #        norm_i = weight[i]*exp(begin_group -nu[i])
            #        opg_g  +=  op[i]
            #        norm_g +=  norm_i
            #    opg[ig] = opg_g/norm_g
    return opg

cpdef avg_subgroups(double [:] U, double [:] weight, long [:] group_idx):
    """ Compute weights for subgroups """
    cdef int Ng
    Ng = len(group_idx) - 1
    cdef int i, ig
    cdef double norm_i
    cdef double [:] norm_g = np.empty(Ng)
    cdef double begin_group
    with nogil:
        for ig in range(Ng):
            norm_i = 0.0
            begin_group = U[group_idx[ig]]
            for i in range(group_idx[ig], group_idx[ig+1]):
                norm_i += weight[i]*exp(begin_group -U[i])
            norm_g[ig] = norm_i
    return norm_g



def avg_mg_table(table, group_idx, emp=False, debug=False, verbose=False):
    """
    Average an opacity table for one 

    Parameters
    ----------
    table  :  (Nr,Nt,Np)  ndarray 
              dict or a pytable node shoule have following attributes: 'opp_mg', 'opr_mg', 'emp_mg', 'groups'
    group_idx :(Ng+1,)
             an array of indices indicating where to put group boundaries in nu
    """
     
    cdef int i,j,k, mode
    cdef int Nr, Nt, Np
    nu = table['groups'][:]
    U = np.empty(nu.shape)
    rho = table['dens'][:]
    temp = table['temp'][:]
    Nr, Nt, Np = len(rho), len(temp), len(nu)-1
    Ng = len(group_idx) - 1

    opp_mg = np.empty((Nr, Nt, Ng))
    opr_mg = np.empty((Nr, Nt, Ng))

    if debug:
        Bg_p = np.empty((Nt, Ng))
        Bg_r = np.empty((Nt, Ng))
        Bnu_p_all = []
        Bnu_r_all = []
    if emp:
        emp_mg = np.empty((Nr, Nt, Ng))

    for j in range(Nt):
        U[:] = nu/temp[j]
        Bnu_p, Bnu_p_err = mg_weight_base(U, PLANCK_MEAN, epsrel_max=1e-9)
        Bnu_r, Bnu_p_err = mg_weight_base(U, ROSSELAND_MEAN, epsrel_max=1e-9)
        if debug:
            Bg_p[j,:] = avg_subgroups(nu, Bnu_p, group_idx)
            Bg_r[j,:] = avg_subgroups(nu, Bnu_r, group_idx)
            Bnu_p_all.append(Bnu_p)
            Bnu_r_all.append(Bnu_p)
        if verbose:
            print '{0}/{1}: {2:.1e} eV'.format(j, Nt, temp[j])

        for i in range(Nr):
            opp_mg[i,j,:] = avg_mg_spectra(U, table['opp_mg'][i,j], Bnu_p, group_idx, PLANCK_MEAN)
            opr_mg[i,j,:] = avg_mg_spectra(U, table['opr_mg'][i,j], Bnu_r, group_idx, ROSSELAND_MEAN)

        if emp:
            for i in range(Nr):
                emp_mg[i,j,:] = avg_mg_spectra(U, table['emp_mg'][i,j], Bnu_p, group_idx, PLANCK_MEAN)


    out = {'opp_mg': opp_mg, 'opr_mg': opr_mg}
    if emp:
        out['emp_mg'] =  emp_mg
    if debug:
        out['Bg_p'] = Bg_p
        out['Bg_r'] = Bg_r
        out['Bnu_p'] = np.array(Bnu_p_all)
        out['Bnu_r'] = np.array(Bnu_r_all)

    return out










