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
#from libc.stdlib import size_t
from cython.parallel import parallel, prange
#from joblib import Parallel, delayed

cdef extern from "gsl/gsl_math.h":
    struct gsl_function:
        double (* function) (double x, void * params) nogil
        void * params


cdef extern from "gsl/gsl_integration.h":
    int gsl_integration_qng (const gsl_function * f,
                         double a, double b,
                         double epsabs, double epsrel,
                         double *result, double *abserr,
                         size_t * neval)


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


cpdef mg_weight_base(double [:] U, int mode, float epsrel_max, int Ng, double [:] B):
    """
    Produce weights used in Planck and Rosseland computations

    Parameters
    ----------
     - U : ndarray, (Ng+1)
               group boundaries [eV] / plasma temperature [eV]
     - mode   : int
               PLANCK_MEAN or ROSSELAND_MEAN
    """
    cdef int Nbreak, idx, kind_int

    if mode == PLANCK_MEAN:
        weight_int = planck_weight_int
    elif mode == ROSSELAND_MEAN:
        weight_int = rosseland_weight_int
    else:
        raise ValueError

    Ng = len(U)-1
    Nbreak =  np.argmin(np.abs(U.base-10000))
    pars = [0, 0.0, 1e-9, 10]
    if mode == PLANCK_MEAN:
        for idx in range(Nbreak):
            #_qagse(func,a,b,args,full_output,epsabs,epsrel,limit)
            B[idx], _, _ = _qagse(planck_weight_func_scalar,
                                    U[idx], U[idx+1],
                                    (U[idx],), *pars)
    elif mode == ROSSELAND_MEAN:
        for idx in range(Nbreak):
            B[idx], _, _ = _qagse(rosseland_weight_func_scalar,
                                    U[idx], U[idx+1],
                                    (U[idx],), *pars)

    B.base[Nbreak:Ng] = weight_int(U.base[Nbreak:Ng], U.base[Nbreak+1:Ng+1])
    return B.base, 0.0



cdef void avg_mg_spectra(double [:] U, double [:] op,\
            double [:] weight, long [:] group_idx, int mode, int Ng, double [:] opg) nogil:
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
    cdef int ig, i
    cdef double opg_g, norm_g, norm_i, res
    cdef double begin_group
    for ig in range(Ng):
        opg_g = 0.0
        norm_g = 0.0
        begin_group = U[group_idx[ig]]
        if mode == PLANCK_MEAN:
            for i in range(group_idx[ig], group_idx[ig+1]):
                norm_i = weight[i]*exp(begin_group -U[i])
                opg_g += op[i]*norm_i
                norm_g += norm_i
            opg[ig] = opg_g/norm_g
        elif mode == ROSSELAND_MEAN:
            for i in range(group_idx[ig], group_idx[ig+1]):
                norm_i = weight[i]*exp(begin_group -U[i])
                opg_g += norm_i/op[i]
                norm_g += norm_i
            opg[ig] = norm_g/opg_g
    return

cdef void avg_subgroups(double [:] U, double [:] weight, long [:] group_idx,
        int Ng, double [:] norm_g) nogil:
    """ Compute weights for subgroups """
    cdef int i, ig
    cdef double norm_i
    cdef double begin_group
    with nogil:
        for ig in range(Ng):
            norm_i = 0.0
            begin_group = U[group_idx[ig]]
            for i in range(group_idx[ig], group_idx[ig+1]):
                norm_i += weight[i]*exp(begin_group -U[i])
            norm_g[ig] = norm_i
    return



cpdef avg_mg_table(table, long [:] group_idx, list fields, verbose=False):
    """
    Average an opacity table for one 

    Parameters
    ----------
    table  :  (Nr,Nt,Np)  ndarray 
              dict or a pytable node shoule have following attributes: 'opp_mg', 'opr_mg', 'emp_mg', 'groups'
    group_idx :(Ng+1,)
             an array of indices indicating where to put group boundaries in nu
    fields    : list
             a list for output variables that must be a subset of 
             ['opp_mg', 'opr_mg', 'eps_mg', 'Bnu_p', 'Bnu_r', 'Bg_p', 'Bg_r']
    """
     
    cdef str key
    cdef dict out = {}
    cdef int i,j,k, mode
    cdef int verbose_flag=0
    cdef int OPP_MG_ID=0, OPR_MG_ID=1, EMP_MG_ID=2, BNU_P_ID=3, BNU_R_ID=4, BG_P_ID=5, BG_R_ID=6
    cdef list out_opt = ['opp_mg', 'opr_mg', 'eps_mg', 'Bnu_p', 'Bnu_r', 'Bg_p', 'Bg_r']
    cdef int [:] out_flag = np.zeros(7, dtype='int32')
    cdef int Nr, Nt, Np
    Nr, Nt, Np = table['opp_mg'].shape
    Np -= 1
    cdef int Ng = len(group_idx) - 1
    cdef double [:] nu = table['groups'][:]
    cdef double [:] U = np.empty(Np)
    cdef double [:] rho = table['dens'][:]
    cdef double [:] temp = table['temp'][:]
    cdef double [:,:,:] opp_mg, opr_mg, emp_mg
    cdef double [:,:] opp_mg_t, opr_mg_t, emp_mg_t
    cdef double [:] Bnu_p_t = np.empty(Np)
    cdef double [:] Bnu_r_t = np.empty(Np)
    cdef double [:, :] Bg_p, Bg_r

    # setting an flag array with values to be computed
    for i, key in enumerate(fields):
        k = 0
        for j in range(len(out_opt)):
            if key == out_opt[j]:
                out_flag[j] = 1
                k = 1
        if k==0:
            raise ValueError('Key {0} not in valid lists of keys!'.format(key))

    # Allocate all requested arrays
    if out_flag[OPP_MG_ID]:
        opp_mg = np.empty((Nr, Nt, Ng))
        opp_mg_t = np.empty((Nr, Ng))
    if out_flag[OPR_MG_ID]:
        opr_mg = np.empty((Nr, Nt, Ng))
        opr_mg_t = np.empty((Nr, Ng))
    if out_flag[EMP_MG_ID]:
        emp_mg = np.empty((Nr, Nt, Ng))
        emp_mg_t = np.empty((Nr, Ng))
    if out_flag[BG_P_ID]:
        Bg_p = np.empty((Nt, Ng))
    if out_flag[BG_R_ID]:
        Bg_r = np.empty((Nt, Ng))
    if out_flag[BNU_P_ID]:
        Bnu_p = np.empty((Nt, Np))
    if out_flag[BNU_R_ID]:
        Bnu_r = np.empty((Nt, Np))


    for j in range(Nt):
        for i in range(Np):
            U[i] = nu[i]/temp[j]
        # computing bases
        mg_weight_base(U, PLANCK_MEAN, 1e-9, Ng, Bnu_p_t)
        mg_weight_base(U, ROSSELAND_MEAN, 1e-9, Ng,  Bnu_r_t)

        if out_flag[BG_P_ID]: avg_subgroups(nu, Bnu_p_t, group_idx, Ng, Bg_p[j,:])
        if out_flag[BG_R_ID]: avg_subgroups(nu, Bnu_r_t, group_idx, Ng, Bg_r[j,:])
        if out_flag[BNU_P_ID]: Bnu_p[j] = Bnu_p_t
        if out_flag[BNU_R_ID]: Bnu_r[j] = Bnu_p_t
        #if verbose:
        #    print '{0}/{1}: {2:.1e} eV'.format(j, Nt, temp[j])

        if out_flag[OPP_MG_ID]:
            opp_mg_t = table['opp_mg'][:,j,:]
            with nogil:
                for i in range(Nr):
                    avg_mg_spectra(U, opp_mg_t[i], Bnu_p_t, group_idx,
                            PLANCK_MEAN, Ng, opp_mg[i,j,:])

        if out_flag[OPR_MG_ID]:
            opr_mg_t = table['opr_mg'][:,j,:]
            with nogil:
                for i in range(Nr):
                    avg_mg_spectra(U, opr_mg_t[i], Bnu_r_t, group_idx,
                            ROSSELAND_MEAN, Ng, opr_mg[i,j,:])
        if out_flag[EMP_MG_ID]:
            emp_mg_t = table['emp_mg'][:,j,:]
            with nogil:
                for i in range(Nr):
                    avg_mg_spectra(U, emp_mg_t[i], Bnu_p_t, group_idx,
                            PLANCK_MEAN, Ng, emp_mg[i,j,:])


    if out_flag[OPP_MG_ID]: out['opp_mg'] = opp_mg.base
    if out_flag[OPR_MG_ID]: out['opr_mg'] = opr_mg.base
    if out_flag[EMP_MG_ID]: out['emp_mg'] = emp_mg.base

    if out_flag[BG_P_ID]: out['Bg_p'] = Bg_p.base
    if out_flag[BG_R_ID]: out['Bg_r'] = Bg_r.base

    if out_flag[BNU_P_ID]: out['Bnu_p'] = Bnu_p.base
    if out_flag[BNU_R_ID]: out['Bnu_r'] = Bnu_r.base
    return










