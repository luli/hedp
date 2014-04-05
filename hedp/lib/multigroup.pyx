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
    struct gsl_function_struct:
        double (* function) (double x, void * params) nogil
        void * params

ctypedef gsl_function_struct gsl_function


cdef extern from "gsl/gsl_integration.h":
    int gsl_integration_qng (const gsl_function * f,
                         double a, double b,
                         double epsabs, double epsrel,
                         double *result, double *abserr,
                         size_t * neval) nogil


cdef int PLANCK_MEAN=1, ROSSELAND_MEAN=2, PLANCK_EMISS_MEAN=3


cdef double planck_weight_func_scalar(double u, void * offset) nogil:
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

    return u**3*exp(-u+(<double *> offset)[0])/(1 - exp(-u))



cdef double rosseland_weight_func_scalar(double u, void * offset) nogil:
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
    return u**4*exp(-u+(<double *> offset)[0])/(1-exp(-u))**2

cdef int mg_weight_base(double [:] U, int mode, double epsrel, int Ng, double [:] B) nogil:
    """
    Produce weights used in Planck and Rosseland computations

    Parameters
    ----------
     - U : ndarray, (Ng+1)
               group boundaries [eV] / plasma temperature [eV]
     - mode   : int
               PLANCK_MEAN or ROSSELAND_MEAN
     - epsrel:   : float 
               Input: requested relative error
               # Output: achieved maximum absolute error
    """
    cdef int Nbreak, i, kind_int, status
    cdef double epsabs=0.0, total_abserr=0.0, abserr=0.0
    cdef gsl_function weight_int
    cdef size_t neval


    if mode == PLANCK_MEAN:
        weight_int.function = &planck_weight_func_scalar
    elif mode == ROSSELAND_MEAN:
        weight_int.function = &rosseland_weight_func_scalar
    else:
        with gil:
            raise ValueError
    for i in range(Ng):
        weight_int.params = <void*> &U[i]
        status = gsl_integration_qng(&weight_int, U[i], U[i+1],
                             epsabs, epsrel, &B[i], &abserr, &neval)
        total_abserr = max(abserr, total_abserr)

    #eps[0] = total_abserr

    return 0



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



cpdef dict avg_mg_table(table, long [:] group_idx, list fields, verbose=False):
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
    cdef list out_opt = ['opp_mg', 'opr_mg', 'emp_mg', 'Bnu_p', 'Bnu_r', 'Bg_p', 'Bg_r']
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
    cdef double [:,:,:] opp_mg_nu, opr_mg_nu, emp_mg_nu
    cdef double [:] Bnu_p_t = np.empty(Np)
    cdef double [:] Bnu_r_t = np.empty(Np)
    cdef double [:, :] Bg_p, Bg_r, Bnu_p, Bnu_r
    cdef double epsrel = 1e-9

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
        opp_mg_nu = np.empty((Nr, Nt, Np))
    if out_flag[OPR_MG_ID]:
        opr_mg = np.empty((Nr, Nt, Ng))
        opr_mg_nu = np.empty((Nr, Nt, Np))
    if out_flag[EMP_MG_ID]:
        emp_mg = np.empty((Nr, Nt, Ng))
        emp_mg_nu = np.empty((Nr,Nt, Np))
    if out_flag[BG_P_ID]:
        Bg_p = np.empty((Nt, Ng))
    if out_flag[BG_R_ID]:
        Bg_r = np.empty((Nt, Ng))
    if out_flag[BNU_P_ID]:
        Bnu_p = np.empty((Nt, Np))
    if out_flag[BNU_R_ID]:
        Bnu_r = np.empty((Nt, Np))


    # computing planck mean opacity
    if out_flag[BNU_P_ID] or out_flag[BG_P_ID] or out_flag[OPP_MG_ID]:
        if out_flag[OPP_MG_ID]:
            opp_mg_nu = table['opp_mg'][:,:,:]
        with nogil:
            for j in range(Nt):
                for k in range(Np):
                    U[k] = nu[k]/temp[j]

                mg_weight_base(U, PLANCK_MEAN, epsrel, Ng, Bnu_p_t)

                if out_flag[BG_P_ID]: avg_subgroups(nu, Bnu_p_t, group_idx, Ng, Bg_p[j,:])
                if out_flag[BNU_P_ID]: 
                    for k in range(Np):
                        Bnu_p[j, k] = Bnu_p_t[k]
                if out_flag[OPP_MG_ID]:
                    for i in range(Nr):
                        avg_mg_spectra(U, opp_mg_nu[i,j,:], Bnu_p_t, group_idx,
                                PLANCK_MEAN, Ng, opp_mg[i,j,:])

    # computing rosseland mean opacity
    if out_flag[BNU_R_ID] or out_flag[BG_R_ID] or out_flag[OPR_MG_ID]:
        if out_flag[OPR_MG_ID]:
            opr_mg_nu = table['opr_mg'][:,:,:]
        with nogil:
            for j in range(Nt):
                for k in range(Np):
                    U[k] = nu[k]/temp[j]

                mg_weight_base(U, ROSSELAND_MEAN, epsrel, Ng, Bnu_r_t)

                if out_flag[BG_R_ID]: avg_subgroups(nu, Bnu_r_t, group_idx, Ng, Bg_r[j,:])
                if out_flag[BNU_R_ID]: 
                    for k in range(Np):
                        Bnu_r[j, k] = Bnu_r_t[k]

                if out_flag[OPR_MG_ID]:
                    for i in range(Nr):
                        avg_mg_spectra(U, opr_mg_nu[i,j,:], Bnu_r_t, group_idx,
                                ROSSELAND_MEAN, Ng, opr_mg[i,j,:])

    # computing planck mean emissivity
    if out_flag[EMP_MG_ID]:
        opp_mg_nu = table['opp_mg'][:,:,:]
        with nogil:
            for j in range(Nt):
                for k in range(Np):
                    U[k] = nu[k]/temp[j]

                mg_weight_base(U, PLANCK_MEAN, epsrel, Ng, Bnu_p_t)

                for i in range(Nr):
                    avg_mg_spectra(U, emp_mg_nu[i,j,:], Bnu_p_t, group_idx,
                            PLANCK_MEAN, Ng, emp_mg[i,j,:])

    if out_flag[OPP_MG_ID]: out['opp_mg'] = opp_mg
    if out_flag[OPR_MG_ID]: out['opr_mg'] = opr_mg
    if out_flag[EMP_MG_ID]: out['emp_mg'] = emp_mg

    if out_flag[BG_P_ID]: out['Bg_p'] = Bg_p
    if out_flag[BG_R_ID]: out['Bg_r'] = Bg_r

    if out_flag[BNU_P_ID]: out['Bnu_p'] = Bnu_p
    if out_flag[BNU_R_ID]: out['Bnu_r'] = Bnu_r
    return out










