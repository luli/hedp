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

    struct gsl_integration_struct:
        size_t limit
        size_t size
        size_t nrmax
        size_t i
        size_t maximum_level
        double *alist
        double *blist
        double *rlist
        double *elist
        size_t *order
        size_t *level

ctypedef gsl_integration_struct gsl_integration_workspace

cdef extern from "gsl/gsl_integration.h":
    gsl_integration_workspace * gsl_integration_workspace_alloc (const size_t n) nogil
    void gsl_integration_workspace_free (gsl_integration_workspace * w) nogil
    int gsl_integration_qag (const gsl_function * f,
                         double a, double b,
                         double epsabs, double epsrel, size_t limit,
                         int key,
                         gsl_integration_workspace * workspace,
                         double *result, double *abserr) nogil



cdef int PLANCK_WEIGHT=1, ROSSELAND_WEIGHT=2, UNIFORM_WEIGHT=3
cdef int ARITHMETIC_MEAN=4, GEOMETRIC_MEAN=5
cdef int QNG_INTEGRATOR=1000, QAG_INTEGRATOR=1001

cdef double SMALL_FLOAT=1.0e-8


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

cdef int mg_weight_base(double [:] U, int weight_type, double epsrel,
        double [:] B, int integrator_flag) nogil:
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
    cdef int Nbreak, i, kind_int, status, Ng
    cdef double epsabs=0.0, total_abserr=0.0, abserr=0.0
    cdef gsl_function weight_int
    cdef size_t neval, workspace_size=100, qag_limit=99, qag_key=6  # GSL_INTEG_GAUSS61 ( 61 point Gauss-Kronrod rule)
    cdef gsl_integration_workspace * qag_workspace

    Ng = U.shape[0]-1

    if weight_type == PLANCK_WEIGHT:
        weight_int.function = &planck_weight_func_scalar
    elif weight_type == ROSSELAND_WEIGHT:
        weight_int.function = &rosseland_weight_func_scalar
    elif weight_type == UNIFORM_WEIGHT:
        for i in range(Ng):
            B[i] = U[i+1] - U[i]
        return 0
    else:
        with gil: raise ValueError
    if integrator_flag == QNG_INTEGRATOR:
        for i in range(Ng):
            weight_int.params = <void*> &U[i]
            status = gsl_integration_qng(&weight_int, U[i], U[i+1],
                                 epsabs, epsrel, &B[i], &abserr, &neval)
            total_abserr = max(abserr, total_abserr)
            if status!=0:
                with gil: raise ValueError(
                        'Error in qng for frequency index {0} format is {1}'.format(i, status)) 
    elif integrator_flag == QAG_INTEGRATOR:
        qag_workspace = gsl_integration_workspace_alloc(workspace_size)
        for i in range(Ng):
            weight_int.params = <void*> &U[i]
            status = gsl_integration_qag(&weight_int, U[i], U[i+1],
                                 epsabs, epsrel, qag_limit, qag_key, qag_workspace,
                                 &B[i], &abserr)
            total_abserr = max(abserr, total_abserr)
            if status!=0:
                with gil: raise ValueError(
                        'Error in qag for frequency index {0} format is {1}'.format(i, status)) 
        gsl_integration_workspace_free(qag_workspace)


    else:
        with gil: raise ValueError
    return  0



cdef int avg_mg_spectra(double [:] U, double [:] op,\
            double [:] weight, long [:] group_idx,
            int weight_type, int mean_type, double [:] opg) nogil:
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
    cdef int ig, i, Ng
    cdef double opg_g, norm_g, norm_i, res
    cdef double begin_group
    Ng = group_idx.shape[0] - 1

    for ig in range(Ng):
        opg_g = 0.0
        norm_g = 0.0
        begin_group = U[group_idx[ig]]
        if mean_type == ARITHMETIC_MEAN:
            if weight_type == PLANCK_WEIGHT:
                for i in range(group_idx[ig], group_idx[ig+1]):
                    norm_i = weight[i]*exp(begin_group -U[i])
                    opg_g += op[i]*norm_i
                    norm_g += norm_i
            elif weight_type == UNIFORM_WEIGHT:
                for i in range(group_idx[ig], group_idx[ig+1]):
                    norm_i = weight[i]
                    opg_g += op[i]*norm_i
                    norm_g += norm_i
            else:
                with gil: raise ValueError('Wrong weight_type {0} with ARITHMETIC_MEAN!'.format(weight_type))
            opg[ig] = opg_g/norm_g
        elif mean_type == GEOMETRIC_MEAN:
            if weight_type == ROSSELAND_WEIGHT:
                for i in range(group_idx[ig], group_idx[ig+1]):
                    norm_i = weight[i]*exp(begin_group -U[i])
                    opg_g += norm_i/op[i]
                    norm_g += norm_i
            elif weight_type == UNIFORM_WEIGHT:
                for i in range(group_idx[ig], group_idx[ig+1]):
                    norm_i = weight[i]
                    opg_g += norm_i/op[i]
                    norm_g += norm_i
            else:
                with gil: raise ValueError('Wrong weight_type {0} with GEOMETRIC_MEAN!'.format(weight_type))
            opg[ig] = norm_g/opg_g
        else:
            with gil: raise ValueError('Unknown mean_type {0}!'.format(mean_type))

    return 0

cdef int avg_subgroups(double [:] U, double [:] weight, long [:] group_idx, double [:] norm_g) nogil:
    """ Compute weights for subgroups """
    cdef int i, ig, Ng
    cdef double norm_i
    cdef double begin_group
    Ng = group_idx.shape[0] - 1

    for ig in range(Ng):
        norm_i = 0.0
        begin_group = U[group_idx[ig]]
        for i in range(group_idx[ig], group_idx[ig+1]):
            norm_i += weight[i]*exp(begin_group -U[i])
        norm_g[ig] = norm_i
    return 0

cdef int compute_projection(double [:] U, long [:] group_idx,
        double [:] weight_nu, double [:] weight_g,
        double [:] op_mg, double [:] op_itp) nogil:
    cdef int i, ig, Np, Ng
    cdef double norm_i
    cdef double begin_group
    Np = U.shape[0] - 1
    Ng = group_idx.shape[0] - 1

    for ig in range(Ng):
        begin_group = U[group_idx[ig]]
        for i in range(group_idx[ig], group_idx[ig+1]):
            op_itp[i] = op_mg[ig]
    return 0


cdef int compute_error(double [:] nu, long [:] group_idx,
        double [:] op_nu, double [:] op_itp, double op_err, dict cost_pars, int Np) nogil:
    return 0

cdef int all_dens_masked(int [:] mask) nogil:
    cdef int N, i, status
    N = mask.shape[0]
    status = 0
    for i in range(N):
        status += mask[i]
    return <int> (status==0)


cpdef dict avg_mg_table(table, long [:] group_idx, list fields, 
            weight_pars={'opp': 'planck', 'opr': 'rosseland', 'emp': 'planck'}, epsrel=1e-9,
            integrator='qng',
            mask=None, cost_pars={}):
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
    cdef dict out = {}, ccost_pars = {}
    cdef int i,j,k, mode
    cdef int verbose_flag=0
    cdef int OPP_MG_ID=0, OPR_MG_ID=1, EMP_MG_ID=2, BNU_P_ID=3, BNU_R_ID=4,
    cdef int BG_P_ID=5, BG_R_ID=6, OPP_ITP_ID=7, OPR_ITP_ID=8, EMP_ITP_ID=9
    cdef int OPP_ERR_ID=10, OPR_ERR_ID=11, EMP_ERR_ID=12
    cdef list out_opt = ['opp_mg', 'opr_mg', 'emp_mg', 'Bnu_p', 'Bnu_r',
                         'Bg_p', 'Bg_r', 'opp_itp', 'opr_itp', 'emp_itp',
                         'opp_err', 'opr_err', 'emp_err']
    cdef dict weight_str2int = {'planck': PLANCK_WEIGHT, 'rosseland': ROSSELAND_WEIGHT,
                                'uniform': UNIFORM_WEIGHT}
    cdef int OPP_WEIGHT=PLANCK_WEIGHT, OPR_WEIGHT=ROSSELAND_WEIGHT, EMP_WEIGHT=PLANCK_WEIGHT
    cdef int [:] out_flag = np.zeros(13, dtype='int32')
    cdef int Nr, Nt, Np
    Nr, Nt, Np = table['opp_mg'].shape
    cdef int Ng = len(group_idx) - 1
    cdef double [:] nu = table['groups'][:]
    cdef double [:] U = np.empty(Np+1)
    cdef double [:] rho = table['dens'][:]
    cdef double [:] temp = table['temp'][:]
    cdef double [:,:,:] opp_mg, opr_mg, emp_mg
    cdef double [:,:,:] opp_nu, opr_nu, emp_nu
    cdef double [:,:,:] opp_itp, opr_itp, emp_itp
    cdef double [:,:] opp_err, opr_err, emp_err
    cdef int [:,:] mask_arr
    cdef double [:] Bnu_p_t = np.empty(Np)
    cdef double [:] Bnu_r_t = np.empty(Np)
    cdef double [:, :] Bg_p, Bg_r, Bnu_p, Bnu_r
    cdef double cepsrel
    cdef int integrator_flag

    if epsrel is not None:
        cepsrel = <double> epsrel
    else:
        cepsrel = 1.0e-9

    if mask is None:
        mask_arr = np.ones((Nr, Nt), dtype='int32')
    else:
        mask_arr = mask
    if cost_pars:
        ccost_pars = cost_pars

    if integrator=='qng':
        integrator_flag = QNG_INTEGRATOR
    elif integrator=='qag':
        integrator_flag = QAG_INTEGRATOR
    else:
        raise ValueError('Wrong value for the integrator should be one of qag, qng!')


    # setting an flag array with values to be computed
    for i, key in enumerate(fields):
        k = 0
        for j in range(len(out_opt)):
            if key == out_opt[j]:
                out_flag[j] = 1
                k = 1
        if k==0:
            raise ValueError('Key {0} not in valid lists of keys!'.format(key))

    if 'opp' in weight_pars:
        OPP_WEIGHT = weight_str2int[weight_pars['opp']]
    if 'opr' in weight_pars:
        OPR_WEIGHT = weight_str2int[weight_pars['opr']]
    if 'emp' in weight_pars:
        EMP_WEIGHT = weight_str2int[weight_pars['emp']]


    # Allocate all requested arrays
    if out_flag[OPP_ERR_ID]:
        out_flag[OPP_ITP_ID] = 1
    if out_flag[OPR_ERR_ID]:
        out_flag[OPR_ITP_ID] = 1
    if out_flag[EMP_ERR_ID]:
        out_flag[EMP_ITP_ID] = 1


    if out_flag[OPP_ITP_ID]:
        opp_itp = np.empty((Nr, Nt, Np))
        out_flag[OPP_MG_ID] = 1
        out_flag[BG_P_ID] = 1
    if out_flag[OPR_ITP_ID]:
        opr_itp = np.empty((Nr, Nt, Np))
        out_flag[OPR_MG_ID] = 1
        out_flag[BG_R_ID] = 1
    if out_flag[EMP_ITP_ID]:
        emp_itp = np.empty((Nr,Nt, Np))
        out_flag[EMP_MG_ID] = 1
        out_flag[BG_P_ID] = 1
    if out_flag[OPP_MG_ID]:
        opp_mg = np.empty((Nr, Nt, Ng))
        opp_nu = np.empty((Nr, Nt, Np))
    if out_flag[OPR_MG_ID]:
        opr_mg = np.empty((Nr, Nt, Ng))
        opr_nu = np.empty((Nr, Nt, Np))
    if out_flag[EMP_MG_ID]:
        emp_mg = np.empty((Nr, Nt, Ng))
        emp_nu = np.empty((Nr,Nt, Np))
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
            opp_nu = table['opp_mg'][:,:,:]
        with nogil:
            for j in range(Nt):
                if all_dens_masked(mask_arr[:,j]): continue
                for k in range(Np+1):
                    U[k] = nu[k]/temp[j]

                mg_weight_base(U, OPP_WEIGHT, cepsrel, Bnu_p_t, integrator_flag)

                if out_flag[BG_P_ID]: avg_subgroups(nu, Bnu_p_t, group_idx, Bg_p[j,:])
                if out_flag[BNU_P_ID]: 
                    for k in range(Np):
                        Bnu_p[j, k] = Bnu_p_t[k]
                if out_flag[OPP_MG_ID]:
                    for i in range(Nr):
                        if mask_arr[i,j]!=1: continue
                        avg_mg_spectra(U, opp_nu[i,j,:], Bnu_p_t, group_idx,
                                OPP_WEIGHT, ARITHMETIC_MEAN,  opp_mg[i,j,:])
                if out_flag[OPP_ITP_ID]:
                    for i in range(Nr):
                        if mask_arr[i,j]!=1: continue
                        compute_projection(U, group_idx,  Bnu_p_t, Bg_p[j,:],
                                opp_mg[i,j,:], opp_itp[i,j,:])


    # computing rosseland mean opacity
    if out_flag[BNU_R_ID] or out_flag[BG_R_ID] or out_flag[OPR_MG_ID]:
        if out_flag[OPR_MG_ID]:
            opr_nu = table['opr_mg'][:,:,:]
        with nogil:
            for j in range(Nt):
                if all_dens_masked(mask_arr[:,j]): continue
                for k in range(Np+1):
                    U[k] = nu[k]/temp[j]

                mg_weight_base(U, OPR_WEIGHT, cepsrel, Bnu_r_t, integrator_flag)

                if out_flag[BG_R_ID]: avg_subgroups(nu, Bnu_r_t, group_idx, Bg_r[j,:])
                if out_flag[BNU_R_ID]: 
                    for k in range(Np):
                        Bnu_r[j, k] = Bnu_r_t[k]

                if out_flag[OPR_MG_ID]:
                    for i in range(Nr):
                        if mask_arr[i,j]!=1: continue
                        avg_mg_spectra(U, opr_nu[i,j,:], Bnu_r_t, group_idx,
                                OPR_WEIGHT, GEOMETRIC_MEAN, opr_mg[i,j,:])
                if out_flag[OPR_ITP_ID]:
                    for i in range(Nr):
                        if mask_arr[i,j]!=1: continue
                        compute_projection(U, group_idx,  Bnu_r_t, Bg_r[j,:],
                                opr_mg[i,j,:], opr_itp[i,j,:])

    # computing planck mean emissivity
    if out_flag[EMP_MG_ID]:
        emp_nu = table['emp_mg'][:,:,:]
        with nogil:
            for j in range(Nt):
                if all_dens_masked(mask_arr[:,j]): continue
                for k in range(Np+1):
                    U[k] = nu[k]/temp[j]

                mg_weight_base(U, EMP_WEIGHT, cepsrel, Bnu_p_t, integrator_flag)

                if out_flag[EMP_ITP_ID]: avg_subgroups(nu, Bnu_p_t, group_idx, Bg_p[j,:])

                for i in range(Nr):
                    if mask_arr[i,j]!=1: continue
                    avg_mg_spectra(U, emp_nu[i,j,:], Bnu_p_t, group_idx,
                            EMP_WEIGHT, ARITHMETIC_MEAN, emp_mg[i,j,:])

                if out_flag[EMP_ITP_ID]:
                    for i in range(Nr):
                        if mask_arr[i,j]!=1: continue
                        compute_projection(U, group_idx,  Bnu_p_t, Bg_p[j,:],
                                emp_mg[i,j,:], emp_itp[i,j,:])

    if out_flag[OPP_MG_ID]: out['opp_mg'] = opp_mg.base
    if out_flag[OPR_MG_ID]: out['opr_mg'] = opr_mg.base
    if out_flag[EMP_MG_ID]: out['emp_mg'] = emp_mg.base

    if out_flag[OPP_ITP_ID]: out['opp_itp'] = opp_itp.base
    if out_flag[OPR_ITP_ID]: out['opr_itp'] = opr_itp.base
    if out_flag[EMP_ITP_ID]: out['emp_itp'] = emp_itp.base

    if out_flag[BG_P_ID]: out['Bg_p'] = Bg_p.base
    if out_flag[BG_R_ID]: out['Bg_r'] = Bg_r.base

    if out_flag[BNU_P_ID]: out['Bnu_p'] = Bnu_p.base
    if out_flag[BNU_R_ID]: out['Bnu_r'] = Bnu_r.base
    return out




cpdef int cellcentered_interpolate(double[:] gr0, double[:] val0, double [:] gr, double [:] val) nogil: 
    """ Take a cell centered variable val0 on a 1D grid gr0 and do a nearest neighbour interpolation onto 
    a different grid gr"""
    cdef int i, j
    cdef int Nx0, Nx
    cdef float nodeA, nodeB
    Nx0 = val0.shape[0]
    Nx =  val.shape[0]
    with gil:
        if gr0.shape[0] != Nx0+1:
            raise ValueError('Initial grid should have {0} points while only {1} were provided!'.format(Nx0+1, gr0.shape[0])) 
        if gr.shape[0] != Nx+1:
            raise ValueError('New grid should have {0} points while only {1} were provided!!'.format(Nx+1, gr.shape[0])) 
    j=0
    for i in range(Nx):
        nodeA, nodeB = gr[i], gr[i+1]
        if nodeA <= gr0[0] or nodeB >= gr0[Nx0]:
            # extrapolating
            val[i] = SMALL_FLOAT
        else:
            if gr0[j] <= nodeA < nodeB <= gr0[j+1]:
                val[i] = val0[j]
            elif gr0[j] <= nodeA <= gr0[j+1]<=nodeB:
                val[i] = 0.5*(val0[j] + val0[j+1])
                j=j+1
            else:
                with gil:
                    raise ValueError

    








