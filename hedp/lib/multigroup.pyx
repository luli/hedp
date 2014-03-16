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

from scipy.integrate import quadrature, simps, quad, fixed_quad

from libc.math cimport exp


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

      u**3 exp(offset-u)
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


def weight_base(groups, temp, kind='planck', epsrel_max=1e-9):
    """
    Produce weights used in Planck and Rosseland computations

    Parameters
    ----------
     - groups [ndarray]: group boundaries [eV]
     - temp [float]: plasma temperature [eV]
     - kind [str]: 'planck' or 'rosseland'
    """
    cdef int Ng, Nbreak, idx, kind_int
    cdef int PLANCK=1, ROSSELAND=2

    if kind == 'planck':
        weight_int = planck_weight_int
        kind_int = PLANCK
    elif kind == 'rosseland':
        weight_int = rosseland_weight_int
        kind_int = ROSSELAND
    else:
        raise ValueError

    Ng = len(groups)-1
    cdef double [:] B=np.zeros(Ng)
    cdef double [:] err = np.zeros(Ng)
    cdef double [:] U = np.array(groups/temp)
    Nbreak =  np.argmin(np.abs(U.base-100))
    for idx in range(Nbreak):
        if kind_int == PLANCK:
            B[idx], err[idx] = quad(planck_weight_func_scalar,
                                    a=U[idx], b=U[idx+1],
                                    args=(U[idx],), epsabs = 0)
        elif kind_int == ROSSELAND:
            B[idx], err[idx] = quad(rosseland_weight_func_scalar,
                                    a=U[idx], b=U[idx+1],
                                    args=(U[idx],), epsabs = 0)

    B.base[Nbreak:Ng] = weight_int(U.base[Nbreak:Ng], U.base[Nbreak+1:Ng+1])
    epsrel = (err.base/B.base).max()
    if epsrel > epsrel_max:
        raise ValueError
    return B.base, epsrel


