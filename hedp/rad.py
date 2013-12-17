#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.
import numpy as np
from hedp.cst import eV2K, eV2Hz
from scipy import constants as cst


def planck(x, tele, var='lambda'):
    """
    Planck's distribution

    Parameters:
    -----------
     - x [ndarray] photon energy array. The units are given by
                   the 'var' argument.
     - tele [ndarray] temperature [eV]
     - var [str] gives the variable for the photons energy.
             May be in following list:
                * 'lambda' : wavelenght [nm]
                * 'nu'     : frequency [Hz]
                * 'nu_eV'  : frequency [eV]

    Returns:
    --------
     If var == 'lambda':
        Blambda [ndarray] planck distribution W.m⁻².sr⁻¹.m⁻¹
     elif var == "nu":
        Bnu [ndarray]  W.m⁻².sr⁻¹.Hz⁻¹

    """
    if var not in ['lambda', 'nu', 'nu_eV']:
        raise ValueError("Input parameter var should be one of 'lambda', 'nu', 'nu_eV'!")

    tele = tele*eV2K # to K
    if var == 'lambda':  # Blambda
        lmbda = x/1.0e9  # to m
        return 2*cst.h*cst.c**2/(lmbda**5*(np.exp(cst.h*cst.c / (cst.k*lmbda*tele)) - 1.0))
    elif var.startswith('nu'): # Bnu
        nu = x
        if var == 'nu_eV':
            nu *= eV2Hz
        return 2*cst.h*nu**3/(cst.c**2 * (np.exp(cst.h*nu/(cst.k*tele)) - 1.0))


def iplanck(lmbda, Blmbda):
    """
    Inverse Planck's distribution in temperature

    Parameters:
    -----------
     - lmbda [float] photon wavelenght [nm]
     - Blmbda [ndarray] Spectral Radiance [W.m⁻².sr⁻¹.nm⁻¹]

    Returns:
    --------
     Te [ndarray]: black body temperature [eV]

    """
    Blmbda = Blmbda*1.0e9 #to W.m⁻².sr⁻¹.m⁻¹
    lmbda = lmbda/1.0e9
    a = cst.c*cst.h/(lmbda*cst.k*eV2K)
    b = 2*cst.h*cst.c**2/lmbda**5
    return a/np.log(1.0 + b/Blmbda)

