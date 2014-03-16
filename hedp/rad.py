#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.
import numpy as np
from hedp.cst import eV2K, eV2Hz
from scipy import constants as cst
from scipy.integrate import simps, quad


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
        if type(x) is float:
            nu = x*1.0
        else:
            nu = x.copy()
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


def planck_mean(nu, op, temp):
    """
    Compute Planck mean opacity

    Parameters:
    -----------
     - nu [ndarray] photon energy [eV]
     - op [ndarray] opacity [cm⁻¹ or cm².g⁻¹]
     - temp [float] plasma temperature [eV]

    Returns:
    --------
      Planck mean opacity in the same units as op.
    """
    Bnu = planck(nu, temp, var='nu_eV')
    Bnu /= simps(Bnu, nu) # normalized Planckian

    return simps(Bnu*op, nu)


def rosseland_mean(nu, op, temp):
    """
    Compute Rosseland mean opacity

    Parameters:
    -----------
     - nu [ndarray] photon energy [eV]
     - op [ndarray] opacity [cm⁻¹ or cm².g⁻¹]
     - temp [float] plasma temperature [eV]

    Returns:
    --------
      Planck mean opacity in the same units as op.
    """
    Bnu = planck(nu, temp, var='nu_eV')
    iBnu = Bnu*nu # using the fact that ∂Bν/∂T= Bν * hν/(kT²)
    iBnu /= simps(iBnu, nu) # normalized Planckian

    return 1./simps(iBnu/op, nu)

def planck_mg2gray(groups, op, temp):
    """
    Average Planck multigroup opacity to gray

    Parameters:
    -----------
     - groups [ndarray] photon energy boundaries (Ng+1,) [eV]
     - op [ndarray] opacity (Ng,) [cm⁻¹ or cm².g⁻¹]
     - temp [float] plasma temperature [eV]

    Returns:
    --------
      Planck mean opacity in the same units as op.
    """
    def Bnu_fn(nu):
        return planck(nu, temp, var='nu_eV')
    Bnu_i = np.zeros(op.shape)
    for k in range(len(op)):
        Bnu_i[k] = quad(Bnu_fn, groups[k], groups[k+1])[0]
    return (Bnu_i*op).sum()/Bnu_i.sum()


def rosseland_mg2gray(groups, op, temp):
    """
    Average Rosseland mean opacity to gray

    Parameters:
    -----------
     - nu [ndarray] photon energy [eV]
     - op [ndarray] opacity [cm⁻¹ or cm².g⁻¹]
     - temp [float] plasma temperature [eV]

    Returns:
    --------
      Planck mean opacity in the same units as op.
    """
    def dBnu_fn(nu):
        # using the fact that ∂Bν/∂T= Bν * hν/(kT²)
        return nu*planck(nu, temp, var='nu_eV')

    dBnu_i = np.zeros(op.shape)
    for k in range(len(op)):
        dBnu_i[k] = quad(dBnu_fn, groups[k], groups[k+1])[0]

    return ((dBnu_i/op).sum()/dBnu_i.sum())**(-1)
