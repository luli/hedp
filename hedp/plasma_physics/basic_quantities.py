#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np

from scipy.constants import e, c, m_e, epsilon_0, k
from scipy.constants import physical_constants

eV2K = physical_constants['electron volt-kelvin relationship'][0]

def critical_density(lmbda):
    """
    Compute critical density for a plasma
    
    Parameters:
    -----------
     - lmbda [ndarray or float]: wavelengt [nm]

    Returns:
     - Nc: [cm⁻³]
    """
    lmbda *= 1e-9
    omega = 2*np.pi*c/lmbda
    return 1e-6*epsilon_0*m_e*omega**2/e**2

def coulomb_logarithm(nele, znuc, tele):
    """
    Compute Coulomb logarithm

    Warning: untested implementation! Use  log_lambda instead!

    Parameters:
    -----------
     - nele: electron density in [cm⁻³]
     - znuc: nuclear charge
     - tele: mean temperature in [eV]

     Returns:
     --------
        ln Λ_spec
    """
    #Ne = nele*1e6 # cm⁻³ to m⁻³
    #tele = tele*eV2K
    #Lambda = (3.*(k*tele)**(3./2))/(4*(np.pi*Ne)**0.5 * (znuc*e)**3)

    return np.fmax(2, 24. - np.log(nele**0.5/tele))

def log_lambda(nele, Znuc, temp, spec='e', source='Atzeni2004'):
    """
    Compute the Coulomb logarithm for electrons or ions

    Parameters:
    -----------
     - nele: electron density in [cm⁻³]
     - znuc: nuclear charge
     - temp: mean temperature in [eV]
     - spec: specie i or e
     - source: literature source from where the formula is taken. 
             Possible options are:
               * Atzeni2004 :   The Physics of Inertial Fusion: BeamPlasma Interaction,
                                Hydrodynamics, Hot Dense Matter, 2004, page 367, section 10.9.1
               * Drake2006 :  High-Energy-Density Physics, Fundamentals, Inertial Fusion, and
                                Experimental Astrophysics, page 48 

     Returns:
     --------
        ln Λ_spec
    """
    if spec not in ['e', 'i']:
        raise ValueError("The 'spec' argiment {} must be either 'i' (ions) or 'e' (electons)".format(spec))
    if source == 'Atzeni2004':
        if spec == 'e':
            if not np.all(temp > 10):
                print('Warning: computing Ln Λ_e outside of its validity range Te > 10 eV !')
            return 7.1 - 0.5*np.log(nele*1e-21) + np.log(temp*1e-3)
        elif spec == 'i':
            if not np.all(temp < Znuc/2.*10e3):
                print('Warning: computing Ln Λ_e outside of its validity range Ti < 10 A keV !')
            return 9.2 - 0.5*np.log(nele*1e-21) + 1.5*np.log(temp*1e-3)
    elif source == 'Drake2006':
        if spec == 'e':
            print('Warning: validity domain for Ln Λ_e not defined in Drake (2006)!')
            return np.fmax(1, 24. - np.log(nele**0.5/temp))
        else:
            raise NotImplementedError('Ln Λ_i not defined in the Drake (2006) book!')
    else:
        raise NotImplementedError('Source = {} for calculating the Coulomb logarithm is not implemented!'.format(source))


def ei_collision_rate(nele, zbar,tele):
    """
    Compute the electron ion collision rate

    Parameters:
    -----------
     - nele: electron density in [cm⁻³]
     - zbar: mean ionization
     - tele: mean temperature in K

     Returns:
     --------
      ν_ei  [s^-1]
    """
    lnLambda  =  coulomb_logarithm(nele, zbar, tele)
    Ne = nele * 1e6
    nu_ei = 4./3 * (2*np.pi/m_e)**0.5 * (Ne*zbar*e**4*lnLambda)/(k*tele)**(3./2) / (4*np.pi*epsilon_0)**2
    return nu_ei


def ff_collision_frequency(nele, zbar,tele, lmbda):
    """
    Compute inverse bremsstrahlung coefficient
     
     ν_ib = (ne * ν_ei / nc) * 1/√(1 - ne/nc)

    Parameters:
    -----------
     - nele: electron density in [cm⁻³]
     - zbar: mean ionization
     - tele: mean temperature un eV
     - lmbda [ndarray or float]: wavelengt [nm]
    """
    nc = critical_density(lmbda)
    nu_ei = ei_collision_rate(nele, zbar,tele)
    nu_ff = (nele*nu_ei/nc)*(1/(1 - nele/nc)**0.5)
    nu_ff[nele>nc] = np.nan
    return nu_ff


def isentropic_sound_speed(Abar, Zbar, gamma, tele):
    """
    Compute the ion sound speed for an ideal gas (NRL formulary):

    Parameters:
    -----------
     - Abar: atomic number
     - Zbar: mean ionization
     - gamma: adiabatic index
     - tele: electron temperature [eV]
    Returns:
    -----------
     adiabatic sound speed [km/s]
    """
    return 9.79*(gamma*Zbar*tele/Abar)**0.5


def spitzer_conductivity(nele, tele, znuc, zbar):
    """
    Compute the Spitzer conductivity
    Parameters:
    -----------
     - nele [g/cm³]
     - tele [eV]
     - znuc: nuclear charge
     - zbar: mean ionization

    Returns:
    --------
     - Spitzer conductivity [Ω⁻¹.cm⁻¹]
    """

    lnLam = coulomb_logarithm(nele, znuc, tele)
    return 1./(1.03e-2*lnLam*zbar*(tele)**(-3./2))


def spitzer_conductivity2(nele, tele, znuc, zbar):
    """
    Compute the Spitzer conductivity
    Parameters:
    -----------
     - nele [g/cm³]
     - tele [eV]
     - znuc: nuclear charge
     - zbar: mean ionization

    Returns:
    --------
     - Spitzer conductivity [cm².s⁻¹]
    """

    lnLam = coulomb_logarithm(nele, znuc, tele)
    return 2e21*tele**(5./2)/(lnLam*nele*(zbar+1))
