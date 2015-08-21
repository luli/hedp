#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

from .cst import eV2K

import numpy as np
from scipy.constants import m_p, k as k_b
from scipy.optimize import brentq


def rh_shock_temperature(rho_2,  u_s, gamma, Z, Abar, T_min=0.01, T_max=1000):
    """
    Calculate the post-shock temperature from the shock velocity with Rankine-Hugoniot equations. Uses Thomas-Fermi formula for the plasma ionization.
    Source: "High-Energy-Density Physics Fundamentals, Inertial Fusion, and Experimental Astrophysics"
            R.P.Drake 2006, eqn. (4.20), page 116.

    Parameters
    ----------
      - rho_2 : (float) post-shock density [g/cm^3]
      - u_s   : (float) shock velocity (in the shock frame) [km/s]
      - gamma : (float) adiabatic index
      - Z     : (float) atomic number of the element (can be non integer for mixtures)
      - Abar  : (float) meat atomic mass
      - T_min, T_max: (floats) optional bounds for the temperature in eV
    
    Returns
    -------
      - temp: (float) temperature in eV
    
      
    """

    def solve_h(T_eV, rho_2,  u_s, gamma, Z, Abar):
        from hedp.eos.ionization import thomas_fermi_ionization
        Zbar = thomas_fermi_ionization(np.array([rho_2]), np.array([T_eV]), Z, Abar)
        return  T_eV  \
                 - Abar*m_p/(1 + Zbar)*(u_s*1e3)**2*\
                2*(gamma - 1)/((gamma  + 1)**2*(k_b*eV2K))

    res = brentq(solve_h, T_min, T_max,
                 args=(rho_2, u_s, gamma, Z, Abar))
    return res


