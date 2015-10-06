#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np
from numpy.testing import assert_allclose
import hedp.plasma_physics
from scipy.constants import physical_constants

eV2K = physical_constants['electron volt-kelvin relationship'][0]



def test_critical_density():
    """Checking the critical density calculation"""
    F0 = hedp.plasma_physics.critical_density(500)
    F1 = 1.1148542159362519e21/(500*1e-3)**2
    assert_allclose(F0, F1, atol=0.1)

def test_ei_collision_rate():
    """
    Check nu_ei Spitzer
    """
    from hedp.eos.ionization import thomas_fermi_ionization
    from hedp.plasma_physics import collision_rate

    dens = 2.7 
    temp = 1.0e3
    Z = 13.
    Abar = 26.98

    Zbar = thomas_fermi_ionization(dens, temp, Z, Abar)

    nu_ei = collision_rate(dens, temp, Abar, Zbar, kind='ei', source='Atzeni2004')
    nu_ei_ref = [2e15] # value taken from 
    # "Coupling of detailed configuration kinetics and hydrodynamics in materials submitted
    # to x-ray free-electron-laser irradiation" Peyrusse 2012, Fig 1.
    assert_allclose(nu_ei, nu_ei_ref)


def test_coulomb_logarithm_old():
    "Checking lnΛ (Zeldovich p419)"
    F0 = hedp.plasma_physics.coulomb_logarithm(1e12, 1, 1e6/eV2K)
    F1 = np.array([5.97])
    #assert_allclose(F0, F1, rtol=1e-2)


def test_coulomb_logarithm():
    """Checking ln Λ  calculations """
    F0 = hedp.plasma_physics.log_lambda(1e19, 1, 1e2, spec='e', source='Atzeni2004')
    F1 = np.array([7.1])
    assert_allclose(F0, F1, rtol=1e-2)
