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

#def test_ei_collision_rate():
#    Ne = 1e21 # 1/cc
#    tele = 1000 # eV
#    tele_K = tele*11640
#    zbar = 3.5
#    F0 = hedp.plasma_physics.ei_collision_rate(Ne,zbar, tele_K)
#
#    ln_l =  hedp.plasma_physics.coulomb_logarithm(Ne,zbar, tele_K)
#    F1 = 2.91e-6*1e21*3.5*ln_l/1000**(3./2)
#
#    assert_allclose(F0, F1, rtol=1e-2)



def test_coulomb_logarithm():
    "Checking lnΛ (Zeldovich p419)"
    F0 = hedp.plasma_physics.coulomb_logarithm(1e12, 1, 1e6/eV2K)
    F1 = np.array([5.97])
    assert_allclose(F0, F1, rtol=1e-2)
