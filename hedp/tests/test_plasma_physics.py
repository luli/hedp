#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_allclose
import hedp.plasma_physics


def test_critical_density():
    """Checking the critical density calculation"""
    F0 = hedp.plasma_physics.critical_density(500)
    F1 = 1.1148542159362519e21/(500*1e-3)**2
    assert_allclose(F0, F1, atol=0.1)

def test_ei_collision_rate():
    Ne = 1e21 # 1/cc
    tele = 1000 # eV
    tele_K = tele*11640
    zbar = 3.5
    F0 = hedp.plasma_physics.ei_collision_rate(Ne,zbar, tele_K)

    ln_l =  hedp.plasma_physics.coulomb_logarithm(Ne,zbar, tele_K)
    F1 = 2.91e-6*1e21*3.5*ln_l/1000**(3./2)

    assert_allclose(F0, F1, rtol=1e-2)



