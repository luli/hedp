#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np
import hedp.rad as rad
from hedp.cst import eV2K
from scipy import constants as cst
from scipy.constants import physical_constants
from scipy.integrate import quad, quadrature

#import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

lmbda = np.logspace(np.log10(1.0e-9), np.log10(1.0e-3), 1000) # m
Te_eV = 1000.0

class TestPlanck():
    def test_nu_lambda(self):
        """
        Check that Bnu(nu, Te) * c/lmbda =  Blmbda(lmbda, Te)
        """
        nu = cst.c/lmbda
        x0 = rad.planck(lmbda*1.0e9, Te_eV, 'lambda')
        x1 = cst.c/lmbda**2 * rad.planck(nu, Te_eV, 'nu')
        assert_allclose(x0, x1)

    def test_iplanck(self):
        """
        Check that iBlambda(lmbda, Blambda(lmbda, x)) = x
        """
        Flux = rad.planck(lmbda*1.0e9, Te_eV, 'lambda')
        Flux = Flux*1.0e-9 #to W.m⁻².sr⁻¹.nm⁻¹
        Tout = rad.iplanck(lmbda*1.0e9, Flux)
        assert_allclose(Tout, Te_eV)

    def test_planck_integral(self):
        """
        Check that \int Bnu dθdφdν = σT⁴
        """

        Te = 200.0  # quad integration fails if the temperature is too big
        Te_eV = Te/eV2K

        sigma = physical_constants['Stefan-Boltzmann constant'][0]
        x0 = sigma*Te**4
        x1, x1_err = quad(lambda x: rad.planck(x, Te_eV, 'nu'), 0, 1.0e17)
        # 1e17 Hz max should be well enough for 200 eV
        # np.infinty doesn't work as integration bound (I probably don't
        # know enough about quad parameters)
        assert_allclose(x0, x1*np.pi)



    #sop.planck


def test_cooling_function():
    """ Testing from the example in
      High-Energy-Density Physics: Fundamentals, Inertial Fusion, page 256
    """
    from hedp.rad import planck_op2cooling_fn
    Zbar = 1.0
    A = 1.0
    rho = 10/6.02e+23
    tele = 10*11640
    kappa = 5e-37/rho
    res = planck_op2cooling_fn(kappa, rho, tele, A, Zbar)
    np.testing.assert_allclose(res, 1e-22, rtol=1e-1)




