#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright CNRS 2012, roman.yurchak@polytechnique.edu
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import sys
import os, os.path
import warnings

import numpy as np
from hedp.math.abel import abel 
from hedp.plasma_physics import critical_density


warnings.simplefilter("ignore")


def synthetic_phase_shift_cyl(d, lmbda):
    """
    Synthetic phase shift for a cylindrical simulation
    Parameters:
    -----------
      - d [dict]:  dictionary data containing fields (2D ndarrays), at least
          - r: radius array [cm] (must be equidistant)
          - z: height array [cm] (must be equidistant)
          - nele: electron density [cm⁻³]
      - lmbda: probe wavelenght [nm]

    Returns:
    --------
      - phase shift: 2D ndarray
    """
    Ne = d['nele']
    Nc = critical_density(lmbda)

    mu = np.sqrt(1 - Ne/Nc) - 1
    mu[Ne>Nc] = np.nan

    dr = np.diff(d['x'])[0,0]
    mu_dl = abel(mu, dr)/(2*lmbda*1e-9*1e2)
    
    return  np.ma.array(mu_dl, mask=np.isnan(mu_dl))






