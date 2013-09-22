#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os, os.path
import warnings

from hedp.math.abel import abel 
from hedp.math.derivative import laplace
from hedp.plasma_physics import critical_density

#from numexpr import evaluate
import scipy.ndimage as nd


warnings.simplefilter("ignore")


def synthetic_shadowgraphy_cyl(d, lmbda, L=10):
    """
    Compute angle of refraction for a plasma assuming cylindrical symmetry on an axis
    orthogonal to the propagation axis.

    Parameters:
    -----------
      - d [dict]:  dictionary data containing fields (2D ndarrays), at least
          - r: radius array [cm] (must be equidistant)
          - z: height array [cm] (must be equidistant)
          - dens: solid density [g.cm⁻³]
          - Abar: mean atomic mass [g.mol⁻¹]
          - Zbar: mean ionization
      - lmbda: probe wavelenght
      - L: distance to detector [cm]
    Returns:
    --------
      - theta: 2D ndarray: refracted angle

    Source: Shlieren and shadowgraph techniques. G.Settles 
    """
    Ne = d['nele']
    Nc = critical_density(lmbda)

    Ref = np.sqrt(1 - Ne/Nc)
    Ref[Ne>Nc] = np.nan

    dr = np.diff(d['r'])[0,0]
    Ref_dl = abel(Ref, dr)
    d2Ref_dl = laplace(Ref_dl)
    dI0 = 1./(1. + L*d2Ref_dl)
    return dI0






