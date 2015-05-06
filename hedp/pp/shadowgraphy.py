#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import sys
import os, os.path
import warnings

import numpy as np

from hedp.math.abel import abel 
from hedp.math.derivative import laplace
from hedp.plasma_physics import critical_density

#from numexpr import evaluate
import scipy.ndimage as nd


warnings.simplefilter("ignore")

def synthetic_shadowgraphy_cyl(d, lmbda, L=10, absorption=True, refraction=False):
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
      - lmbda: probe wavelenght [nm]
      - L: distance to detector [cm]
    Returns:
    --------
      - theta: 2D ndarray: refracted angle

    Source: Shlieren and shadowgraph techniques. G.Settles 
    """
    dI = np.ones(d['nele'].shape)
    Ne = d['nele']
    Nc = critical_density(lmbda)

    if refraction:
        # this doesn't seem to work so well

        Ref = 1 - np.sqrt(1 - Ne/Nc)
        Ref[Ne>Nc] = np.nan


        dr = np.diff(d['r'])[0,0]
        Ref_dl = abel(Ref, dr)
        d2Ref_dl = laplace(Ref_dl, dr)
        #pval =  (np.abs(np.gradient(Ref_dl)[0])/dr + np.abs(np.gradient(Ref_dl)[1])/dr)*180/np.pi
        #return d2Ref_dl
        dI0 = 1./(1. + d2Ref_dl)
        dI *= dI0
    if absorption:
        from scipy.constants import c
        nu_ei = 3e-6*d['nele']*d['Zbar']*10/d['tele']**(3./2)
        nu_ei = 1.0
        print(Nc)
        kernel = nu_ei*(Ne/Nc)/(c*(1 - Ne/Nc)**0.5)
        dr = np.diff(d['r'])[0,0]
        print(dr)
        kappa = kernel
        #kappa = abel(kernel, dr)
        dI0 = kappa#np.exp(-kappa)
        dI *= dI0
    return dI

