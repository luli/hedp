#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os, os.path
import hashlib
import warnings

import numpy as np
import tables
import yt.mods
import flash.output
import hedp
from hedp.math.abel import abel 
from hedp.math.derivative import laplace
from scipy.constants import N_A
from hedp.plasma_physics import critical_density

from numexpr import evaluate
import scipy.ndimage as nd


warnings.simplefilter("ignore")


def xray_pp_2d(d, species, nu, spect_ip):
    """
    Postprocess simulation to produce Xray
    
    Parameters:
    -----------
      - d [dict]:  data with all the fields
      - species [dict]: of species
      - nu [ndarray]: array of frequences [eV]
      - spect_ip [ndarray]: normalized spectra on ip

    Returns:
    --------
      - trans [ndarray]: transmissions
    """
    spect_ip = spect_ip[np.newaxis, np.newaxis, :]
    dnu = np.diff(nu)[0]
    nu = nu#[np.newaxis, np.newaxis, :]
    species_keys = sorted(species.keys())

    # projected density
    dr = np.diff(d['r'])[0,0]
    pd = {key: abel(d['dens']*d[key], dr) for key in species}
    # getting the opacity
    op = {key: hedp.opacity.henke.cold_opacity(species[key], pd[key], nu) for key in species}


    op  = hedp.math.add_multiple(*[op[key] for key in species])
    

    tm = np.sum(spect_ip * np.exp(-op), axis=-1)*dnu
    return tm

def simulated_shadowgraphy(d, lmbda, L=10):
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
    Ne = d['dens']*d['Na']*d['Zbar']/d['Abar']
    Nc = critical_density(lmbda)

    Ref = np.sqrt(1 - Ne/Nc)
    Ref[Ne>Nc] = np.nan

    dr = np.diff(d['r'])[0,0]
    Ref_dl = abel(Ref, dr)
    d2Ref_dl = laplace(Ref_dl)
    dI0 = 1./(1. + L*d2Ref_dl)
    return dI0






