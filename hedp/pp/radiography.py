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
import hedp
from hedp.math.abel import abel 
from hedp.diags.xray import xray_filter_transmission, Kalpha_profile,\
                                          ff_profile, ip_sensitivity


warnings.simplefilter("ignore")


def backlighter_setup(bl):
    nu = bl['nu']
    em_Kalpha = bl['Kalpha_ratio']*Kalpha_profile(bl['material'], bl['nu'], bl['Kalpha_width'])
    em_ff = (1-bl['Kalpha_ratio'])*ff_profile(bl['nu'], bl['bremshtrahlung_temp']) #
    op = em_ff + em_Kalpha
    transmission = xray_filter_transmission(nu, layers=bl['filters'])
    spec_ip = op*transmission*ip_sensitivity(nu)
    Norm = np.trapz(spec_ip, nu)
    spec_ip = spec_ip/Norm
    kalpha_nu0 = nu[np.argmax(em_Kalpha)]/1e3

    return nu, spec_ip, kalpha_nu0

def synthetic_radiography(rhol, species, nu, spect_ip, hdf5_backend='pickle', transform=None):
    """
    Postprocess simulation to produce Xray
    
    Parameters:
    -----------
      - d [dict]:  a dictionary with projected <rho L> values in cm.g.cm^-3 
      - species [dict]: of species
      - nu [ndarray]: array of frequences [eV]
      - spect_ip [ndarray]: normalized spectra on ip
      - hdf5_backend: pytables or h5py
      - transform: a optional function
          def transform(d, op):
              # do something with d, op
              return d, op

    Returns:
    --------
      - trans [ndarray]: transmissions
    """
    spect_ip = spect_ip[np.newaxis, np.newaxis, :]
    dnu = np.diff(nu)[0]
    nu = nu#[np.newaxis, np.newaxis, :]

    # projected density
    op = {key: hedp.opacity.henke.cold_opacity(species[key], rhol[key], nu, hdf5_backend) for key in species}

    print(op['targ'].shape)



    op_int  = hedp.math.add_multiple(*[op[key] for key in species])
    tm = np.sum(spect_ip * np.exp(-op_int), axis=-1)*dnu
    return tm

def synthetic_radiography_cyl(d, species, nu, spect_ip, hdf5_backend='pickle', transform=None):
    """
    Postprocess simulation to produce Xray
    
    Parameters:
    -----------
      - d [dict]:  data with all the fields
      - species [dict]: of species
      - nu [ndarray]: array of frequences [eV]
      - spect_ip [ndarray]: normalized spectra on ip
      - hdf5_backend: pytables or h5py
      - transform: a optional function
          def transform(d, op):
              # do something with d, op
              return d, op

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
    # getting the opacitoy
    op = {key: hedp.opacity.henke.cold_opacity(species[key], pd[key], nu, hdf5_backend) for key in species}
    #if 'tube' in op:
    #    op['tube'][550:700,:37] = 1e-9

    if transform is not None:
        d, op = transform(d, op)


    op_int  = hedp.math.add_multiple(*[op[key] for key in species])
    tm = np.sum(spect_ip * np.exp(-op_int), axis=-1)*dnu
    return tm
