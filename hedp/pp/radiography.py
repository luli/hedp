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

warnings.simplefilter("ignore")


def synthetic_radiography_cyl(d, species, nu, spect_ip, hdf5_backend='pickle'):
    """
    Postprocess simulation to produce Xray
    
    Parameters:
    -----------
      - d [dict]:  data with all the fields
      - species [dict]: of species
      - nu [ndarray]: array of frequences [eV]
      - spect_ip [ndarray]: normalized spectra on ip
      - hdf5_backend: pytables or h5py

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
    op = {key: hedp.opacity.henke.cold_opacity(species[key], pd[key], nu, hdf5_backend) for key in species}


    op  = hedp.math.add_multiple(*[op[key] for key in species])
    

    tm = np.sum(spect_ip * np.exp(-op), axis=-1)*dnu
    return tm
