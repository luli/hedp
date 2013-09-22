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
import hedp.opacity.henke
from hedp.diags.xray import xray_filter_transmission, Kalpha_profile,\
                                ff_profile,ip_sensitivity

from time import time
from numexpr import evaluate


warnings.simplefilter("ignore")

def fslice(filename, fields, resolution=800, cache="/dev/shm", bounds=None):
    """
    Load and cache filelds from a FLASH output

    Parameters:
    ----------
      - filename [str]: path to filename
      - fileds [list]: a list of fields we want to load
      - resolution [int]: requested resolution
      - cache [bool or str]: cache the output somewhere
    """
    # doing some homogenization

    filename = os.path.abspath(filename)
    resolution = int(resolution)
    fields = sorted(fields)  
    cache_miss = True

    if cache:
        m = hashlib.md5()
        hash_in = "".join([filename,
                           "".join(fields),
                           str(resolution)])
        m.update(hash_in)
        hash_key = m.hexdigest()
        cache_file = os.path.join(cache, hash_key+'.hdf')
        if os.path.exists(cache_file):
            cache_miss = False


    if not cache or cache and cache_miss:
        pf = yt.mods.load(filename)

        d = {}

        def _get_a_field(field):
            # fixing the stupid dot problem
            ffield = field
            if 'packmeshchkreadhdf5' not in pf:
               ffield = '{:<4s}'.format(field)
            R, Z, D = flash.output.slice(2, 0.0, ffield, pf, resolution=resolution, bounds=bounds)
            return R, Z, D

        D_unsorted = map(_get_a_field, fields)

        for key, val in zip(fields, D_unsorted):
            d[key] = val[2]
        d['z'] = val[1]
        d['r'] = val[0]

    if cache:
        if cache_miss:
            f = tables.openFile(cache_file, 'w')
            for key in d:
                atom = tables.Atom.from_dtype(d[key].dtype)
                ds = f.createCArray(f.root, key, atom, d[key].shape)
                ds[:] = d[key]
            f.root._v_attrs.t = pf.current_time
            f.root._v_attrs.filename = filename
            f.close()
            d['t'] = pf.current_time
            d['filename'] = filename
        else:
            f = tables.openFile(cache_file, 'r')
            d = {}
            for key in fields + ['r', 'z']:
                d[key] = getattr(f.root, key)[:]
            d['t'] = f.root._v_attrs.t
            d['filename'] = f.root._v_attrs.filename
            f.close()
    else:
        d['t'] = pf.current_time
        d['filename'] = filename

    return hedp.Storage(d)

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






