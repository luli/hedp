#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import sys
import os, os.path
import hashlib
import warnings

import numpy as np
import tables
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import yt.mods
from yt.frontends.flash.data_structures import FLASHStaticOutput
import flash.output
import hedp
from hedp.math.abel import abel
import hedp.opacity.henke
from hedp.diags.xray import xray_filter_transmission, Kalpha_profile,\
                                ff_profile,ip_sensitivity

from time import time
from numexpr import evaluate



def fslice(filename, fields, resolution=800, cache="/dev/shm", bounds=None, method='nearest'):
    """
    Load and cache filelds from a FLASH output

    Parameters:
    ----------
      - filename [str]: path to filename
      - fileds [list]: a list of fields we want to load
      - resolution [int]: requested resolution
      - cache [bool or str]: cache the output somewhere
      - method:  interpolation to use in scipy.interpolate.griddata
    """
    # doing some homogenization

    filename = os.path.abspath(filename)
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
        #pf = yt.frontends.flash.data_structures.FLASHStaticOutput(filename)
        #print filename

        d = {}

        def _get_a_field(field):
            # fixing the stupid dot problem
            ffield = field
            if 'packmeshchkreadhdf5' not in pf:
               ffield = '{:<4s}'.format(field)
            R, Z, D = flash.output.slice(2, 0.0, ffield, pf, resolution=resolution, bounds=bounds, method=method)
            return R, Z, D

        D_unsorted = map(_get_a_field, fields)

        for key, val in zip(fields, D_unsorted):
            d[key] = val[2]
        d['z'] = val[1]
        d['r'] = val[0]
        d['x'] = d['r']
        d['y'] = d['z']


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
