#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np
from henke import cold_opacity
try:
    from snop import snop_opacity
    SNOP_PRESENT = True
except ImportError:
    SNOP_PRESENT = True
try:
    from hdf_interpolator import hdf_opacity
    OPGHDF5_PRESENT = True
except ImportError:
    OPGHDF5_PRESENT = True

def opacity_map( dens, tele, mat, nu, backend='snop', mat_names=None, tables=None):
    assert dens.shape == tele.shape, 'dens and tele arrays should be of the same shape'
    op = np.zeros(dens.shape+nu.shape)*np.nan
    if mat_names is not None:
        mat = mat.astype(np.int)
    for mat_el in np.unique(mat):
        mat_mask = (mat == mat_el)
        if mat_names is not None:
            mat_name = mat_names[mat_el]
        else:
            mat_name = mat_el
        if backend == 'snop':
            if SNOP_PRESENT:
                op[mat_mask] = snop_opacity(mat_name, dens[mat_mask],
                        np.ones(tele[mat_mask].shape),
                        #tele[mat_mask],
                        nu)
            else:
                raise ValueError("Trying to use SNOP backend, but pysnop doesn't seem to be installed")
        elif backend == 'henke':
            op[mat_mask] = cold_opacity(mat_name, dens[mat_mask], nu)
        elif backend == 'hdf5':
            if OPGHDF5_PRESENT:
                op[mat_mask] = hdf_opacity(tables[mat_name], dens[mat_mask],
                        tele[mat_mask],
                        nu)
            else:
                raise ValueError("Trying to use OpgHdf5 backend, but opacplot2 doesn't seem to be installed")

    return op
