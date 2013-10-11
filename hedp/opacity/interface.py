#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from henke import cold_opacity
try:
    from snop import snop_opacity
    SNOP_PRESENT = True
except ImportError:
    SNOP_PRESENT = True

def opacity_map( dens, tele, mat, nu, code='snop', mat_names=None):
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
        if code == 'snop':
            if SNOP_PRESENT:
                op[mat_mask] = snop_opacity(mat_name, dens[mat_mask],
                        np.ones(tele[mat_mask].shape),
                        #tele[mat_mask],
                        nu)
            else:
                raise ValueError("Trying to use SNOP backend, but pysnop doesn't seem to be installed")
        elif code == 'henke':
            op[mat_mask] = cold_opacity(mat_name, dens[mat_mask], nu)
    return op
