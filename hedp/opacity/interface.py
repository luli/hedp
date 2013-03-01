#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from henke import cold_opacity
from snop import snop_opacity

def opacity_map( dens, tele, mat, nu, code='snop'):
    assert dens.shape == tele.shape, 'dens and tele arrays should be of the same shape'
    op = np.zeros(dens.shape+nu.shape)*np.nan
    for mat_el in np.unique(mat):
        mat_mask = (mat == mat_el)
        if code == 'snop':
            op[mat_mask] = snop_opacity(mat_el, dens[mat_mask],
                    np.ones(tele[mat_mask].shape),
                    #tele[mat_mask],
                    nu)
        elif code == 'henke':
            op[mat_mask] = cold_opacity(mat_el, dens[mat_mask], nu)
    return op





