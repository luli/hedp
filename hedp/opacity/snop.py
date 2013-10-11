#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright CNRS 2012, roman.yurchak@polytechnique.edu
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import os
import sys

import numpy as np

from pysnop.snop_wrap import run as run_snop

from hedp import matdb


def snop_opacity(element, dens, tele, nu, full_output=False):
    """
    Parameters:
    -----------
     - element [str]: chemical element
     - dens [ndarray]: array of densities [g.cm⁻³]
     - dens [ndarray]: array of electron temperatures [eV]
     - nu [ndarray]: array of energies [eV]
          must be between [10 eV, 30 keV]
    Returns:
    --------
        opacity in cm⁻¹
    """

    db = matdb(element)
    s_args = dict(db.snop)
    s_args['arho'] = dens
    s_args['atele'] = np.fmax(tele, 1e-3) # just to be safe
    s_args['dsm'] = db.solid_dens
    for key, val in {'lte': False, 'lsff': True,
                    'lsbf': True, 'lsbb':True}.iteritems():
        s_args[key] = [val]*len(s_args['z'])
    s_args['fg'] =  np.array([np.min(nu)*1e-3, np.max(nu)*1e-3])
    s_args['nu'] = nu
    s_args['nout'] = 1
    d = run_snop(**s_args)
    if full_output:
        return d
    else:
        op = d['op_total'].copy()
        del d
        return op


if __name__ == '__main__':
    op = snop_opacity('Al', np.array([[0.1, 1.], [0.3, 0.5]]), 
            np.array([[10,20], [1., 3]]), [100, 102, 103], full_output=False)
    print nu
