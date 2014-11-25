#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

from scipy.constants import N_A
from yt.data_objects.field_info_container import add_field

def _nion(field, data):
    return data['dens'] * data['sumy'] * N_A

add_field ('nion', function=_nion, take_log=True)

def _nele(field, data):
    return data['dens'] * data['ye'] * N_A

add_field ('nele', function=_nele, take_log=True)


def _abar(field, data):
    return 1.0 / data['sumy']

add_field ('abar', function=_abar, take_log=False)

def _zbar(field, data):
    return data['ye'] / data['sumy']

add_field ('zbar', function=_zbar, take_log=False)

def radiative_grid(r, opacity_file):
    """ 
    Parameters:
    -----------
      - r: ray orho ray
      - opacity file: patj to a hdf5 opacity file 
    """
    import numpy as np
    from opacplot2.opg_hdf5 import OpgHdf5
    op = OpgHdf5.open_file(opacity_file)
    groups =  op['groups'][:]
    Ng = len(groups) - 1
    vars_list = ['r{0:3}'.format(idx) for idx in range(Ng+1)]
    x = r['x']
    X, G = np.meshgrid(x, groups)
    R = np.zeros(X.shape)
    OP = np.zeros(X.shape)
    for idx in range(Ng):
        R[idx,:] = r['r{0:03}'.format(idx+1)]

    return X, G, R, OP


    


