#!/usr/bin/python
# -*- coding: utf-8 -*-
from opacplot2.opg_hdf5 import OpgHdf5
import numpy as np
from scipy.interpolate import griddata


def hdf_opacity(path, dens, tele, nu):
    op = OpgHdf5.open_file(path)
    dens_mg = op['dens'][:]
    temp_mg = op['temp'][:]
    groups =  op['groups'][:]
    group_center = 0.5*(groups[1:]  + groups[:-1])
    Ng = len(groups) - 1
    D, T, G = np.meshgrid(dens_mg, temp_mg, group_center)
    Di, Gi = np.meshgrid(dens, nu)
    Ti, _ = np.meshgrid(tele, nu)

    points = np.concatenate((D.flatten()[:,np.newaxis], T.flatten()[:,np.newaxis], G.flatten()[:,np.newaxis]), axis=1)
    xi = np.concatenate((Di.flatten()[:,np.newaxis], Ti.flatten()[:,np.newaxis], Gi.flatten()[:,np.newaxis]), axis=1)
    data = op['opp_mg'][:].flatten()
    res = griddata(points, data, xi, method='nearest')
    return res.reshape(nu.shape+dens.shape)

