#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import cumtrapz

from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy import constants
from scipy.constants import physical_constants

# everything is in SI units

def planck(tele, nu=None, lmbda=None):
    """
    Plot the Planck distribution

    Parameters:
    -----------
     - tele [ndarray] temperature [eV]
     - lmbda [ndarray] photon wavelenght [nm] (optional)
     - nu [ndarray] photon energy [eV] (optional)

    Returns:
    --------
     I0 [ndarray] planck distribution

    """
    if nu is None and lmbda is None:
        raise ValueError('You should specify either nu or lmbda!')
    elif nu is not None and lmbda is not None:
        raise ValueError('Parameters nu and lmbda cannot be specified at the same time!')
    elif nu is None:
        nu = physical_constants['inverse meter-electron volt relationship'][0]*1.0e9/lmbda

    Bnu  =  2*(nu*constants.e)**3/((constants.h*constants.c)**2 *
                    ( np.exp(nu / tele) - 1.  ))
    return Bnu


def compute_emiss(I0, op, dx=1, axis=0, _sum=False):
    if axis==0:
        op = op[::-1]
    elif axis==1:
        op = op[:,::-1]
    #if mask is not None:
    #    # this part of simulation is opaque
    #    op[mask] = 1e3
   

    cop = cumtrapz(op, axis=axis, dx=dx)
    if axis==0:
        cop = np.vstack((np.zeros((1,cop.shape[1])), cop))
    elif axis==1:
        cop = np.hstack((np.zeros((cop.shape[0], 1)), cop))
    if axis==0:
        cop = cop[::-1]
    elif axis==1:
        cop = cop[:,::-1]
    em = I0*np.exp(-cop)
    if _sum:
        return np.sum(em, axis=axis)
    else:
        return em

def polar2cartesian(r, t, grid, x, y, order=3):

    X, Y = np.meshgrid(x, y)

    new_r = np.sqrt(X*X+Y*Y)
    new_t = np.arctan2(X, Y)

    ir = interp1d(r, np.arange(len(r)), bounds_error=False)
    it = interp1d(t, np.arange(len(t)))

    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())

    new_ir[new_r.ravel() > r.max()] = len(r)-1
    new_ir[new_r.ravel() < r.min()] = 0

    return map_coordinates(grid, np.array([new_ir, new_it]),
                            order=order).reshape(new_r.shape)

