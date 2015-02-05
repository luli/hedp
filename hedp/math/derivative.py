#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np
from scipy.linalg import circulant
import scipy.ndimage as nd

def gradient(f, x=None, dx=1, axis=-1):
    """
    Return the gradient of 1 or 2-dimensional array.
    The gradient is computed using central differences in the interior
    and first differences at the boundaries. 
    Irregular sampling is supported (it isn't supported by np.gradient)

    Parameters
    ----------
    f: 1d or 2d numpy array
       Input array.
    x: array_like, optional
       Points where the function f is evaluated. It must be of the same
       length as f.shape[axis].
       If None, regular sampling is assumed (see dx)
    dx: float, optional
       If `x` is None, spacing given by `dx` is assumed. Default is 1.
    axis: int, optional
       The axis along which the difference is taken.

    Returns
    -------
    out: array_like
        Returns the gradient along the given axis. 

    To do:
      implement smooth noise-robust differentiators for use on experimental data.
      http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/
    """
    if x is None:
        x = np.arange(f.shape[axis])*dx
    else:
        assert x.shape[0] == f.shape[axis]
    I = np.zeros(f.shape[axis])
    I[:2] = np.array([0,-1])
    I[-1] = 1
    I = circulant(I)
    I[0,0] = -1
    I[-1,-1] = 1
    I[0,-1] = 0
    I[-1,0] = 0 
    H = np.zeros((f.shape[axis],1))
    H[1:-1,0] = x[2:]-x[:-2]
    H[0] = x[1] - x[0]
    H[-1] = x[-1] - x[-2]
    if axis==0:
        return np.dot(I/H, f)
    else:
        return np.dot(I/H, f.T).T

def laplace(f, dx):
    """Compute laplace operator assyming cylindrical geometry
    Parameters:
    -----------
     - f is an array of the shape (z,r)
     - dx: float: sampling distance (must be the same in x,y) [cm]
    """
    flarge = np.zeros((f.shape[0]+2, f.shape[1]+2)) # assume that everything is zero at the edges
    flarge[1:-1,1:-1] = f
    flarge[:,0] = flarge[:,1] # f(-r,z) = f(r,z)
    df = nd.filters.laplace(flarge)/(dx**2)
    return df[1:-1,1:-1]

