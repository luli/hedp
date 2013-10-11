#!/usr/bin/python
# -*- coding: utf-8 -*-
import time

import numpy as np
from hedp.math.abel import abel, abel_analytical_step, iabel
from hedp.math.derivative import gradient
import scipy.ndimage as nd
from numpy.testing import assert_allclose


"""
To run tests run
nosetests -s abel.py
"""

def setup():
    pass


def test_abel_zeros():
    # just a sanity check
    n = 64
    x = np.zeros((n,n))
    assert (abel(x, inverse=False)==0).all()

def test_abel_gaussian():
    n = 500
    r = np.linspace(0, 5., n)
    dr = np.diff(r)[0]
    rc = 0.5*(r[1:]+r[:-1])
    fr = np.exp(-rc**2)
    Fn = abel(fr, dr=dr)
    Fn_a = np.pi**0.5*np.exp(-rc**2)
    yield assert_allclose,  Fn, Fn_a, 1e-2, 1e-3

def test_laplace():
    dx = 1e-3
    r = np.arange(1e-6, 1, dx)
    z = np.arange(-2, 4, dx)
    R, Z = np.meshgrid(r, z, indexing='ij')
    D0 =  R+Z
    yield assert_allclose, nd.filters.laplace(D0)[1:-1,1:-1], np.zeros(D0.shape)[1:-1, 1:-1], 1e-6, 1e-6
    D0 =  R**2
    yield  assert_allclose,nd.filters.laplace(D0)[1:-1,1:-1]/dx**2, 2*np.ones(D0.shape)[1:-1, 1:-1], 1e-6



