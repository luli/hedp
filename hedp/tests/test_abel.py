#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.
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
    yield assert_allclose, nd.filters.laplace(D0)[1:-1,1:-1]/dx**2, 2*np.ones(D0.shape)[1:-1, 1:-1], 1e-6



def test_abel_step():
    n = 800
    r = np.linspace(0, 20, n)
    dr = np.diff(r)[0]
    rc = 0.5*(r[1:]+r[:-1])
    fr = np.exp(-rc**2)
    #fr += 1e-1*np.random.rand(n)
#    plt.plot(rc,fr,'b', label='Original signal')
    F = abel(fr,dr=dr)
    F_a = (np.pi)**0.5*fr.copy()

    F_i = abel(F,dr=dr, inverse=True)
    #sys.exit()
#    plt.plot(rc, F_a, 'r', label='Direct transform [analytical expression]')
#    mask = slice(None,None,5)
#    plt.plot(rc[mask], F[mask], 'ko', label='Direct transform [computed]')
#    plt.plot(rc[mask], F_i[mask],'o',c='orange', label='Direct-inverse transform')
    yield assert_allclose, fr, F_i, 5e-3, 1e-6, 'Test that direct>inverse Abel equals the original data'
    yield assert_allclose, F_a, F, 5e-3, 1e-6, 'Test direct Abel transforms failed!'

    

