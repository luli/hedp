#!/usr/bin/python
# -*- coding: utf-8 -*-
import time

import numpy as np
from hedp.math.abel import abel0, abel_analytical_step, iabel
from hedp.math.derivative import gradient
from numpy.testing import assert_allclose

abel = abel0

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
    Fn = abel(fr, r=rc)
    Fn_a = np.pi**0.5*np.exp(-rc**2)
    yield assert_allclose,  Fn, Fn_a, 1e-2, 1e-3


