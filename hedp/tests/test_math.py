#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from hedp.math.derivative import gradient
from numpy.testing import assert_allclose
from hedp.math.integrals import Int_super_gaussian_ring, Int_super_gaussian


def test_gradient():
   y = np.random.rand(100)
   assert_allclose(np.gradient(y), gradient(y))


def test_Int_super_gaussian_ring():
    a = 2.0
    r_c = 10.0
    gamma = 4.0
    res = Int_super_gaussian_ring(a, r_c, gamma)
    assert_allclose(res, np.pi*((r_c+a)**2 - (r_c-a)**2), rtol=1e-1)

def test_Int_super_gaussian():
    a = 2.0
    gamma = 4.0
    res = Int_super_gaussian(a, gamma)
    assert_allclose(res, np.pi*(a)**2, rtol=2e-1)
