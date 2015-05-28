#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from numpy.testing import assert_allclose
from unittest.case import SkipTest

from hedp.opacity.AGS import project_on_grid, SelectRect

def test_project_on_grid():
    points = np.array([1.15, 1.99, 3.4])
    grid = np.arange(0, 5, 0.1)

    idx, proj_points = project_on_grid(points, grid)
    yield assert_allclose, proj_points, np.array([1.1, 2.0, 3.4])


def test_select_rectangle():
    rho_lim = 1e-6, 1.2e-4
    temp_lim = 1e1, 1.2e3
    weight = 1.0
    regexp = None
    Nrho=5
    Ntemp=5
    d = SelectRect(rho_lim, temp_lim, weight, regexp, Nrho, Ntemp)

    rho = np.logspace(-8, 3, 12)
    temp = np.logspace(-2, 3, 6)
    #print(rho)
    #print(temp)

    raise SkipTest

    ridx, tidx, _ = d(rho, temp)
    yield assert_allclose, rho[ridx], np.array([1e-6, 1e-5, 1e-4])
    yield assert_allclose, temp[ridx], np.array([1e1, 1e2, 1e3])




