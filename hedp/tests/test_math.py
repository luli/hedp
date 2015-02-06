#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from hedp.math.derivative import gradient
from numpy.testing import assert_allclose

def test_gradient():
   y = np.random.rand(100)
   assert_allclose(np.gradient(y), gradient(y))
