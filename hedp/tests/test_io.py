#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path

import nose
from numpy.testing import assert_allclose
import numpy as np

import hedp.io

BASE_DIR = os.path.join(os.path.dirname(__file__), 'data')

def test_andor():
    f = hedp.io.SifFile(os.path.join(BASE_DIR, 'io/shot26_ref.sif.bz2'))

    yield assert_allclose, f.data.shape, (1024, 1024)

def test_hamamatsu():
    import bz2
    f_ref = bz2.BZ2File(os.path.join(BASE_DIR, 'io/test_image.npy.bz2'))
    f = hedp.io.HamamatsuFile(os.path.join(BASE_DIR, 'io/test_image.img.bz2'),
            offset="auto", dtype="int16")

    raise nose.SkipTest
    ref = np.load(f_ref).astype(np.int16)
    yield assert_allclose(ref, f.data)

