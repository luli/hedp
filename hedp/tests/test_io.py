#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
from numpy.testing import assert_allclose
import hedp.io

BASE_DIR = os.path.join(os.path.dirname(__file__), 'data')

def test_andor():
    f = hedp.io.SifFile(os.path.join(BASE_DIR, 'io/shot26_ref.sif.bz2'))

    yield assert_allclose, f.data.shape, (1024, 1024)

def test_hamamatsu():
    f = hedp.io.HamamatsuFile(os.path.join(BASE_DIR, 'io/shot05_ref.img.bz2'),
            offset="from_end_4k", dtype="int16")
    yield assert_allclose, f.data.shape, (1024, 1344)

