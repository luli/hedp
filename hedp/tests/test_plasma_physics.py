#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy.testing import assert_allclose
import hedp.plasma_physics


def test_critical_density():
    """Checking the critical density calculation"""
    F0 = hedp.plasma_physics.critical_density(500)
    F1 = 1.1148542159362519e21/(500*1e-3)**2
    assert_allclose(F0, F1, atol=0.1)

