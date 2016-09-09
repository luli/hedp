#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# ajout supergaussian 1D, Gabriel Rigon 2016
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np
from scipy.integrate import quad

def super_gaussian_ring(r,a,r_c, gamma):
    """Super gaussian of thickness a, centred at r==r_c
    and with exponent gamma
    """
    return np.exp(-((r-r_c)/a)**gamma)

def super_gaussian_ring_dr(r,a,r_c, gamma):
    return 2*np.pi*r*super_gaussian_ring(r,a,r_c, gamma)

def Int_super_gaussian_ring(a, r_c, gamma):
    """ Area under a supergaussian of radius a and parameter gamma, 
    centered at r_c.

    Integration between 0 and r_c+20*a (that's supposed to be + inf)
    This basically gives the area of the focal spot
    """
    res =  quad(super_gaussian_ring_dr, 0, r_c+20*a, args=(a,r_c, gamma), 
            points=[r_c])
    if res[1] < 1e-6 and res[0] != 0:
        return res[0]
    else:
        raise ValueError("Quad integration was not convergent: error {0}".format(res[1]))

def Int_super_gaussian(a, gamma):
    """ Area under a supergaussian of radius a and paramater gamma"""
    return Int_super_gaussian_ring(a, 0.0, gamma)

def Int_super_gaussian1D(a,gamma):
    """Area under a supergaussian 1D of radius a and parameter gamma"""
    res = quad(super_gaussian_1D, 0, 20*a, args=(a, gamma), points=[0.0])
    if res[1] < 1e-6 and res[0] != 0:
        return res[0]
    else:
        raise ValueError("Quad integration was not convergent: error {0}".format(res[1]))

def super_gaussian_1D(r,a,gamma):
    """Super gaussian of thickness a, centered at r=0 and with exponent gamma"""
    return np.exp(-(r/a)**gamma)
