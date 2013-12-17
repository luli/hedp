#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np
from scipy.integrate import quad

def super_gaussian(r,a,r_c, gamma):
    """Super gaussian of thickness a, centred at r==r_c
    and with exponent gamma
    """
    return np.exp(-((r-r_c)/a)**gamma)

def super_gaussian_dr(r,a,r_c, gamma):
    return 2*np.pi*r*super_gaussian(r,a,r_c, gamma)

def Int_super_gaussian(a, r_c, gamma):
    """
    Integration between 0 and r_c+20*a (that's supposed to be + inf)
    This basically gives the area of the focal spot
    """
    res =  quad(super_gaussian_dr, 0, r_c+20*a, args=(a,r_c, gamma), 
            points=[r_c])
    if res[1] < 1e-6 and res[0] != 0:
        return res[0]
    else:
        raise ValueError("Quad integration was not convergent: error {0}".format(res[1]))


#print Int_super_gaussian(1,10, 100)/np.pi
#print (11**2-9**2)

