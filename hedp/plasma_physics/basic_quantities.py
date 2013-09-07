#!/usr/bin/python
# -*- coding: utf-8 -*-
from scipy.constants import e, c, m_e, epsilon_0
import numpy as np

def critical_density(lmbda):
    """
    Compute critical density for a plasma
    
    Parameters:
    -----------
     - lmbda [ndarray or float]: wavelengt [nm]

    Returns:
     - Nc: [cm⁻³]
    """
    lmbda *= 1e-9
    omega = 2*np.pi*c/lmbda
    return 1e-6*epsilon_0*m_e*omega**2/e**2
