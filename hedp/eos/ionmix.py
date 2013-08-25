#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.constants import N_A

def ionmix_validity(rho, temp, Zbar, Abar):
    """
    Returns the distance to the IONMIX EoS validity domain:
      if Ion_density < 1e20 (T/Zbar)³ cm⁻³ return 0
      else: return orthogonal distance to that validity limit 
          in the log log space

    Parameters
    ----------
      - rho: ndarray: density [g.cm⁻³]
      - temp:ndarray: temperature [eV]
      - Zbar:ndarray: average ionization
      - Abar:ndarray or float: average atomic mass

    Returns
    -------
      - d: ndarray: distance to the validity region in
           log log space


    """
    return temp >= (rho*N_A*Zbar**3/(1e20*Abar))**(1./3)
