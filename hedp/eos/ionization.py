#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright CNRS 2012, roman.yurchak@polytechnique.edu
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.


import numpy as np

def thomas_fermi_ionization(rho, Te, Z, A):
    """ Thomas Fermi pressure ionization
        The Physics of Inertial Fusion, Atzeni (p335)

        Parameters:
        -----------
            - rho : ndarray: density [g.cm⁻³]
            - Te : ndarray: temperature [eV]
            - A : float:
            - Z : float:
        Returns:
        --------
            - Zion: ndarray: average ionization
    """
    assert Z>0 and (A>Z)
    assert not (np.any(Te<0) or np.any(rho<0))
    alpha=14.3139
    beta=0.6624
    rho1=rho/(A*Z)
    zero_Te_mask = (Te==0)
    x = np.ones(rho.shape)*np.nan

    # T == 0
    x[zero_Te_mask]=alpha*rho1[zero_Te_mask]**beta

    # T > 0 
    a = [0.003323, 0.971832, 9.26148e-5, 3.10165]
    b = [-1.7630, 1.43175, 0.31546]
    c = [-0.366667, 0.983333]

    T1=Te[~zero_Te_mask]/Z**(4./3)
    Tf=T1/(1.+T1)
    A1=a[0]*T1**a[1]+a[2]*T1**a[3]
    B=-np.exp(b[0]+b[1]*Tf+b[2]*Tf**7)
    C=c[0]*Tf+c[1]
    Q1=A1*rho1[~zero_Te_mask]**B
    Q=(rho1[~zero_Te_mask]**C+Q1**C)**(1./C)
    x[~zero_Te_mask]=alpha*Q**beta

    Zion=Z*x/(1.+x+np.sqrt(1.+2*x))
    return Zion
