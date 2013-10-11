# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#
# Copyright CNRS 2012, roman.yurchak@polytechnique.edu
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np
cimport numpy as np

cpdef char[:,::1] rect_mask_comp(double[:, ::1] X, double[:, ::1] Y,
        char[:,::1] mask, double a, double b_min, double b_max) nogil:
    """
    Parameters:
    -----------
     - rect [ndarray]:   [[x0,y0], [x1, y1]]  array of rectangle points 
     - X [ndarray] : X indices
     - Y [ndarray] : Y indices
    
    Returns:
    --------
     - mask [ndarray]: mask of the same shape as X, Y within the rectangle
    """
    cdef int i, j
    cdef double val
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if mask[i,j]:
                val = a*X[i,j] - Y[i,j]
                if (val < -b_max) or (val > -b_min):
                    mask[i,j] = 0
    return mask
