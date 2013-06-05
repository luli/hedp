# -*- coding: utf-8 -*-
import numpy as np
from .cython import rect_mask_comp

SMALL=1.0e-16


def rect_mask(rect, X, Y, mask=None):
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
    lines = np.zeros((4,2))
    # computing the equation of lines passing by the 4 points
    for offset in range(4):
        idx = [offset, (offset + 1) % 4 ]
        coords = rect[idx]
        lines[offset,:] = np.linalg.solve(coords, np.ones(2))
    # computing the mask within those 4 lines
    if mask is None:
        mask = np.ones(X.shape, dtype=np.uint8)
    for idx in [[0,2], [1,3]]:
        clines = lines[idx]
        # equation of the form b*y = 1
        if abs(clines[0,0]) < SMALL:
            midx = np.sort((1./clines[:,1]).astype(np.int))
            mask[:midx[0],:] = 0
            mask[midx[1]:,:] = 0
        # equation of the form a*x = 1
        elif abs(clines[0,1]) < SMALL:
            midx = np.sort((1./clines[:,0]).astype(np.int))
            mask[:, :midx[0]] = 0
            mask[:, midx[1]:] = 0
        # equation of the form c_0*x + c_1*y = 1
        #   => y = a*x + b
        else:
            a = -clines[0,0]/clines[0,1]
            b_min, b_max = np.sort(1./clines[:,1])
            mask = rect_mask_comp(X,Y, mask, a, b_min, b_max)

    return np.asarray(mask, dtype=np.bool)
