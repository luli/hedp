#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np
from scipy.interpolate import interp1d


_MCP2Sens = {'S20': np.array([  2.80000000e-03,  -3.50400000e+00,   1.10790000e+03])}

_Gain2MCP = {'S20': np.array([ 46.,  472.]) }

def goi_sens(gain, model):
    """
    Return the Radiant Sensitivity of LULI GOIs.

    Parameters:
    -----------
       - gain [int]: value of the gain
       - model [str]: GOI_id id. Currently in ['S20']

    Returns:
    --------
       - rad_sens: relative gain in W/W
    """
    MCP2Sens = _MCP2Sens[model]
    Gain2MCP = _Gain2MCP[model]

    MCP = np.polyval(Gain2MCP, gain)
    return np.polyval(MCP2Sens, MCP)

