#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright CNRS 2012, roman.yurchak@polytechnique.edu
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np


def add_multiple(*arrs):
    """
    this could be seen as np.add(x,y) with unlimited number of arguments
    """
    if len(arrs) > 1:
        return add_multiple(*arrs[1:]) + arrs[0]
    else:
        return arrs[0]
    return 
