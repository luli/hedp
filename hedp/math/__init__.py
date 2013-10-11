#!/usr/bin/python
# -*- coding: utf-8 -*-
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
