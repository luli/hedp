#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright CNRS 2012, roman.yurchak@polytechnique.edu
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import numpy as np

def _fget(obj, *pargs, **args):
    if hasattr(obj, '__call__'):
        return obj(*pargs, **args)
    else:
        return obj

class FieldOfVeiw2D():
    def __init__(self, bounds=None,center=(0, 0), extent=None,
                force_aspect_ratio=None, ):
        """
        This class is used to define a field of view in 2D for use in simulations and
        with experimental data.

        Parameters:
        -----------
         - bounds: FoV bounds (xmin, xmax, ymin, ymax) [optional]
         - center: center of the FoV (x0, y0)
         - extent: extent of the FoV (dx, dy)
         - force_aspect_ratio: keep constant aspect ratio (aspect_ratio, axis)

         bounds, center, and extent can be either tuples or a function returning a tuple
         that takes a time
        """
        if force_aspect_ratio is not None:
            self.aspect_ratio, self.aspect_ratio_keep_axis = force_aspect_ratio
        if bounds:
            self.bounds = bounds
        elif extent and center:
            self.center, self.extent = center,  extent
        else:
            raise ValueError


    @staticmethod
    def _bounds2extent(bounds):
        center = np.sum(bounds[:2])/2, np.sum(bounds[2:])/2
        extent = bounds[1] - center[0], bounds[3] - center[1]
        return center, extent

    @staticmethod
    def _extent2bounds(center, extent):
        bounds = (center[0] - extent[0], center[0] + extent[0],
                  center[1] - extent[1], center[1] + extent[1])
        return bounds

    @staticmethod
    def _normalize(bounds, aspect_ratio, axis):
        center, extent = FieldOfVeiw2D._bounds2extent(bounds)
        if axis == 0:
            extent = extent[0], extent[0]*aspect_ratio
        elif axis == 1:
            extent = extent[0]*aspect_ratio, extent[1]
        return FieldOfVeiw2D._extent2bounds(center, extent)

    def get_bounds(self, t=None):
        """
        Compute bounds for the field of view
        """
        if hasattr(self, 'extent'):
            center = _fget(self.center, t)
            extent = _fget(self.extent, t)
            bounds = FieldOfVeiw2D._extent2bounds(center, extent)
        elif hasattr(self, 'bounds'):
            bounds = _fget(self.bounds, t)

        if hasattr(self, 'aspect_ratio'):
            bounds = FieldOfVeiw2D._normalize(bounds, self.aspect_ratio,
                                        self.aspect_ratio_keep_axis)
        return bounds
