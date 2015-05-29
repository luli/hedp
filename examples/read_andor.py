#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hedp.io

import pylab as plt
import scipy.ndimage
import sys

sif = hedp.io.SifFile(sys.argv[1])
cs = plt.imshow(scipy.ndimage.filters.median_filter(sif.data, 3))#, cmap=plt.cm.Paired)
plt.colorbar(cs)
plt.show()    
