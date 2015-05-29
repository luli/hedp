#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# just call this file as
# python hamamatsu.py filename.img
import hedp.io
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(
        description="Open a Hamamatsu streak camera's .img file")
parser.add_argument('filepath', type=str,
                           help='path to the .img file')

args = parser.parse_args()
#  some logic to determine offset mode depending on the folder
offset = 'from_end'
if sum([key in args.filepath for key in ['Rear_SOP_1D']]) and\
    not sum([key in args.filepath for key in ['ref.img', 'alignemts']]):
    offset =  'from_end_4k'
elif sum([key in args.filepath for key in ['Transverse_SOP_1D']]):
    offset =  'from_end'
offset = 'from_end_4'
print(offset)


sp = hedp.io.HamamatsuFile(args.filepath, offset, dtype="int16")
print(sp._offset_data)
print(sp.data.shape, sp._nbytes)

d = sp.data
cs = plt.imshow(sp.data, vmax=np.percentile(sp.data, 99.9))

plt.colorbar(cs)
plt.show()
