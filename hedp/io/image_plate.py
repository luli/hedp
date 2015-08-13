#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path


def parse_inf(path):
    """
    Parameters:
     - path: path to .inf file containing image plate information

    Returns:
      a dictionary with appropriate keys
    """

    with open(path, 'r') as fh:
        txt = fh.readlines()

    out = {}
    for key, idx, dtype in [('L', 9, int),
                            ('Res', 3, int),
                            ('S', 8, int)
                           ]:
        out[key] = dtype(txt[idx])

    return out


def convert_ip_to_psl(basepath, nbits=16):
    """
    Open a IP scan and convert the data to PSL

    Parameters:
    -----------
      - basepath: path to one of .img, .tif, .inf files
         at present .tif, .inf files should also be present in the same directory
      - nbits: resolution of the image in bits (default 16 bits)

    Returns:
    --------
      a 2D array with the data
    """
    import skimage.io
    skimage.io.use_plugin('freeimage')


    basedir, basefile = os.path.split(basepath)
    basefile, baseext = os.path.splitext(basefile)
    files_set = {}
    for ext in ['inf', 'tif', 'img']:
        path = os.path.join(basedir, '.'.join([basefile, ext]))
        if os.path.exists(path):
            files_set[ext] = path

    if 'inf' not in files_set:
        raise ValueError('Cannot convert IP scan with .inf file missing')

    if 'tif' not in files_set:
        raise NotImplementedError('Reading of .img files not implemented yet!')


    pars = parse_inf(files_set['inf'])

    data = skimage.io.imread(files_set['tif'])

    alpha = (pars['Res']/100.)**2*(4000/pars['S'])
    beta = pars['L']*(data.astype('float64')/2**nbits-0.5)

    return alpha*10**beta, pars
