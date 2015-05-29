#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
import sys

import numpy as np
import urllib
from scipy.interpolate import interp1d

import hedp
from time import time
import numbers

HENKE_DATA_PATH = os.path.join(hedp.MATDB_PATH, 'henke_op')

## creating an empty DB file if non existant
#if not os.path.exists(HENKE_DATA_PATH):
#    with tables.openFile(HENKE_DATA_PATH, 'w') as f:
#        pass

def cold_opacity(element, dens=0.1, nu=None, hdf5_backend='pytables'):
    """
    Parameters:
    -----------
     - element [str]: chemical element
     - dens [float or ndarray]: array of densities [g.cm⁻³]
     - nu [ndarray]: array of energies [eV]
          must be between [10 eV, 30 keV]
     - hdf5_backend [str]: pytables or h5py [use because of
              a bug between pytables and yt]
    Returns:
    --------
        opacity in cm⁻¹
    """
    #hdf5_backend = 'h5py'
    if hdf5_backend == 'h5py':
        import h5py
        with h5py.File(HENKE_DATA_PATH+'.h5', 'r') as f:
            if not '/'+element in f:
                print("Warning: couldn't find cold opacity for {0} ; trying to download...".format(element))
                f.close()
                download_full(element)
                f = h5py.File(HENKE_DATA_PATH+'.h5', 'r')
            nu0 = f[element]['nu'][:]
            op0 = f[element]['op'][:]
            f.close()
    elif hdf5_backend == 'pytables':
        import tables
        with tables.open_file(HENKE_DATA_PATH+'.h5', 'a') as f:
            if not '/'+element in f:
                print("Warning: couldn't find cold opacity for {0} ; trying to download...".format(element))
                f.close()
                download_full(element)
                f = tables.openFile(HENKE_DATA_PATH+'.h5', 'r')

            nu0 = getattr(f.root, element).nu[:]
            op0 = getattr(f.root, element).op[:]
            f.close()
    elif hdf5_backend == 'pickle':
        import pickle
        with open(HENKE_DATA_PATH+'.pickle', 'rb') as handle:
              mdict = pickle.load(handle)
              nu0, op0 = mdict[element]
    else:
        raise ValueError
    #print hdf5_backend

    if nu is not None:
        op = interp1d(nu0, op0)(nu)
    else:
        op = op0
    if isinstance(dens, numbers.Number):
        if dens < 0:
            dens = hedp.matdb(element).solid_dens
        return op*dens
    elif dens.ndim <= 1:
        return op*dens.reshape(-1,1)
    elif dens.ndim == 2:
        return dens[:,:,np.newaxis]*op[np.newaxis, np.newaxis, :]


def sync_pickle_db():
    import tables
    import pickle
    f = tables.open_file(HENKE_DATA_PATH+'.h5', 'r')
    out_dir = {}
    specs =  [key for key in dir(f.root) if not key.startswith('_')]
    for el in specs:
        nu0 = getattr(f.root, el).nu[:]
        op0 = getattr(f.root, el).op[:]
        out_dir[el] = nu0, op0
    with open(HENKE_DATA_PATH+'.pickle', 'wb') as handle:
      pickle.dump(out_dir, handle)
    f.close()





def download_full(element, dens=None):
    """
    Download files to database for given element
    """
    import tables
    db = hedp.matdb(element)
    op_tot = []
    nu_tot = []
    for nu in [(10,100),(100, 1000), (1000, 9000), (9000, 30000)]:
        if dens is None:
            if db.solid_dens:
                if db.solid_dens>0.1:
                    solid_dens = db.solid_dens
                else:
                    solid_dens = 1e4*db.solid_dens  # this is a gas
            else:
                solid_dens = 0.1
        else:
            solid_dens=dens
        nu_arr, op_arr =  download(db.formula, solid_dens, nu=nu)
        nu_tot.append(nu_arr)
        op_tot.append(op_arr)
    op = np.concatenate(op_tot)
    nu = np.concatenate(nu_tot)
    _, mask = np.unique(nu, return_index=True)

    data = {'op': op[mask], 'nu': nu[mask]}

    with tables.openFile(HENKE_DATA_PATH+'.h5', 'a') as f:
        atom = tables.Atom.from_dtype(data['op'].dtype)
        group = f.createGroup(f.root, element)
        for name, arr in data.iteritems():
            ds = f.createCArray(group, name, atom, arr.shape)
            ds[:] = arr


def download(formula, dens, nu=(10,20000)):
    """
    Download a cold opacity table from Xro website.

    Parameters:
    -----------
      - formula [str]: chemical formula of the component
      - dens [float]: density [g.cm]
      - nu [tuple]: Min and Max frequencies [eV]

    Returns:
    --------
      - nu [ndarray]: array of energies [eV]
      - op [ndarray]: array of opacities [cm².g⁻¹]

    """
    # all of this could be more simpler done with requests, etc.
    # but trying to use only standard library functions 
    import urllib
    import urllib2
    import sys
    if sys.version_info.major > 2:
        from html.parser import HTMLParser
    else:
        from HTMLParser import HTMLParser
    from cStringIO import StringIO

    class LinksExtractor(HTMLParser):

        url = None

        def handle_starttag(self, tag, attrs):
            # Only parse the 'anchor' tag.
            if tag == "a":
               # Check the list of defined attributes.
               for name, value in attrs:
                   # If href is defined, print it.
                   if name == "href":
                       self.url = value

    url_base = 'http://henke.lbl.gov'
    url_query = '/cgi-bin/filter.pl'
    post_vars = dict(Material="Enter Formula",
            Scan="Energy",
            Npts=500,
            Plot="Linear", # Linear
            Output="Plot")
    assert type(dens) is float, 'dens should be a float'
    post_vars['Formula'] = formula
    post_vars['Min'], post_vars['Max'] = nu
    post_vars['Density'] = dens
    post_vars['Thickness'] = 0.1
    req = urllib2.Request(url_base + url_query,
            data=urllib.urlencode(post_vars))
    response = urllib2.urlopen(req)
    response_txt = response.read()
    html_parser = LinksExtractor()
    html_parser.feed(response_txt)
    html_parser.close()
    if not html_parser.url:
        raise ValueError('Failed to parse output html code!')
    req = urllib2.Request('http://henke.lbl.gov' + html_parser.url)
    response = urllib2.urlopen(req)
    data = StringIO(response.read())
    return parse(data)

def parse(f):
    """
    Parse the file downloaded from the website

    Returns:
    --------
      - nu [ndarray]: array of energies [eV]
      - op [ndarray]: array of opacities [cm².g⁻¹]
    """
    header = f.readline().decode('utf-8')
    regexp = r'^[\s\w]+=(?P<dens>(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)+)[\s\w]+=(?P<thick>[0-9.]+)'
    try:
        dens, thick  = re.match(regexp, header).groups()
        dens, thick = float(dens), float(thick)
    except AttributeError:
        raise ValueError("Parsing of header failed!")
    f.seek(0)
    nu, op = np.loadtxt(f, skiprows=2).T
    op = np.abs(np.log(op) / ( thick * 1e-4 * dens))
    return nu, op

if __name__=='__main__':
    import matplotlib.pylab as plt
    #nu, op = download('Al', 1.24)
    #save2matdb('Al', nu, op)
    #download_full('polystyrene')
    op =  cold_opacity('TMPTA_Br30',
            np.array([[0.1], [0.3]]),
            [100])
    #plt.loglog(nu, op)
    #plt.show()
    print(op[...,0])






