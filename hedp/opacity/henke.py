#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# Copyright CNRS 2012, roman.yurchak@polytechnique.edu
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

import re
import os
import sys

import numpy as np
import mechanize
import urllib
from scipy.interpolate import interp1d

import hedp
import tables
from time import time
import numbers
#from pyquery import PyQuery as pq
#try:
#    from .. import matdb
#except ValueError:
#    sys.path.append('../')
#    import matdb

HENKE_DATA_PATH = os.path.join(hedp.MATDB_PATH, 'henke_op.h5')

# creating an empty DB file if non existant
if not os.path.exists(HENKE_DATA_PATH):
    with tables.openFile(HENKE_DATA_PATH, 'w') as f:
        pass

def cold_opacity(element, dens=-1, nu=None, hdf5_backend='h5py'):
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
    if hdf5_backend == 'h5py':
        import h5py
        with h5py.File(HENKE_DATA_PATH, 'r') as f:
            if not '/'+element in f:
                print "Warning: couldn't find cold opacity for {0} ; trying to download...".format(element)
                f.close()
                download_full(element)
                f = h5py.File(HENKE_DATA_PATH, 'r')
            nu0 = f[element]['nu'][:]
            op0 = f[element]['op'][:]
            f.close()
    elif hdf5_backend == 'pytables':
        with tables.open_file(HENKE_DATA_PATH, 'r') as f:
            print 'ok'
            if not '/'+element in f:
                print "Warning: couldn't find cold opacity for {0} ; trying to download...".format(element)
                f.close()
                download_full(element)
                f = tables.openFile(HENKE_DATA_PATH, 'r')

            nu0 = getattr(f.root, element).nu[:]
            op0 = getattr(f.root, element).op[:]
            f.close()

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


def download_full(element):
    """
    Download files to database for given element
    """
    db = hedp.matdb(element)
    op_tot = []
    nu_tot = []
    for nu in [(10,100),(100, 1000), (1000, 9000), (9000, 30000)]:
        nu_arr, op_arr =  download(db.formula, db.solid_dens, nu=nu)
        nu_tot.append(nu_arr)
        op_tot.append(op_arr)
    op = np.concatenate(op_tot)
    nu = np.concatenate(nu_tot)
    _, mask = np.unique(nu, return_index=True)

    data = {'op': op[mask], 'nu': nu[mask]}

    with tables.openFile(HENKE_DATA_PATH, 'a') as f:
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
    br = mechanize.Browser(factory=mechanize.RobustFactory())
    #br.addheaders = [('User-agent',
    #    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) \
    #    Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
    #br.set_proxies({"http":"this_proxy.com:8080"})
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
    response = br.open('http://henke.lbl.gov/cgi-bin/filter.pl',
            data=urllib.urlencode(post_vars))
    response_txt = response.read()
    link = br.links().next()
    filename = br.retrieve('http://henke.lbl.gov'+link.url)
    print 'Henke opacity downloaded for',dens,'g/cc'
    return parse(filename[0])

def parse(path):
    """
    Parse the file downloaded from the website

    Returns:
    --------
      - nu [ndarray]: array of energies [eV]
      - op [ndarray]: array of opacities [cm².g⁻¹]
    """
    with open(path, 'r') as f:
        header = f.readline()
    regexp = r'^[\s\w]+=(?P<dens>(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)+)[\s\w]+=(?P<thick>[0-9.]+)'
    try:
        dens, thick  = re.match(regexp, header).groups()
        dens, thick = float(dens), float(thick)
    except AttributeError:
        raise ValueError("For some reason couldn't parse the snop file {0}".format(path))
    nu, op = np.loadtxt(path,skiprows=2).T
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
    print op[...,0]






