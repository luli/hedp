#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
import os
import sys

import numpy as np
import mechanize
import urllib
from scipy.interpolate import interp1d

import hedp
from time import time
import numbers
#from pyquery import PyQuery as pq
#try:
#    from .. import matdb
#except ValueError:
#    sys.path.append('../')
#    import matdb

def cold_opacity(element, dens=-1, nu=None):
    """
    Parameters:
    -----------
     - element [str]: chemical element
     - dens [float or ndarray]: array of densities [g.cm⁻³]
     - nu [ndarray]: array of energies [eV]
          must be between [10 eV, 30 keV]
    Returns:
    --------
        opacity in cm⁻¹
    """
    filepath = os.path.join(hedp.MATDB_PATH, 'henke', element+'.dat')
    if not os.path.exists(filepath):
        print "Warning: cold opacity files don't seem to exit; trying to download..."
        download_full(element)
    nu0, op0 = np.loadtxt(filepath).T
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

        ## this works but is soo slow
        #op = np.array([dens*op_at_nu for op_at_nu in np.nditer(op)])
        #print np.rollaxis(op, 0, 3).shape
        #return np.rollaxis(op, 0, 3) # setting this to be the right shape


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

    op = op[mask]
    nu = nu[mask]
    print op.shape, nu.shape
    save2matdb(element, nu, op)



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
    br.addheaders = [('User-agent',
        'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) \
        Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
    #br.set_proxies({"http":"cache.polytechnique.fr:8080"})
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
    regexp = r'^[\s\w]+=(?P<dens>[0-9.]+)[\s\w]+=(?P<thick>[0-9.]+)'
    try:
        dens, thick  = re.match(regexp, header).groups()
        dens, thick = float(dens), float(thick)
    except AttributeError:
        raise ValueError("For some reason couldn't parse the snop file {0}".format(path))
    nu, op = np.loadtxt(path,skiprows=2).T
    op = np.abs(np.log(op) / ( thick * 1e-4 * dens))
    return nu, op

def save2matdb(element, nu, op):
    """Takes a file and saves it to MATDB"""

    filepath = os.path.join(hedp.MATDB_PATH, 'henke', element+'.dat')
    np.savetxt(filepath, np.array([nu, op]).T)

if __name__=='__main__':
    import matplotlib.pylab as plt
    #nu, op = download('Al', 1.24)
    #save2matdb('Al', nu, op)
    #download_full('polystyrene')
    op =  get_cold_opacity('TMPTA_Br30',
            np.array([[0.1], [0.3]]),
            [100])
    #plt.loglog(nu, op)
    #plt.show()
    print op[...,0]






