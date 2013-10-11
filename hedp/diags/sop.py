#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright CNRS 2012, roman.yurchak@polytechnique.edu
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.


import numpy as np
from scipy.integrate import cumtrapz

from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy import constants as cst
from scipy.constants import physical_constants
from streak import streak_sens
from goi import goi_sens
from hedp.rad import planck, iplanck
import inspect

def compute_emiss(I0, op, dx=1, axis=0, _sum=False):
    if axis==0:
        op = op[::-1]
    elif axis==1:
        op = op[:,::-1]
    #if mask is not None:
    #    # this part of simulation is opaque
    #    op[mask] = 1e3
   

    cop = cumtrapz(op, axis=axis, dx=dx)
    if axis==0:
        cop = np.vstack((np.zeros((1,cop.shape[1])), cop))
    elif axis==1:
        cop = np.hstack((np.zeros((cop.shape[0], 1)), cop))
    if axis==0:
        cop = cop[::-1]
    elif axis==1:
        cop = cop[:,::-1]
    em = I0*np.exp(-cop)
    if _sum:
        return np.sum(em, axis=axis)
    else:
        return em

def polar2cartesian(r, t, grid, x, y, order=3):

    X, Y = np.meshgrid(x, y)

    new_r = np.sqrt(X*X+Y*Y)
    new_t = np.arctan2(X, Y)

    ir = interp1d(r, np.arange(len(r)), bounds_error=False)
    it = interp1d(t, np.arange(len(t)))

    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())

    new_ir[new_r.ravel() > r.max()] = len(r)-1
    new_ir[new_r.ravel() < r.min()] = 0

    return map_coordinates(grid, np.array([new_ir, new_it]),
                            order=order).reshape(new_r.shape)

def get_sop_slit_width(path, super_gaussian_factor=6):
    """
    Compute the slit width from the reference image by
    fitting the profile with a supergaussian

    Parameters:
    -----------
      - path [str]: path to .img file

    Returns:
    --------
      - slit width [float]: in px

    """
    from hedp.io import HamamatsuFile
    from scipy.optimize import curve_fit
    hm = HamamatsuFile(path)
    img = hm.data
    #print img.data

    def slit_width(r, r0, w, h, M):
        return h*np.exp(-2*((r-r0)/w)**super_gaussian_factor) + M
    signal_proj = img.sum(axis=-1)[100:-100]
    x = np.arange(len(signal_proj))

    popt, pcov = curve_fit(slit_width, x, signal_proj, (400, 20, 1e5, 0))
    return 2*popt[1]

def se_calibration(counts, lmbda, F,  transmission, detectorsize,
                                    method='explicit', out='tele', diag='sop',**args):
    """
    Compute the calibration for a streaked self emission system.

    This function calculates the number of electrons ejected by the
    photocathode in a SOP configuration.

    Parameters:
    -----------
      General:
      --------
          - lmbda [ndarray]: wavelenght (nm) array
          - F  [float]:  F number of the first lens
          - transmission  [ndarray]:    Total transmission of the optical system, including the
                          filter (same shape as the nu) 
          - detectorsize  [float] Size of a pixel in spatial direction in μm
          - method [str]: method to use

      Streak only parameters:
      ----------------------

      - slitwidth  [float]   Size of the slit in px
      - sweepspeed        Sweep speed of the streak [100ns, 50ns, etc] in pixel/ns

      GOI relative:
      -------------

    Returns:
    --------
      - Flux or tele[eV], Flux_norm [float]: photon flux (W.m⁻².sr⁻¹.nm⁻¹.counts⁻¹)

    """
    args = locals()
    args.update(args['args'])
    if method not in ['implicit', 'explicit'] or\
            out not in ['tele', 'flux'] or\
            diag not in ['sop', 'goi']:
        raise ValueError('Wrong input arguments!')
    # deleting all the arguments we don't want to pass
    for key in ['counts', 'out', 'diag', 'args']:
        del args[key]

    cal_function = {'sop': _sop_calibration, 'goi': _se_calibration_goi }[diag]

    if method == 'explicit':
        Flux = cal_function(**args)
        if out == 'flux':
            return Flux*counts
        elif out == 'tele':
            max_idx = np.argmax(transmission)
            lmbda_max = lmbda[max_idx]
            return iplanck(lmbda_max, counts*Flux)

    elif method == "implicit":
        tele_max = np.array([0.1]) # staring temperature at 0.1 eV
        counts_max = 0
        # Upper bound for the temperature we would need to go to
        while counts_max < counts.max():
            tele_max *= 2
            counts_max =  cal_function(tele=tele_max, **args)[0]
        tele_i = np.linspace(0, tele_max, 1000)
        counts_i = cal_function(tele=tele_i, **args)
        calibration_fit = interp1d(counts_i, tele_i, bounds_error=False, fill_value=0)
        return calibration_fit(counts)

def _sop_calibration(lmbda, F,  transmission, detectorsize,
                                    slitwidth, sweepspeed, tele=None, method='explicit'):
    """
    Compute the calibration for a streaked self emission system.

    This function calculates the number of electrons ejected by the
    photocathode in a SOP configuration.

    Parameters:
    -----------
      - lmbda [ndarray]: wavelenght (nm) array
      - F  [float]:  F number of the first lens
      - transmission  [ndarray]:    Total transmission of the optical system, including the
                      filter (same shape as the nu) 
      - detectorsize  [float] Size of a pixel in spatial direction in μm
      - slitwidth  [float]   Size of the slit in px
      - sweepspeed        Sweep speed of the streak [100ns, 50ns, etc]in pixel/ns
      - tele [ndarray]: plasma temperature [eV]

    Returns:
    --------
      if method == 'explicit':
        - Flux_norm [float]: photon flux (W.m⁻².sr⁻¹.nm⁻¹.counts⁻¹)
      elif method == 'implicit':
        - counts [ndarray]: counts value on CCD 
                          of the same shape as counts with the temperature
    """
    solid_angle = np.pi/(4*F**2) # solid angle of the first optics in sr
    # Données Tommaso [Joule/count] Streak S20
    K_Jcounts = 6.6434e-18  #
    # Normalize with streak sensitivity for the given wavelenght
    mstreak_sens = lambda lmbda: streak_sens(lmbda, 'hamamatsu', 'S20_2')[0]
    K_Jcounts = K_Jcounts

    # Surface on the target for 1px (m^2)
    S_px = (detectorsize*1e-6)**2
    #tr_itp = interp1d(lmbda, transmission)
    # Transmission at 420 nm
    # Time spend on each pxl:
    # Streak slit 100um = 8px pour calibre :
    # This remains approximately true for this polar experiment
    t_px = slitwidth * sweepspeed/1024.
    # Approximate width of the filter system ( ~10 nm):
    if method == 'explicit':
        max_idx = np.argmax(transmission)
        Tr_max = transmission[max_idx]  #tr_itp(420.)
        lmbda_max = lmbda[max_idx]
        dlmbda = np.abs(np.trapz(transmission, lmbda)/Tr_max) # nm

        Flux_coeff = K_Jcounts/(S_px * solid_angle  * t_px * Tr_max * dlmbda *\
                                    mstreak_sens(lmbda_max)/mstreak_sens(532.))
        return Flux_coeff

    elif method == 'implicit':
        
        K = (S_px * solid_angle  * t_px)/K_Jcounts

        counts = K * np.trapz(
                (transmission*mstreak_sens(lmbda)/mstreak_sens(532.))[np.newaxis,:]\
                *planck(lmbda[np.newaxis,:], tele[:,np.newaxis]),
                dx=np.diff(lmbda*1e-9)[0], axis=1)
        return counts

def _se_calibration_goi(lmbda, F, transmission, detectorsize, gain, timewindow,
                                                    tele=None, method='explicit'):
    """
    Compute the calibration for a streaked self emission system.

    This function calculates the number of electrons ejected by the
    photocathode in a SOP configuration.

    Parameters:
    -----------
      - lmbda [ndarray]: wavelenght (nm) array
      - F  [float]:  F number of the first lens
      - gain [float]: Gain of the GOI
      - timewindow [float]: Time window [s]
      - transmission  [ndarray]:    Total transmission of the optical system, including the
                      filter (same shape as the nu) 
      - detectorsize  [float] Size of a pixel in spatial direction in μm

    Returns:
    --------
      - Flux_norm [float]: photon flux (W.m⁻².sr⁻¹.nm⁻¹.counts⁻¹)

    """
    solid_angle = np.pi/(4*F**2) # solid angle of the first optics in sr
    # Données Sophie [Joule/count] GOI S20
    K_Jcounts = 1.2e-19  # J
    K_Jcounts = K_Jcounts/(goi_sens(gain, "S20")/goi_sens(5, "S20"))
    # Surface on the target for 1px (m^2)
    S_px = (detectorsize*1e-6)**2
    # Transmission at 420 nm
    # Approximate width of the filter system ( ~10 nm):
    if method == 'explicit':
        Tr_max = np.max(transmission)
        dlmbda = np.abs(np.trapz(transmission, lmbda)/Tr_max) # nm 

        Flux_coeff = K_Jcounts/(S_px * solid_angle  * timewindow * Tr_max * dlmbda)
        return Flux_coeff

    elif method == 'implicit':
        K = (S_px * solid_angle  * timewindow)/K_Jcounts

        counts = K * np.trapz(
                transmission[np.newaxis,:]\
                *planck(lmbda[np.newaxis,:], tele[:,np.newaxis]),
                dx=np.diff(lmbda*1e-9)[0], axis=1)
        return counts


