#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import cumtrapz

from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy import constants as cst
from scipy.constants import physical_constants

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


def sop_calibration_berenice(lmbda, F, magnification,  transmission, detectorsize,
                                    slitwidth, sweepspeed):
    """
    Compute the calibration for a streaked self emission system.

    This function calculates the number of electrons ejected by the
    photocathode in a SOP configuration.

    Parameters:
    -----------
      - lmbda [ndarray]: wavelenght (nm) array
      - F  [float]:  F number of the first lens
      - magnification [float]:    Magnification of the optical system
      - transmission  [ndarray]:    Total transmission of the optical system, including the
                      filter (same shape as the nu) 
      - detectorsize  [float] Size of a pixel in spatial direction in μm
      - slitwidth  [float]   Size of the slit in px
      - sweepspeed        Sweep speed of the streak [100ns, 50ns, etc]in pixel/ns

    Returns:
    --------
      - Flux_norm [float]: photon flux (W.m⁻².sr⁻¹.nm⁻¹.counts⁻¹)

    """
    solid_angle = np.pi/(4*F**2) # solid angle of the first optics in sr
    # Données Tommaso [counts/Joule] Streak S20 (Il faut le verifier)
    K_counts2J = 6.6434e18
    # Surface on the target for 1px (m^2)
    S_px = (detectorsize*1e-6)**2
    tr_itp = interp1d(lmbda[::-1], transmission[::-1])
    # Transmission at 420 nm
    Tr_420 = tr_itp(420)
    # Time spend on each pxl:
    # Streak slit 100um = 8px pour calibre :
    # This remains approximately true for this polar experiment
    #slitwidth = 10 # px
    t_px = slitwidth * sweepspeed/1024.
    # Approximate width of the filter system ( ~10 nm):
    dlmbda = np.abs(np.trapz(transmission[::-1], lmbda[::-1])/Tr_420) # nm 

    #print S_px, solid_angle, t_px, Tr_420 #K_counts2J, dlmbda

    Flux_coeff = 1./(S_px * solid_angle  * t_px * Tr_420 * K_counts2J * dlmbda)
    return Flux_coeff






