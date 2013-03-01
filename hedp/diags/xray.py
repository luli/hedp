#!/usr/bin/python
# -*- coding: utf-8 -*-
import hedp
from hedp.opacity.henke import cold_opacity
import numpy as np

def photon_deposition_depth(nu, op, I0, depth):
    """
    Computes energy deposition depth for photons
    
    Parameters:
    -----------
      - nu [ndarray] frequences [eV]
      - op [ndarray] opacities on the material the energy is going to be
        deposited in [cm⁻¹]
      - I0 [ndarray] photon intensity distribution
      - depth [ndarray] array of possible depth [cm]

    Returns:
    --------
     - Er [ndarray] amount of energy deposited at r
    """
    assert nu.shape == op.shape == I0.shape
    Nnu = len(nu)
    Ndepth = len(depth)
    # transorm everything to 2d arrays of dimension (Nnu, Ndepth)
    nu = np.repeat(nu.reshape((-1,1)), Ndepth, axis=1)
    op = np.repeat(op.reshape((-1,1)), Ndepth, axis=1)
    I0 = np.repeat(I0.reshape((-1,1)), Ndepth, axis=1)
    depth = np.repeat(depth.reshape((1,-1)), Nnu, axis=0)

    return np.trapz(I0*nu*op*np.exp(-op*depth), nu, axis=0)


def ip_sensitivity(nu, type='MS'):
    """
    Compute the IP sensitivity characteristics based on:
    Meadowcroft, A. L., Bentley, C. D., & Stott, E. N. (2000).
    Evaluation of the sensitivity and fading characteristics of
    an image plate system for x-ray diagnostics.

    Paramters:
    ----------
     - nu [ndarray]: frequency range [eV]
     - type [str]: IP type [SR, TR or MS]

     Returns:
     --------
     sens [ndarray]: IP response per incident x-ray photon mPSL/photon
    """
    if (nu>31e3).any():
        raise ValueError('Sensitivity for energies higher then 31 keV are currently not implemented.')
    if type == 'MS':
        coefs = np.array([ -6.14439758e-41,   9.61360686e-36,  -6.35400572e-31,
         2.29958071e-26,  -4.92818627e-22,   6.28509051e-18,
        -4.48290436e-14,   1.44524511e-10,  -3.67608247e-08,
         4.06557285e-04,   7.69401421e-02])
    else:
        raise ValueError('Other IP types not implemented yet')
    return np.polyval(coefs, nu)

def xray_filter_transmission(nu, element=None, thickness=None,layers=None):
    """
    Computes filter transmission for solid foils

    Parameters:
    -----------
     - nu [ndarray] photon frequency array [eV]
     - element: [str] filter element, should be in matdb
     - thickness: [float] foil thickness [cm]
     - layers: [dict] dict of layers
         {'element1': thickness1, 'element2': thickness2, ...}
    """
    wrong_args = np.array([arg is None for arg in [element, thickness, layers]])
    if wrong_args.all() or (~wrong_args).all():
        raise ValueError('You should provide either element and thickness or layers dict!')
    if layers is None:
        op = cold_opacity(element, 
                        hedp.matdb(element).solid_dens,
                        nu)
        return np.exp(-op*thickness)
    else:
        Transmissions = [xray_filter_transmission(nu, el, thick)\
                for el, thick in layers.iteritems()]
        return np.prod(np.array(Transmissions), axis=0)

def Kalpha_profile(el, nu, sigma=100):
    """
    Computes the Kalpha line profile
    Parameters:
    -----------
     - nu [ndarray] photon frequency array [eV]
     - element: [str] filter element, should be in matdb
     - sigma: [float] gaussian with in eV

     Returns:
     --------

     - profile: [ndarray] Kα line profile as a normalized gaussian
    """
    nu0 = hedp.matdb(el).spect[u'Kα1']
    return 1./(sigma*(2*np.pi)**0.5)* np.exp(-(nu-nu0)**2/(2*sigma**2))

def ff_profile(nu, tele):
    """
    Computes the Thermal bremsstrahlung emission

    Parameters:
    -----------
     - nu [ndarray] photon frequency array [eV]
     - tele: [float] temperature [eV]

     Returns:
     --------

     - profile: [ndarray] Kα line profile as a normalized gaussian
    """
    return np.exp(-nu/tele)/(1+tele)


if __name__ == '__main__':
    import matplotlib.pyplot as plt



