#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

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

def compute_spectra(pars, nu, bl_el='Cl'):
    a, b, dkalpha, tele = pars
    spectra = a*Kalpha_profile(bl_el, nu, dkalpha) + b*ff_profile(nu, tele)
    spectra /= np.trapz(spectra, nu)
    return spectra

class StepsIP(object):

    def __init__(self, mat, thick, transm, nu=np.linspace(10,20e3,1000),
            filters={}):
        """
        Compute the Xray spectra from steps transmissions
        """
        assert len(mat) == len(thick)
        assert len(mat) == len(transm)
        self.mat = mat
        self.thick = thick
        self.exp_tr = transm
        self.nu = nu
        self.filters = filters
        self.sp_sens = self._get_opacity()
        self.estimate_spectra()

    def _get_opacity(self):
        sp_tr = np.empty((len(self.mat), len(self.nu)))
        sp_tr_filters = np.ones(self.nu.shape)
        for el, thick in self.filters.iteritems():
            sp_tr_filters *= np.exp(-cold_opacity(el, nu=self.nu)*thick)
        ip_sens = ip_sensitivity(self.nu)
        for k, el in enumerate(self.mat):
            sp_tr[k] = np.exp(-cold_opacity(el, nu=self.nu)*self.thick[k])\
                    * sp_tr_filters * ip_sens
        return sp_tr

    def estimate_spectra(self):
        from scipy.odr import odrpack as odr
        def ofunc(pars, x, sp_sens, nu, el):
            a, b, dkalpha, tele = pars
            spectra = a*Kalpha_profile(el, nu, dkalpha) + b*ff_profile(nu, tele)
            spectra /= np.trapz(spectra, nu)

            spectra = np.tile(spectra, (sp_sens.shape[0], 1))
            comp_tr = np.trapz(sp_sens*spectra, nu, axis=-1)
            return comp_tr


        def ofunc2(pars, thick, nu, el):
            spectra= compute_spectra(pars, nu)

            sp_tr_filters = np.ones(nu.shape)
            for filt_el, filt_thick in self.filters.iteritems():
                sp_tr_filters *= np.exp(-cold_opacity(filt_el, nu=nu)*filt_thick)
            ip_sens = ip_sensitivity(nu)
            comp_tr = []
            for this_thick in thick:
                sp_sens = np.exp(-cold_opacity(el, nu=nu)*this_thick)\
                        * sp_tr_filters * ip_sens * spectra
                comp_tr.append(np.trapz(sp_sens, nu, axis=0))

            return np.array(comp_tr).reshape((1,-1))
            #return ((comp_tr - exp_tr)**2).sum()
        beta0=[1,1,100,1000]
        my_data = odr.RealData(self.thick, self.exp_tr, sy=0.05*np.ones(self.exp_tr.shape))
        my_model = odr.Model(ofunc2,
                extra_args=(self.nu, 'polystyrene')
                )
        my_odr = odr.ODR(my_data,my_model, beta0=beta0)
        my_odr.set_job(fit_type=2)
        my_odr.set_iprint(final=2,iter=1, iter_step=1)
        fit = my_odr.run()
        self.beta = fit.beta#[::-1]
        self.sdbeta = fit.sd_beta#[::-1]

        print ofunc2(self.beta, self.thick, self.nu, 'polystyrene')
        print self.exp_tr

        print '\n'
        print beta0
        print self.beta

        #print ofunc([1,1,100,1000], self.sp_sens, self.exp_tr, self.nu, 'Cl')




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ax = [plt.subplot(2,1,idx+1) for idx in range(2)]
    steps_psl = np.array([0.02, 0.04, 0.06, 0.08, 0.12, 0.27,  0.27, 0.27, 0.27, 0.30])
    steps_tr = 1 - steps_psl/steps_psl.max()
    steps_tr = steps_tr[:5]
    steps_mat = ['mylar']*5
    steps_thick = np.array([25,50,100, 200, 400])*1e-4
    steps = StepsIP(steps_mat, steps_thick, steps_tr,
            filters={'PVDC': 37.5e-4, 'mylar': 100e-4})
    for k in range(len(steps.mat)):
        ax[0].plot(steps.nu, steps.sp_sens[k], label='{0} um'.format(steps.thick[k]*1e4))
    ax[0].legend()
    ax[0].set_ylim(0, 0.3)
    beta0 = [1,1,100,1000] 
    ax[1].plot(steps.nu, compute_spectra(steps.beta, steps.nu))
    plt.savefig('/tmp/test.png', bbox_inches='tight')



