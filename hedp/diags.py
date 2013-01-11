#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import cumtrapz

from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

# everything is in SI units

C_CST = 3.0e8
H_CST = 6.626068e-34
KB_CST = 1.3806503e-23
NA_CST = 6.02214129e23
def SOP_diag(d, lmbda):
    Te = d['tele']

    plank  =  2*H_CST*C_CST**2/\
            ( lmbda**5*( np.exp(H_CST*C_CST / (lmbda*KB_CST*Te)) - 1.  ))

    # this is the emission intensity
    return plank

def kramer_unsoldt_opacity(dens, Z, A, Zbar, Te, lmbda):
    """
    Computes the  Kramer-Unsoldt opacity [Zel’dovich & Raizer 1967 p 27]
    cf. Thèse de Tommaso
    
    Parameters:
    -----------
     dens: [ndarray] density in (g.cm⁻³)
     Z: [ndarray] atomic number 
     A: [ndarray] atomic mass
     Zbar: [ndarray] ionization
     Te: [ndarray] electron temperature (eV)
     lmdba: [ndarray] wavelength (nm)

    Returns:
    --------
     out: [ndarray] of the same shape as input containing the opacity [cm⁻¹]
    """
                                          # check sign here
    Ibar = 10.4*Z**(4./3) * (Zbar/Z)**2 / (1 - Zbar/Z)**(2./3)
    Ibar = np.fmax(Ibar, 6.0)
    y = 1240./(lmbda * Te)
    y1 = Ibar / Te
    Ni = dens * NA_CST / A
    #print Ibar, y, y1, Ni
    return np.fmax(7.13e-16* Ni * (Zbar + 1)**2 * np.exp(y - y1) / (Te**2*y**3), 1e-16)

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

