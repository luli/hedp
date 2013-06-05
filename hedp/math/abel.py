#!/usr/bin/python
# -*- coding: utf-8 -*-
# hedp module
# Roman Yurchak, Laboratoire LULI, 11.2012

import time
import sys

import numpy as np
from scipy.integrate import simps
from hedp.math.derivative import gradient



def iabel(r, fr):
    """
    Returns inverse Abel transform. See `abel` for input parameters.

    """
    return abel(fr, r, inverse=True)

def abel(r, fr, inverse=False):
    """
    Returns the direct or inverse Abel transform of a function
    sampled at discrete points.

    This algorithm does a direct computation of the Abel transform:
      * integration near the singular value is done analytically
      * integration further from the singular value with the trapezoidal
        rule.

    There may be better/more general ways to do this especially regarding
    resilience to noise. See:
      * One-dimensional tomography: a comparison of Abel, onion-peeling, and
      filtered backprojection methods. Cameron J. Dasch
      * Reconstruction of Abel-transformable images: The Gaussian basis-set
        expansion Abel transform method. V. Dribinski
      * Using the Hankel-Fourier transform.
    still, this implementation has the advantage of being simple and working
    for both the inverse and the direct transform.

    Parameters
    ----------
    fr:  1d or 2d numpy array
        input array to which direct/inversed Abel transform will be applied.
        For a 2d array, the first dimension is assumed to be the z axis and
        the second the r axis.
    r:   1d array of the same length as fr.shape[-1]
        array of radius at which fr is taken.
    inverse: boolean
        If True inverse Abel transform is applied.

    Returns
    -------
    out: 1d or 2d numpy array of the same shape as fr
        with either the direct or the inverse abel transform.
    """

    assert type(fr).__name__ == 'ndarray'
    if fr.ndim == 1:
        fr = fr[np.newaxis, :]
    if inverse:
         fr = gradient(fr, r, axis=-1)
    result = np.empty(fr.shape)
    # build the integration kernel
    R, Y = np.meshgrid(r, r)
    I = R**2-Y**2
    I = np.where(I>0, I, np.nan)  # remove invalid values to avoid warnings
    I = 1./I**0.5
    if not inverse:
        I = I*R
    # integrate avoiding singular values
    for idx1 in range(len(r)-1):
        result[:, idx1] = simps(fr[:, idx1+1:]*np.tile(I[idx1, idx1+1:],
                                            (fr.shape[0],1)), r[idx1+1:])
    result[:,-1] = 0 # setting last element of the output to 0
    Yl = np.tile(r[:-1], (fr.shape[0], 1))
    Ll = np.tile(r[1:],   (fr.shape[0], 1))
    # linear interpolation of the function near the singular value used
    # to compute the integral analytically
    a = (fr[:,1:] - fr[:,:-1])/(Ll-Yl)
    b = fr[:,1:] - a*r[1:]
    # second part of the integral (computed analytically near the
    # singular values.
    dir_an = [lambda Yl, Ll, a: 0.5*(a*Yl**2*np.log(2*(Ll**2-Yl**2)**0.5+ 2*Ll)\
                - a*Yl**2*np.log(2*Yl)\
                + a*Ll*(Ll**2-Yl**2)**0.5),
              lambda Yl, Ll, b:  b*(Ll**2-Yl**2)**0.5]
    dir_an_axis = lambda Ll, a, b: 0.5*(a*Ll**2 + 2*b*Ll)
    inv_an = [lambda Yl, Ll, a: a*(Ll**2-Yl**2)**0.5,
              lambda Yl, Ll, b: b*np.log(2*(Ll**2-Yl**2)**0.5+ 2*Ll)\
                            - b*np.log(2*Yl) ]

    if not inverse:
        result[:,1:-1] += dir_an[0](Yl[:,1:], Ll[:,1:], a[:,1:])\
                + dir_an[1](Yl[:,1:], Ll[:,1:] , b[:,1:])
        result[:,0] += dir_an_axis(Ll[:,0], a[:,0], b[:,0])
        result = 2*result
    else:
        result[:,1:-1] += inv_an[0](Yl[:,1:], Ll[:,1:], a[:,1:])\
                + inv_an[1](Yl[:,1:], Ll[:,1:] , b[:,1:])
        result[:,0] += inv_an[0](Yl[:,0], Ll[:,0], a[:,0])
        result = -1.0/np.pi*result
    if fr.shape[0] == 1:
        return result[0]
    else:
        return result

def abel_analytical_step(r, fr_z, r0, r1):
    """
    Parameters
    ----------
    r:   1d array of radius at which fr is taken.
    fr_z:  1d along Z direction
        input array to which direct Abel transform will be applied.
    """

    F_1d = np.zeros(r.shape)
    mask = (r>=r0)*(r<r1)
    F_1d[mask] = 2*np.sqrt(r1**2 - r[mask]**2)
    mask = r<r0
    F_1d[mask] = 2*np.sqrt(r1**2 - r[mask]**2) - 2*np.sqrt(r0**2 - r[mask]**2)
    fr_z = fr_z.reshape((-1,1))
    return F_1d*fr_z

def sym_abel_step_1d(r, r0, r1):
    """
    Produces a symmetrical analytical transform of a 1d step
    """
    d = np.empty(r.shape)
    for sens, mask in enumerate([r>=0, r<=0]):
        d[mask] =  abel_analytical_step(np.abs(r[mask]), np.array(1), r0, r1)[0]

    return d





if __name__ == "__main__":
    # just an example to illustrate the limitations of this algorthm
    import matplotlib.pyplot as plt
    #sys.exit()

    n = 300
    r = 5e-3*np.arange(n) 
    r.sort()

    splt= plt.subplot(211)
    fr = np.zeros(n)
    fr[(r>0.6*r.max())*(r<0.8*r.max())] = 1
    fr += 1e-1*np.random.rand(n)
    plt.plot(r,fr,'k.', label='Original signal')
    F = abel(fr,r)
    #sys.exit()
    iF = iabel(F,r)
    plt.plot(r, F, 'g--', label='Abel transform')
    plt.plot(r, iF, 'r-', label='Reconstructed function')
    plt.legend()

    plt.subplot(212)
    fr = np.abs(np.sinc(5*r/r.max()))
    plt.plot(r, fr,'k.', label='Original signal')
    F = abel(fr,r)
    iF = iabel(F, r)
    plt.plot(r, F, 'g--', label='Abel transform')
    plt.plot(r, iF, 'r-', label='Reconstructed function')
    plt.legend()
    plt.show()
