#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import sys

import numpy as np
from scipy.integrate import simps
from scipy.linalg import circulant
from numpy.testing import assert_, assert_array_almost_equal


def gradient(f, x=None, dx=1, axis=-1):
    """
    Return the gradient of 1 or 2-dimensional array.
    The gradient is computed using central differences in the interior
    and first differences at the boundaries. 
    Irregular sampling is supported (it isn't supported by np.gradient)

    Parameters
    ----------
    f: 1d or 2d numpy array
       Input array.
    x: array_like, optional
       Points where the function f is evaluated. It must be of the same
       length as f.shape[axis].
       If None, regular sampling is assumed (see dx)
    dx: float, optional
       If `x` is None, spacing given by `dx` is assumed. Default is 1.
    axis: int, optional
       The axis along which the difference is taken.

    Returns
    -------
    out: array_like
        Returns the gradient along the given axis. 

    To do:
      implement smooth noise-robust differentiators for use on experimental data.
      http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/
    """
    if x is None:
        x = np.arange(f.shape[axis])*dx
    else:
        assert x.shape[0] == f.shape[axis]
    I = np.zeros(f.shape[axis])
    I[:2] = np.array([0,-1])
    I[-1] = 1
    I = circulant(I)
    I[0,0] = -1
    I[-1,-1] = 1
    I[0,-1] = 0
    I[-1,0] = 0 
    H = np.zeros((f.shape[axis],1))
    H[1:-1,0] = x[2:]-x[:-2]
    H[0] = x[1] - x[0]
    H[-1] = x[-1] - x[-2]
    if axis==0:
        return np.dot(I/H, f)
    else:
        return np.dot(I/H, f.T).T

def iabel(fr, r=None):
    """
    Returns inverse Abel transform. See `abel` for input parameters.

    """
    return abel(fr, r, inverse=True)

def abel(fr, r=None, inverse=False):
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
    r:   1d array of the same length as fr.shape[-1] [optional]
        array of radius at which fr is taken. If None the sampling is
        assumed to be range(fr.shape[-1]).
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
    if r is None:
        r = np.arange(fr.shape[1])
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


class TestAbel:
    """
    To run tests run
    nosetests -s abel.py
    """
    def test_timeit(self):
        print '\nTimeit:'
        n_list = [64, 128, 256]
        for el in n_list:
            x = np.random.randn(el,el)
            t = time.time()
            abel(x)
            print '{0:>3d}x{0:>3d}  {1:>5.3f} s'.format(el, time.time() - t)

    def test_zeros(self):
        n = 64
        x = np.zeros((n,n))
        assert (abel(x, inverse=False)==0).all()

    def test_inversion(self):
        n = 256
        fr = np.zeros((n,n)) 
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
        idx = ((X**2+Y**2)<(0.6*n)**2) *  ((X**2+Y**2)>(0.4*n)**2)
        idx = idx + ((X>0.8)*(X<0.85)*(Y>0.8)*(Y<0.85))
        fr[idx] = 1
        fr_hat = iabel(abel(fr))
        err_cont = np.abs(fr - fr_hat)
        err_assym = np.abs(fr_hat[:,0] - fr_hat[0,:])
        r = 5e-3*np.arange(n) 
        r += 3e-3*np.random.rand(n)
        r.sort()
        fr = np.abs(np.sinc(5*r/r.max()))
        err_sin = np.abs(iabel(abel(fr, r), r) - fr)


        print '\nConsecutive iabel->abel error '
        print '               p25        p50        p75         p95         max'
        for key, err in  dict(asymm=err_assym, continuous=err_cont,
                                                    sin=err_sin).iteritems():
            print '{5:13}{0:.3e}  {1:.3e}  {2:.3e}   {3:.3e}  {4:.3e}'.format(
                    np.percentile(err, 25), np.percentile(err, 50),
                    np.percentile(err, 75), np.percentile(err, 95),
                    err.max(), key)
    def test_gradient(self):
        N = 5500
        x = np.arange(N)/(0.1*N)
        f = np.sin(x)
        err = np.abs(gradient(f, x) - np.cos(x)).sum()/len(x)
        assert err < 1e-5
        print 'Gradient err for sin: ', err



if __name__ == "__main__":
    # just an example to illustrate the limitations of this algorthm
    import pylab as plt

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
