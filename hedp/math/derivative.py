#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import circulant
import scipy.ndimage as nd

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

def savgol(x, y, window_size=3, order=2, deriv=0):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    This is a bruteforce method for Savitzky-Golay used for irregularely spaced data.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size > len(x):
        raise TypeError("Not enough data points!")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 1:
        raise TypeError("window_size is too small for the polynomials order")
    if order <= deriv:
        raise TypeError("The 'deriv' of the polynomial is too high.")
    assert x.shape == y.shape
    assert (np.gradient(x)[0]>0).all()  # x is monotone
    half_window = (window_size - 1)/2
    z = np.zeros(x.shape)
    N = len(z)
    for idx in range(N):
        # selecting bounds
        min_bound = idx - half_window
        max_bound = idx + half_window + 1
        # shifting the points we are using on the edges
        if min_bound < 0:
            max_bound += - min_bound
            min_bound = 0
        elif max_bound > N:
            min_bound += (N - max_bound)
            max_bound = N
        bounds = slice(min_bound, max_bound)
        cpoly = np.polyfit(x[bounds], y[bounds], order)
        if deriv>0:
            cpoly = np.polyder(cpoly, deriv)
        z[idx] = np.polyval(cpoly, x[idx])
    return z

def laplace(f, dx):
    """Compute laplace operator assyming cylindrical geometry
    Parameters:
    -----------
     - f is an array of the shape (z,r)
     - dx: float: sampling distance (must be the same in x,y) [cm]
    """
    flarge = np.zeros((f.shape[0]+2, f.shape[1]+2)) # assume that everything is zero at the edges
    flarge[1:-1,1:-1] = f
    flarge[:,0] = flarge[:,1] # f(-r,z) = f(r,z)
    df = nd.filters.laplace(flarge)*dx**2
    return df[1:-1,1:-1]


class TestDeriv(object):
    def test_sav_gol(self):
        """Check that the derivative of sin gives cos"""
        x = np.sort(np.random.rand(1000))*2*np.pi
        y = np.sin(x)
        dy =  savgol(x, y, 5, order=3, deriv=1)
        assert np.allclose(dy, np.cos(x), atol=1.0e-6)
