#!/usr/bin/python
# -*- coding: utf-8 -*-
import time

import numpy as np
from hedp.pp.abel import abel, abel_analytical_step, iabel
from hedp.maths.derivative import gradient
from numpy.testing import assert_, assert_array_almost_equal


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

    def test_step(self):
        n = 512
        fr = np.zeros((n/16,n))
        r = 2e-3*np.arange(n)
        r0, r1 = 0.2, 0.5
        fr[:,(r>r0)*(r<r1)] = 3
        Fn = abel(fr, r)
        Fn_a = abel_analytical_step(3*np.ones(fr.shape[0]),r, r0,r1)
        err =  np.abs(Fn - Fn_a)





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
