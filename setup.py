#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012,
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

from setuptools import setup, find_packages, Extension
from setuptools import find_packages
from Cython.Distutils import build_ext
#from distutils.core import setup
#from distutils.extension import Extension
import numpy as np
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True


# Optional path to find the the GNU scientific library (GSL)

INCLUDE_GSL = None #  "/usr/include"
LIB_GSL = None #  "/usr/lib64"

ext_modules=[
    Extension("hedp.lib.integrators",
             ["hedp/lib/integrators.pyx"],),
    Extension("hedp.lib.selectors",
             ["hedp/lib/selectors.pyx"],),
]

if INCLUDE_GSL:
    ext_modules.append(Extension("hedp.lib.multigroup",
             ["hedp/lib/multigroup.pyx"],
             extra_compile_args=['-O3', '-fopenmp', '-march=native', '-Wall'],
             extra_link_args=['-O3', '-fopenmp', '-march=native'],
             libraries=['gsl', 'gslcblas'],
             library_dirs=[LIB_GSL],
             ))

setup(name='hedp',
      version='0.1',
      description='Toolkit for HEDP experiments analysis and postprocessing of related radiative-hydrodynamic simulations',
      author='Roman Yurchak',
      author_email='rth@crans.org',
      packages=find_packages(),
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      include_dirs=[np.get_include(), INCLUDE_GSL],
      package_data={'hedp': ['hedp/tests/data/*', 'data/db']},
      test_suite="hedp.tests.run"
     )

