#!/usr/bin/python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np
print np.get_include()

setup(name='hedp',
      version='0.1',
      description='Toolkit for HEDP experiments analysis and postprocessing of related radiative-hydrodynamic simulations',
      author='Roman Yurchak',
      author_email='roman.yurchak@polytechnique.edu',
      packages=find_packages(),
      ext_modules = cythonize('hedp/lib/selectors.pyx'),
      include_dirs=[np.get_include()],
      #extra_compile_args=['-O3']
     )

