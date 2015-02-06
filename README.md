#   HEDP module

## Overview

General purpose module to analyse HEDP relevant experiments and hydrodynamic simulations.


## Installation

    python setup.py develop --user


## Dependencies
   The use of this module requires `python-2.7` with the usual scientific python modules such as `numpy`, `scipy`, `matplotlib`, `cython`, `pandas`.

   The following modules are also strongly recommended
 - opacplot2  ( https://github.com/rth/opacplot2)
 - PyEOSPAC (https://github.com/luli/pyeospac)

## List of features
  
####   File formats `hedp.io`

   - parser for the Andor `.sif` image files
   - parser for the Hamamatsu streak camera `.img` files

#### Equation of state (EoS) and opacities
   - Kramer-Unsoldt opacity model
   - generation of a database with cold henke opacities
   - Thomas Fermi pressure ionization.
   - Calculation of Planck and Rosseland (mutigroup) means
   - Automatic group selection for multigroup opacities
   - General interpolators intended for visualisation for the EoS and opacity tables (requires [opacplot](https://github.com/rth/opacplot2) and [pyeospac](https://github.com/luli/pyeospac) modules).


####  Basic mathematical operators `hedp.math`
   - Gradient for a non-informally sampled 1D or 2D data
   - Savitzky-Golay filter 
   - Integrators for the super-gaussian functions
   - Direct and inverse Abel transforms. The integration is carried out numerically with a semi-analytical handling of the singularity.


#### Plasma physics `hedp.plasma_physics`
   Defines a few useful quantities  in plasma physics:  the critical density, Coulomb logarithm, electron-ion collision rates, inverse Bremsstrahlung coefficient,  isentropic sound speed, Spitzer conductivity.


#### Visualization  `hedp.viz`
   - Metric formatter for `matplotlib`
         
#### Diagnostics  `hedp.diag`
   - intensity calibration for the self-emission GOI and SOP
   - IP sensitivity curves for x-rays
   

#### Post-processing  `hedp.pp`
   - Calculation of synthetic radiographs from 2D axis-symmetrical hydrodynamic simulation with the Abel transform 
     
    

## Test suite

[![Build Status](https://travis-ci.org/luli/hedp.svg?branch=master)](https://travis-ci.org/luli/hedp)
[![Coverage Status](https://coveralls.io/repos/luli/hedp/badge.svg?branch=master)](https://coveralls.io/r/luli/hedp?branch=master)

