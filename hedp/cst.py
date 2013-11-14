#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

# List of CGS units used in hedp module
from scipy.constants import physical_constants

eV2K = physical_constants['electron volt-kelvin relationship'][0]
eV2Hz = physical_constants['electron volt-hertz relationship'][0]


R_CST = physical_constants['molar gas constant'][0]*1e7 # erg.K⁻¹.mol⁻¹
SIGMA_B_CST = physical_constants['Stefan-Boltzmann constant'][0]

