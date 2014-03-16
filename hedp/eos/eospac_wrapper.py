#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.
import eospac

from ..cst import R_CST

class EospacEos(dict):
    def __init__(self, rho=None, temp=None, eint=None, pres=None,
                    specie='total', eos_mode='DT', game=-1, gamc=1.66, abar=1.0):
        if game<1:
            raise ValueError('we should have game > 1')
        if gamc<0:
            raise ValueError('we should have gamc > 0')
        if eos_mode == "DT":
            self.update({'rho': rho, 'temp': temp})
        elif eos_mode == "DU":
            self.update({'rho': rho, 'eint': eint})
        else:
            raise ValueError("eos_mode not implemented! It should be in 'DT', 'DU'.")
        self.update(res)
        self['cs'] = self._get_cs(self['rho'], self['pres'], gamc)
        return
