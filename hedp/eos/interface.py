#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

class EosBase:
    def __init__(self):
        pass

def EosInterface(backend='gamma', specie='total', **args):
        """
        Compute the EoS equilibrium state. CGS units are used.

        Parameters:
        ----------
          - backend: the backend that should be used. Current
            possibilities are ['gamma', 'eospac', 'FEOS']
          - specie: [tot, ele, ion, ioncc, rad]
          - **args: implementation specific arguments

        Returns
        -------
          and 
        """

        # First we are determing the EoS mode
        rho = args['rho']
        temp = args['rho']
        eint = args['eint']
        pres = args['pres']

        if rho is not None and temp is not None:
            eos_mode = 'DT'
            self.update({'rho': rho, 'temp': temp})
        elif rho is not None and eint is not None:
            eos_mode = 'DU'
            self.update({'rho': rho, 'eint': eint})
        else:
            raise ValueError('Not enough varibles given or unsupported EoS mode!')

        if backend == 'gamma':
            from .gamma import GammaEos
            res = GammaEos(rho=rho, temp=temp, eint=eint, pres=pres,
                                    specie=specie, eos_mode=eos_mode, **args)
        elif backend == 'eospac':
            from .eospac_wrapper import EospacEos
            res = EospacEos(rho=rho, temp=temp, eint=eint, pres=pres,
                                    specie=specie, eos_mode=eos_mode, **args)
        #self.update(res)
        return



