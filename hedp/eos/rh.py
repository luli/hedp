#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.

from .interface import EosInterface
import numpy as np
from scipy.optimize import root
from pandas import DataFrame
from copy import deepcopy

def _gen_eos_args(state_i, eos_pars):
    """ Generate a dict that we can pass to EosInterface """
    pars = dict( state_i.items() + eos_pars.items())
    for key in ['u_s', 'u_2']:
        if key in pars:
            del pars[key]
    return pars

class RankineHugoniot:
    def __init__(self, state0, state1, backend='gamma', eos_pars={},
            root_opts={'method': 'hybr'}, check=True, **args):
        """
        Compute the hydrodynamic Rankine-Hugoniot relationships
        with different equations of state

        Parameters:
        -----------
          - state0: initial state
          - state1: shocked state
          - backend: eos backend
          - opt_method: optimization method to pass to scipy's minimize
          - eos_pars: EoS backend specific parameters
        """
        # Just making sure there are no weird side-effects anywhere...
        state0 = deepcopy(state0)
        state1 = deepcopy(state1)
        # get all variables for state 0
        state0 = EosInterface(backend=backend,
                **_gen_eos_args(state0, eos_pars))
        if 'rho' in state1:
            state1, u_s, u_p = self._mode_rho2(state0, state1, 
                    backend, eos_pars, root_opts)

        #self.check_consistency(state0, state1, u_s=u_s, u_p=u_p, 
        out = {'u_s': u_s, 'u_p': u_p}
        for state_idx, state in enumerate([state0, state1]):
            for key in state:
                out[key+str(state_idx)] = state[key]
        self.res = DataFrame(out)
        return

    @classmethod
    def solve(cls, *pargs, **vargs):
        """Solve Rankine Hugoniot conditions. See RankineHugoniot class for parameters.
        """
        x = cls(*pargs, **vargs)
        return x.res

    @staticmethod
    def _mode_rho2(state0, state1, backend, eos_pars, root_opts):
        """
        Implicitly solve the Rankine-Hugoniot relationship given a rho2
        """
        # First we get the internal energy and pressure from
        # ε₁ - ε₀ = ½(p₁ + p₀) (1/ρ₀ - 1/ρ₁)
        # + the EoS
        # the normalized version of the previous expression is
        # ε₁/ε₀ - 1 = ½(p₁/ε₀ + p₀/ε₀) (1/ρ₀ - 1/ρ₁)
        rho0, pres0, eint0 = [state0[key] for key in ['rho', 'pres', 'eint']]
        rho1 = state1['rho']

        def objfunc_ener(x, rho0, pres0, eint0, rho1, backend):
            """x = eint1/eint0 in order to condition the problem
            """
            cstate1 = EosInterface(backend=backend, rho=rho1, eint=x*eint0,
                            **eos_pars)
            pres1= cstate1['pres']
            err =  (x - 1.0) - 0.5*((pres0 + pres1)/eint0)*(1/rho0 - 1/rho1)
            return err

        x0 = 1.5*rho1**0 # just use 1 with appropriate shape as initial state
        sol = root(objfunc_ener, x0, 
                args=(rho0, pres0, eint0, rho1, backend),
                method=root_opts['method'])

        eint1 = sol.x*eint0
        mask_bad = (sol.x<1.0)+(sol.x==x0) # this means we were not able to find a solution
        rho1[mask_bad] = np.nan
        eint1[mask_bad] = np.nan

        state1 = EosInterface(backend=backend, rho=rho1, eint=eint1,
                        **eos_pars)
        # if len 1 array, convert to scalar
        if len(state1['eint']) == 1:
            for key in state1:
                if key != 'rho':
                    state1[key]  = state1[key][0]
        pres1 = state1['pres']

        # Now computing shock velocities and particle velocities. 
        u_s_2 = rho1*(pres1 - pres0)/(rho0*(rho1 - rho0))
        u_s = u_s_2**0.5
        u_p = (1 - rho0/rho1)*u_s
        return state1, u_s, u_p

